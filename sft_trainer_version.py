import os
import json
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union
from unsloth import FastLanguageModel

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    HfArgumentParser,
    set_seed,
)
from datasets import load_dataset, Dataset
from trl import SFTTrainer, SFTConfig

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

@dataclass
class ModelArguments:
    model_path: str = field(metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"})
    peft_method: Optional[str] = field(default="lora", metadata={"help": "Method for parameter-efficient fine-tuning (e.g., 'lora')"})

@dataclass
class DataTrainingArguments:
    data_path: str = field(metadata={"help": "Path to the training dataset (in JSON format)"})

@dataclass
class MySFTConfig(SFTConfig):
    output_dir: str = field(default="./ckpts/sft_stage1")
    overwrite_output_dir: bool = field(default=True)
    logging_dir: str = field(default="./train_logs/sft_stage1")
    logging_strategy: str = field(default="steps")
    logging_steps: int = field(default=10)
    per_device_train_batch_size: int = field(default=4)
    num_train_epochs: int = field(default=8)
    learning_rate: float = field(default=5e-6)
    weight_decay: float = field(default=0.1)
    warmup_ratio: float = field(default=0.05)
    save_strategy: str = field(default="epoch")
    save_total_limit: int = field(default=3)
    dataset_text_field="messages"
    bf16: bool = field(default=True)
    gradient_accumulation_steps: int = field(default=4)
    dataloader_num_workers: int = field(default=0)
    dataloader_pin_memory: bool = field(default=False)
    remove_unused_columns: bool = field(default=False)
    group_by_length: bool = field(default=True)
    max_seq_length: int = field(default=4096)
    packing: bool = field(default=True)

def load_and_format_dataset(data_path):
    """加载并格式化数据集为SFTTrainer所需格式"""
    logger.info(f"Loading dataset from {data_path}")
    
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    logger.info(f"Loaded {len(data)} examples")
    
    # 转换为messages格式
    formatted_data = []
    for item in data:
        # 构建assistant回复，包含thinking和final response
        assistant_content = f"## Thinking\n\n{item['Complex_CoT']}\n\n## Final Response\n\n{item['Response']}"
        
        formatted_item = {
            "messages": [
                {"role": "user", "content": item['Question']},
                {"role": "assistant", "content": assistant_content}
            ]
        }
        formatted_data.append(formatted_item)
    
    dataset = Dataset.from_list(formatted_data)
    logger.info(f"Dataset formatting completed. Dataset size: {len(dataset)}")
    
    return dataset

def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, MySFTConfig))
    model_args, data_args, sft_config = parser.parse_args_into_dataclasses()

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    set_seed(sft_config.seed)

    logger.info("Loading tokenizer and model...")
    if model_args.peft_method == "lora":
        from unsloth import FastLanguageModel
        import torch
        max_seq_length = 2048 # Choose any! We auto support RoPE Scaling internally!
        dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
        load_in_4bit = False # Use 4bit quantization to reduce memory usage. Can be False.

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name = model_args.model_path,
            max_seq_length = max_seq_length,
            dtype = dtype,
            load_in_4bit = load_in_4bit,
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_args.model_path, 
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )
    
    logger.info(f"peft_method: {model_args.peft_method}")
    if model_args.peft_method == "lora":
        # from peft import LoraConfig, get_peft_model
        # lora_config = LoraConfig(
        #     r=8,
        #     lora_alpha=16,
        #     lora_dropout=0.1,
        #     bias="none",
        #     task_type="CAUSAL_LM"
        # )
        # model = get_peft_model(model, lora_config)
        # model.print_trainable_parameters()
        # model.train()
        # logger.info("LoRA configuration applied.")

        model = FastLanguageModel.get_peft_model(
            model,
            r = 16, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj",],
            lora_alpha = 16,
            lora_dropout = 0, # Supports any, but = 0 is optimized
            bias = "none",    # Supports any, but = "none" is optimized
            use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
            random_state = 3407,
            use_rslora = False,  # We support rank stabilized LoRA
            loftq_config = None, # And LoftQ
        )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model.gradient_checkpointing_enable()
    
    logger.info("Loading and formatting dataset...")
    train_dataset = load_and_format_dataset(data_args.data_path)

    total_batch_size = (
        sft_config.per_device_train_batch_size
        * sft_config.gradient_accumulation_steps
        * sft_config.world_size
    )
    logger.info(f"Total batch size: {total_batch_size}")
    logger.info(f"Number of training examples: {len(train_dataset)}")
    logger.info(f"Number of training steps per epoch: {len(train_dataset) // total_batch_size}")

    trainer = SFTTrainer(
        model=model,
        # tokenizer=tokenizer,
        train_dataset=train_dataset,
        args=sft_config,
    )

    logger.info("Starting training...")
    trainer.train()
    
    logger.info("Saving model...")
    trainer.save_model(sft_config.output_dir)
    tokenizer.save_pretrained(sft_config.output_dir)
    
    logger.info("Training completed!")

if __name__ == '__main__':
    main()