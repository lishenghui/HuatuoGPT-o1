import os
import json
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    HfArgumentParser,
    DataCollatorForSeq2Seq,
    set_seed,
)
from datasets import load_dataset, Dataset

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

@dataclass
class ModelArguments:
    model_path: str = field(metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"})

@dataclass
class DataTrainingArguments:
    data_path: str = field(metadata={"help": "Path to the training dataset (in JSON format)"})
    max_seq_len: int = field(default=8192)

@dataclass
class MyTrainingArguments(TrainingArguments):
    output_dir: str = field(default="./ckpts/sft_stage1")
    overwrite_output_dir: bool = field(default=True)
    logging_dir: str = field(default="./train_logs/sft_stage1")
    logging_strategy: str = field(default="steps")
    logging_steps: int = field(default=10)
    per_device_train_batch_size: int = field(default=6)
    num_train_epochs: int = field(default=8)
    learning_rate: float = field(default=5e-6)
    weight_decay: float = field(default=0.1)
    warmup_ratio: float = field(default=0.05)
    save_strategy: str = field(default="epoch")
    save_total_limit: int = field(default=3)
    bf16: bool = field(default=True)
    gradient_accumulation_steps: int = field(default=4)  # 减少梯度累积
    dataloader_num_workers: int = field(default=0)
    dataloader_pin_memory: bool = field(default=True)
    remove_unused_columns: bool = field(default=False)
    group_by_length: bool = field(default=True)

def format_batch(batch, tokenizer):
    prompts = []
    for q, cot, resp in zip(batch['Question'], batch['Complex_CoT'], batch['Response']):
        a = f"## Thinking\n\n{cot}\n\n## Final Response\n\n{resp}"
        
        messages = [
            {"role": "user", "content": q},
            {"role": "assistant", "content": a}
        ]
        
        prompt = tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=False
        )
        prompts.append(prompt)
    
    return {"formatted_text": prompts}


def tokenize_batch(batch, tokenizer, max_seq_len):
    input_encodings = []
    label_encodings = []
    
    for text in batch["formatted_text"]:
        messages_user = [{"role": "user", "content": text.split("## Thinking")[0].strip()}]
        query = tokenizer.apply_chat_template(messages_user, tokenize=False, add_generation_prompt=True)
        
        # 编码
        full_ids = tokenizer.encode(text, max_length=max_seq_len, truncation=True)
        query_ids = tokenizer.encode(query, add_special_tokens=False)
        
        # 构建 labels，只对回复部分计算 loss
        labels = [-100] * len(query_ids) + full_ids[len(query_ids):]
        
        input_encodings.append(full_ids)
        label_encodings.append(labels[:len(full_ids)])  # 确保长度一致
    
    return {
        "input_ids": input_encodings,
        "labels": label_encodings
    }
    
def load_train_dataset(data_path, tokenizer, max_seq_len):
    """加载和预处理数据集 - 批处理版本"""
    logger.info(f"Loading dataset from {data_path}")
    
    with open(data_path, 'r') as f:
        data = json.load(f)
    
    logger.info(f"Loaded {len(data)} examples")
    
    dataset = Dataset.from_list(data)
    
    logger.info("Formatting text...")
    dataset = dataset.map(
        lambda batch: format_batch(batch, tokenizer),
        batched=True,
        batch_size=1000,
        remove_columns=dataset.column_names,
        num_proc=1,  # 因为涉及tokenizer，建议单进程
        desc="Formatting"
    )
    logger.info("Tokenizing...")
    dataset = dataset.map(
        lambda batch: tokenize_batch(batch, tokenizer, max_seq_len),
        batched=True,
        batch_size=100,  # 较小的batch_size避免内存问题
        remove_columns=["formatted_text"],
        desc="Tokenizing"
    )
    
    logger.info(f"Dataset preprocessing completed. Dataset size: {len(dataset)}")
    return dataset

def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, MyTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # 设置日志
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    set_seed(training_args.seed)

    logger.info("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_path, 
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    
    tokenizer.pad_token = tokenizer.eos_token
    
    # 启用梯度检查点
    model.gradient_checkpointing_enable()
    
    logger.info("Loading and preprocessing dataset...")
    train_dataset = load_train_dataset(
        data_args.data_path, 
        tokenizer, 
        data_args.max_seq_len
    )

    # 数据收集器
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        return_tensors="pt",
        pad_to_multiple_of=8,
    )

    # 训练信息
    total_batch_size = (
        training_args.per_device_train_batch_size
        * training_args.gradient_accumulation_steps
        * training_args.world_size
    )
    logger.info(f"Total batch size: {total_batch_size}")
    logger.info(f"Number of training examples: {len(train_dataset)}")
    logger.info(f"Number of training steps per epoch: {len(train_dataset) // total_batch_size}")

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    logger.info("Starting training...")
    trainer.train()
    
    logger.info("Saving model...")
    trainer.save_model(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)
    
    logger.info("Training completed!")

if __name__ == '__main__':
    main()