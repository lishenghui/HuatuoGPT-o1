
# In[1]:


import re
import wandb
import torch
from trl import SFTTrainer, SFTConfig
from unsloth import is_bfloat16_supported
from transformers import TrainingArguments
from paaa.ft_datasets.catalog import HeroThreeRolesDataset

wandb.init(name="test", settings=wandb.Settings(_disable_stats=True,  _disable_meta=True))
medqa_dataset = HeroThreeRolesDataset(num_medqa_clients=1)
dataset = medqa_dataset.client_datasets[0].train_set


# In[2]:


from unsloth import FastLanguageModel
import torch
max_seq_length = 2048 # Choose any! We auto support RoPE Scaling internally!
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = False # Use 4bit quantization to reduce memory usage. Can be False.

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Llama-3.2-1B-Instruct",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)


# In[6]:


def add_eos_token(examples):
    texts = examples["text"]
    EOS_TOKEN = tokenizer.eos_token
    return {"text": [re.sub(r"####[^\n]*\n", "", text) + EOS_TOKEN for text in texts]}

dataset = dataset.map(add_eos_token, batched=True)



# In[6]:

print(f"Dataset size: {len(dataset)}")
dataset[0]

# In[5]:


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


# In[6]:

# class FLSFTTrainer(SFTTrainer):
    # def __init__(self, *args, **kwargs):
        # super().__init__(*args, **kwargs)
        # self.num_reset = 0
    # def _maybe_log_save_evaluate(
    #     self, tr_loss, grad_norm, model, trial, epoch, ignore_keys_for_eval, start_time, learning_rate=None
    # ):
    #     if self.control.should_save:
    #         self.num_reset += 1
    #         self.args.max_steps += num_local_steps
    #         for param_group in self.optimizer.param_groups:
    #             old_lr = param_group['lr']
    #             param_group['lr'] = 5e-4 * self.num_reset
    #         self.lr_scheduler.base_lrs = [param_group['lr'] for param_group in self.optimizer.param_groups]
    #         print(f"Learning rate adjusted from {old_lr} to {param_group['lr']} at step {self.state.global_step}")
    #     super()._maybe_log_save_evaluate(
    #         tr_loss, grad_norm, model, trial, epoch, ignore_keys_for_eval, start_time, learning_rate
    #     )

# num_local_steps = 100 # Set this to the number of steps you want to train for.
num_total_steps = 1000 # Set this to the total number of steps you want to train for.
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    args = SFTConfig(
        per_device_train_batch_size = 16,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        dataset_num_proc = 4,
        num_train_epochs = 5, # Set this for 1 full training run.
        # max_steps = num_total_steps,
        learning_rate = 2e-4,
        max_seq_length = max_seq_length,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = 5,
        dataset_text_field = "text",
        optim = "adamw_8bit",
        packing = False, # Can make training 5x faster for short sequences.
        weight_decay = 0.01,
        lr_scheduler_type = "cosine_with_min_lr",
        lr_scheduler_kwargs= {
            "min_lr": 5e-5,  # Set this to the minimum learning rate you want to use.
        },
        seed = 3407,
        output_dir = "llama_1B_MetaMathQA_lr_new",
        save_strategy = "steps",  # <- Save checkpoint after each epoch
        save_steps= 1000,  # <- Save checkpoint every 100 steps
        report_to="wandb",
        save_total_limit = 5,      # Optional: limit total checkpoints to save disk space
    ),
)

import os
os.environ['UNSLOTH_RETURN_LOGITS'] = '1'


# In[7]:


# @title Show current memory stats
gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")



trainer.train()