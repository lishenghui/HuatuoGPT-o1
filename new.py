#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import json
import logging
import pandas as pd
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


# In[2]:


dataset = "FreedomIntelligence/medical-o1-reasoning-SFT"
model_name = "unsloth/Llama-3.2-3B-Instruct"

dataset = load_dataset(dataset, name="en", split="train")


# In[3]:


dataset[0]


# In[4]:


model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_name,
    max_seq_length = 2048,   # Context length - can be longer, but uses more memory
    load_in_4bit = False,     # 4bit uses much less memory
    load_in_8bit = False,    # A bit more accurate, uses 2x memory
    full_finetuning = False, # We have full finetuning now!
    # token = "hf_...",      # use one if using gated models
)


# In[5]:


def generate_conversation(batch):
    qs, cots, resps = batch['Question'], batch['Complex_CoT'], batch['Response']
    conversations = []
    for q, cot, resp in zip(qs, cots, resps):
        a = f"## Thinking\n\n{cot}\n\n## Final Response\n\n{resp}"
        conversations.append([
            {"role" : "user",      "content" : q},
            {"role" : "assistant", "content" : a},
        ])
    return { "conversations": conversations, }


# In[6]:


reasoning_conversations = tokenizer.apply_chat_template(
    dataset.map(generate_conversation, batched = True)["conversations"],
    tokenize = False,
)

data = pd.concat([
    pd.Series(reasoning_conversations)
])
data.name = "text"

from datasets import Dataset
combined_dataset = Dataset.from_pandas(pd.DataFrame(data))
combined_dataset = combined_dataset.shuffle(seed = 3407)


# In[7]:


# model = FastLanguageModel.get_peft_model(
#     model,
#     r = 32,           # Choose any number > 0! Suggested 8, 16, 32, 64, 128
#     target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
#                       "gate_proj", "up_proj", "down_proj",],
#     lora_alpha = 32,  # Best to choose alpha = rank or rank*2
#     lora_dropout = 0, # Supports any, but = 0 is optimized
#     bias = "none",    # Supports any, but = "none" is optimized
#     # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
#     use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
#     random_state = 3407,
#     use_rslora = False,   # We support rank stabilized LoRA
#     loftq_config = None,  # And LoftQ
# )


# In[8]:


# model, tokenizer = FastLanguageModel.from_pretrained(
#     model_name = model_name,
#     max_seq_length = 2048,   # Context length - can be longer, but uses more memory
#     load_in_4bit = False,     # 4bit uses much less memory
#     load_in_8bit = False,    # A bit more accurate, uses 2x memory
#     full_finetuning = False, # We have full finetuning now!
#     # token = "hf_...",      # use one if using gated models
# )


# In[9]:


from trl import SFTTrainer, SFTConfig
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = combined_dataset,
    eval_dataset = None, # Can set up evaluation!
    args = SFTConfig(
        dataset_text_field = "text",
        per_device_train_batch_size = 4,
        gradient_accumulation_steps = 4, # Use GA to mimic batch size!
        warmup_steps = 5,
        # num_train_epochs = 1, # Set this for 1 full training run.
        # max_steps = 30,
        num_train_epochs = 5, # Set this for 1 full training run.
        learning_rate = 2e-5, # Reduce to 2e-5 for long training runs
        logging_steps = 1,
        optim = "adamw_8bit",
        output_dir = "trained_full_funing", # Save model here
        # save_strategy = "steps", # Save model every X steps
        save_strategy = "epoch", # Save model every X steps
        # save_steps = 10, # Save model every 10 steps
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        report_to = "none", # Use this for WandB etc
    ),
)


# In[ ]:


trainer_stats = trainer.train()


# In[ ]:

if False: # Set to True to save the model
    model.save_pretrained_merged("trained_full_funing", tokenizer, save_method = "merged_16bit",)
# model.save("trained_full_funing", tokenizer, save_method = "merged_16bit",)

