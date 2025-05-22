import json
import torch
import os
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from tqdm import tqdm
from rouge import Rouge
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch
import os
import json
from torch.utils.data import DataLoader, Dataset
from transformers import Adafactor, AutoTokenizer, HfArgumentParser, pipeline, AutoModelForCausalLM, GenerationConfig
from dataclasses import dataclass, field
from collections import defaultdict
from tqdm import tqdm
from peft import LoraConfig
from trl import AutoModelForCausalLMWithValueHead

import shutil
import os
import json
from peft import LoraConfig



BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
LORA_MODEL = "../ckpt/text2grad_kodcode_RM/0_4400"
SAVE_DIR = "../ckpt/text2grad_kodcode_RM/0_4400_merge"

def setup_model_and_tokenizer(merge_and_save=True):
    """
    Load model and tokenizer
    Args:
        merge_and_save: Whether to merge LoRA weights and save the model
    """
    try:
        print("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            BASE_MODEL,
            trust_remote_code=True,
            resume_download=True,
        )

        if getattr(tokenizer, "pad_token", None) is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id

        print("Loading base model...")
        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            device_map="balanced",
            trust_remote_code=True,
            resume_download=True,
            low_cpu_mem_usage=True
        )

        print("Loading LoRA weights...")

        model = PeftModel.from_pretrained(
            base_model,
            LORA_MODEL,
            device_map="auto"
        )

        if merge_and_save:
            print("Merging weights...")
            model = model.merge_and_unload()

            print(f"Saving merged model to {SAVE_DIR}")
            model.save_pretrained(SAVE_DIR)
            tokenizer.save_pretrained(SAVE_DIR)
            print("Model and tokenizer saved successfully")

        return model, tokenizer

    except Exception as e:
        print(f"Error in setup_model_and_tokenizer: {e}")
        raise


setup_model_and_tokenizer(merge_and_save=True)