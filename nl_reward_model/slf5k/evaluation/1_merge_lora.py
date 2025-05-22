import json
import torch
import os
import argparse
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    GenerationConfig
)
from tqdm import tqdm
from peft import PeftModel, LoraConfig
from torch.utils.data import DataLoader, Dataset
from dataclasses import dataclass, field
from collections import defaultdict
from trl import AutoModelForCausalLMWithValueHead

# Configuration parameters
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def parse_args():
    parser = argparse.ArgumentParser(description="Merge LoRA weights into base model")
    
    # Model paths
    parser.add_argument("--base_model", type=str, default="meta-llama/Llama-3.1-8B-Instruct",
                        help="Path to the base model")
    parser.add_argument("--lora_model", type=str, default="../ckpt/text2grad_slf5k_RM/0_4000",
                        help="Path to the LoRA model")
    parser.add_argument("--save_dir", type=str, default="../ckpt/text2grad_slf5k_RM/0_4000_merge",
                        help="Directory to save the merged model")
    
    # Additional options
    parser.add_argument("--merge_and_save", action="store_true", default=True,
                        help="Whether to merge LoRA weights and save")
    parser.add_argument("--gpu_ids", type=str, default="0",
                        help="Comma-separated list of GPU IDs to use")
    
    return parser.parse_args()


def setup_model_and_tokenizer(args):
    """
    Load model and tokenizer
    Args:
        args: Command line arguments
    """
    try:
        # Set GPU devices
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
        
        print("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            args.base_model,
            trust_remote_code=True,
            resume_download=True,
            padding_side='right'
        )

        if getattr(tokenizer, "pad_token", None) is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id

        print("Loading base model...")
        base_model = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            torch_dtype=torch.float16,
            device_map="balanced",
            trust_remote_code=True,
            resume_download=True,
            low_cpu_mem_usage=True
        )

        print("Loading LoRA weights...")
        model = PeftModel.from_pretrained(
            base_model,
            args.lora_model,
            torch_dtype=torch.float16,
            device_map="auto"
        )

        if args.merge_and_save:
            print("Merging weights...")
            model = model.merge_and_unload()

            print(f"Saving merged model to {args.save_dir}")
            model.save_pretrained(args.save_dir)
            tokenizer.save_pretrained(args.save_dir)
            print("Model and tokenizer saved successfully")

        return model, tokenizer

    except Exception as e:
        print(f"Error in setup_model_and_tokenizer: {e}")
        raise


if __name__ == "__main__":
    args = parse_args()
    setup_model_and_tokenizer(args)