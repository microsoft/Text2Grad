import os
import json
import torch
import argparse
from tqdm import tqdm
from rouge import Rouge
from collections import defaultdict
from dataclasses import dataclass, field
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    GenerationConfig,
    Adafactor, 
    HfArgumentParser, 
    pipeline
)
from peft import PeftModel, LoraConfig
from trl import AutoModelForCausalLMWithValueHead

def parse_args():
    parser = argparse.ArgumentParser(description="Merge LoRA weights into base model")
    parser.add_argument("--base_model", type=str, default="meta-llama/Llama-3.1-8B-Instruct", 
                        help="Path to base model")
    parser.add_argument("--lora_model", type=str, 
                        default="../ckpt/text2grad_ultrafeedback_RM/0_10000", 
                        help="Path to LoRA model")
    parser.add_argument("--save_dir", type=str, 
                        default="../ckpt/text2grad_ultrafeedback_RM/0_10000_merge", 
                        help="Path to save merged model")
    parser.add_argument("--gpu_ids", type=str, default="2,3,4,5,6,7", 
                        help="GPU IDs to use, comma separated")
    parser.add_argument("--merge_and_save", action="store_true", default=True,
                        help="Whether to merge LoRA weights and save")
    return parser.parse_args()


def setup_model_and_tokenizer(args):
    """
    Load model and tokenizer
    Args:
        args: Command line arguments
    """
    try:
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


def main():
    args = parse_args()
    
    # Set GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
    
    setup_model_and_tokenizer(args)


if __name__ == "__main__":
    main()