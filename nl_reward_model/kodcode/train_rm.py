import os
import json
import random

import torch
import wandb
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset

import deepspeed
from deepspeed.accelerator import get_accelerator

from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
)
from peft import LoraConfig, get_peft_model

deepspeed.ops.op_builder.CPUAdamBuilder().load()


class QACDataset(Dataset):
    """Dataset for Question-Answer-Critique format with JSON output."""
    
    def __init__(self, json_path, tokenizer, prompt_max_length, max_length):
        """
        Initialize the dataset.
        
        Args:
            json_path: Path to the JSON data file
            tokenizer: HuggingFace tokenizer
            prompt_max_length: Maximum length for the prompt
            max_length: Maximum total sequence length
        """
        with open(json_path, "r") as json_f:
            self.json_data = json.load(json_f)
        self.prompt_max_length = prompt_max_length
        self.max_length = max_length
        self.answer_max_length = max_length - prompt_max_length - 1
        self.pad_token_id = tokenizer.pad_token_id
        self.eos_token = tokenizer.eos_token
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.json_data)

    def __getitem__(self, ind):
        item = self.json_data[ind]

        question = item["question"]
        solution = item["solution"]
        code_feedback = item.get("code_feedback", "")
        wrong_code = item.get("wrong_code", [])
        improvement_code = item.get("improvement_code", [])

        prompt = f'''Please analyze the following programming problem and solution:

Problem:
{question}

Submitted Solution:
{solution}

---
**Instructions:**
1. Evaluate the code solution based on:
   - **Correctness**: Does it solve the problem without bugs?
   - **Efficiency**: Is the time and space complexity optimal?
   - **Readability**: Is the code clean and well-structured?
   - **Completeness**: Does it handle all edge cases?
2. Identify specific code snippets that:
   - Have errors or bugs (for wrong_code)
   - Work correctly but could be improved (for improvement_code)
3. Provide a concise paragraph summarizing the overall quality of the solution.

---
**Output Format:**
```json
{{
    "code_feedback": "Evaluate the strengths and weaknesses (if any) of the code solution, concisely written in one paragraph.",
    "wrong_code": ["Include only code snippets with errors or bugs causing test failures. Leave as an empty array if no such errors are found."],
    "improvement_code": ["Include only original code snippets in the Submitted Code that work correctly but could be improved. Leave as an empty array if no improvements are needed."]
}}
```
'''

        answer = json.dumps({
            "code_feedback": code_feedback,
            "wrong_code": wrong_code,
            "improvement_code": improvement_code
        }, indent=2)

        input_text = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"

        src_tokens = self.tokenizer.tokenize(input_text)
        if len(src_tokens) > self.prompt_max_length:
            src_tokens = src_tokens[:self.prompt_max_length]

        tgt_tokens = self.tokenizer.tokenize(answer)
        if len(tgt_tokens) > self.answer_max_length:
            tgt_tokens = tgt_tokens[:self.answer_max_length]

        tokens = src_tokens + tgt_tokens + [self.eos_token]
        assert len(tokens) <= self.max_length

        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        context_length = len(src_tokens)
        labels = [-100] * context_length + input_ids[context_length:]

        assert len(input_ids) == len(labels)

        padding_len = self.max_length - len(input_ids)
        input_ids = input_ids + [self.pad_token_id] * padding_len
        labels = labels + [-100] * padding_len
        attention_mask = [1] * (len(input_ids) - padding_len) + [0] * padding_len

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long)
        }


def prepare_datasets(dataset_file, train_file, valid_file, split_ratio=0.9, seed=42):
    """
    Split the dataset into training and validation sets.
    
    Args:
        dataset_file: Path to the full dataset
        train_file: Path to save the training dataset
        valid_file: Path to save the validation dataset
        split_ratio: Ratio for train/validation split (default: 0.9)
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (train_data, test_data)
    """
    with open(dataset_file, "r") as f:
        full_dataset = json.load(f)

    random.seed(seed)

    random.shuffle(full_dataset)

    split_idx = int(len(full_dataset) * split_ratio)
    train_data = full_dataset[:split_idx]
    test_data = full_dataset[split_idx:]

    with open(train_file, "w") as f:
        json.dump(train_data, f)

    with open(valid_file, "w") as f:
        json.dump(test_data, f)

    print(f"Dataset split complete: {len(train_data)} training samples, {len(test_data)} test samples")
    return train_data, test_data


def setup_model_and_tokenizer(model_name):
    """
    Set up the model and tokenizer.
    
    Args:
        model_name: Name or path of the pretrained model
        
    Returns:
        Tuple of (tokenizer, model)
    """
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.pad_token_id = tokenizer.eos_token_id
    
    return tokenizer, model


def apply_lora(model, r=16, lora_alpha=32, dropout=0.1):
    """
    Apply LoRA to the model.
    
    Args:
        model: The model to apply LoRA to
        r: LoRA rank
        lora_alpha: LoRA alpha parameter
        dropout: Dropout rate
        
    Returns:
        Model with LoRA applied
    """
    lora_config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=dropout,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)
    print("LoRA layers have been added.")
    return model


def get_deepspeed_config(batch_size, learning_rate=2e-5):
    """
    Create DeepSpeed configuration.
    
    Args:
        batch_size: Batch size per GPU
        learning_rate: Learning rate
        
    Returns:
        DeepSpeed configuration dictionary
    """
    return {
        "fp16": {
            "enabled": True,
            "loss_scale": 0,
            "loss_scale_window": 1000,
            "hysteresis": 2,
            "min_loss_scale": 1
        },
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": learning_rate,
                "betas": [0.9, 0.999],
                "eps": 1e-8,
                "weight_decay": 0.01
            }
        },
        "zero_optimization": {
            "stage": 3,
            "offload_optimizer": {
                "device": "cpu",
                "pin_memory": True
            },
            "offload_param": {
                "device": "cpu",
                "pin_memory": True
            },
            "overlap_comm": True,
            "contiguous_gradients": True,
            "sub_group_size": 1e9,
            "stage3_max_live_parameters": 1e9,
            "stage3_max_reuse_distance": 1e9,
            "stage3_gather_16bit_weights_on_model_save": True
        },
        "train_batch_size": 1 * batch_size * torch.cuda.device_count(),
        "gradient_accumulation_steps": 1,
        "train_micro_batch_size_per_gpu": batch_size,
        "gradient_clipping": 1.0,
        "steps_per_print": 2000,
        "wall_clock_breakdown": False
    }


def evaluate_model(model_engine, eval_dataloader, subset_size=200):
    """
    Evaluate the model on a subset of the validation data.
    
    Args:
        model_engine: DeepSpeed model engine
        eval_dataloader: Validation dataloader
        subset_size: Number of samples to evaluate
        
    Returns:
        Average evaluation loss
    """
    model_engine.eval()
    eval_losses = []
    
    # Create a fixed-size random validation dataset
    total_eval_samples = len(eval_dataloader.dataset)
    
    # Generate random indices
    random_indices = torch.randperm(total_eval_samples)[:subset_size]
    eval_subset = torch.utils.data.Subset(eval_dataloader.dataset, random_indices)
    
    # Create subset dataloader
    eval_subset_loader = DataLoader(
        eval_subset,
        batch_size=eval_dataloader.batch_size,
        shuffle=True
    )
    
    with torch.no_grad():
        for eval_step, batch in tqdm(enumerate(eval_subset_loader),
                                     total=len(eval_subset_loader),
                                     desc="Evaluating random subset"):
            input_ids = batch["input_ids"].to(model_engine.local_rank)
            attention_mask = batch["attention_mask"].to(model_engine.local_rank)
            labels = batch["labels"].to(model_engine.local_rank)
            outputs = model_engine(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            
            eval_loss = outputs.loss
            eval_losses.append(eval_loss.item())
    
    avg_eval_loss = sum(eval_losses) / len(eval_losses)
    return avg_eval_loss


def save_checkpoint(model_engine, exp_dir, epoch, step):
    """
    Save model checkpoint.
    
    Args:
        model_engine: DeepSpeed model engine
        exp_dir: Experiment directory
        epoch: Current epoch
        step: Current step
    """
    if model_engine.local_rank == 0:
        checkpoint_dir = os.path.join(exp_dir, f'{epoch}_{step}')
        os.makedirs(checkpoint_dir, exist_ok=True)
        model_engine.save_pretrained(checkpoint_dir)
        print(f"Checkpoint saved to {checkpoint_dir}")


def train(config):
    """
    Main training function.
    
    Args:
        config: Dictionary containing training configuration
    """
    wandb.init(project=config["project_name"], name=config["project_name"])
    
    train_data, test_data = prepare_datasets(
        config["dataset_file"],
        config["train_dataset_file"],
        config["valid_dataset_file"]
    )
    
    tokenizer, model = setup_model_and_tokenizer(config["model_name"])
    model = apply_lora(model)
    
    trainable_parameters = [p for p in model.parameters() if p.requires_grad]
    
    ds_config = get_deepspeed_config(config["batch_size"])
    model_engine, optimizer, _, _ = deepspeed.initialize(
        model=model,
        model_parameters=trainable_parameters,
        config=ds_config
    )
    
    train_dataset = QACDataset(
        config["train_dataset_file"], 
        tokenizer=tokenizer, 
        prompt_max_length=config["prompt_max_length"],
        max_length=config["max_length"]
    )
    train_dataloader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    
    valid_dataset = QACDataset(
        config["valid_dataset_file"], 
        tokenizer=tokenizer, 
        prompt_max_length=config["prompt_max_length"],
        max_length=config["max_length"]
    )
    valid_dataloader = DataLoader(valid_dataset, batch_size=config["batch_size"], shuffle=False)
    
    for epoch in range(config["epochs"]):
        model_engine.train()
        for step, batch in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
            input_ids = batch["input_ids"].to(model_engine.local_rank)
            attention_mask = batch["attention_mask"].to(model_engine.local_rank)
            labels = batch["labels"].to(model_engine.local_rank)
            outputs = model_engine(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            
            model_engine.backward(loss)
            model_engine.step()
            
            get_accelerator().empty_cache()
            
            wandb.log({
                "train/loss": loss.item(),
                "epoch": epoch,
                "step": step
            })
            
            if step % 400 == 0 and step > 200:
                get_accelerator().empty_cache()
                
                avg_eval_loss = evaluate_model(model_engine, valid_dataloader)
                
                wandb.log({
                    "eval/loss": avg_eval_loss,
                    "eval/samples": 200,
                    "epoch": epoch,
                    "step": step
                })
                
                get_accelerator().empty_cache()
                model_engine.train()
                save_checkpoint(model_engine, config["exp_dir"], epoch, step)


def main():
    """Main function to run the training script."""
    config = {
        "model_name": "meta-llama/Llama-3.1-8B-Instruct",
        "dataset_file": "./data/KodCode/kodcode_RM.json",
        "train_dataset_file": "./data/KodCode/kodcode_RM_train.json",
        "valid_dataset_file": "./data/KodCode/kodcode_RM_test.json",
        "batch_size": 2,
        "epochs": 3,
        "prompt_max_length": 900,
        "max_length": 1250,
        "exp_dir": "ckpt/text2grad_kodcode_RM",
        "project_name": "kodcode-RM"
    }
    
    # Create experiment directory if it doesn't exist
    os.makedirs(config["exp_dir"], exist_ok=True)
    
    # Run training
    train(config)


if __name__ == "__main__":
    main()
