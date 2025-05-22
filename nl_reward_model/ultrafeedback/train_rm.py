import os
import json
import torch
import wandb
from tqdm import tqdm
from collections import defaultdict
from torch.utils.data import DataLoader, Dataset

import deepspeed
from deepspeed.accelerator import get_accelerator

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType

deepspeed.ops.op_builder.CPUAdamBuilder().load()


class QACDataset(Dataset):
    """Dataset for Question-Answer-Critique format with JSON output."""
    
    def __init__(self, json_path, tokenizer, prompt_max_length, max_length):
        """Initialize the dataset."""
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

        user_prompt = item["prompt"]
        assistant_response = item["response"]

        prompt = f'''Please critique the following response to a user input and provide feedback and word-level score list:

---
**User Input**:
{user_prompt}
---

---
**Generated Response**:
{assistant_response}
---

---
**Definitions:**
- **good_spans**: phrases from the response that greatly improve its quality by accurately addressing the user input, providing key information, or capturing the core intent effectively, as explained in 'textual_feedback'. Empty if none apply.
- **poor_spans**: phrases from the response that noticeably harm its quality due to inaccuracy, irrelevance, redundancy, poor wording, or missing critical aspects of the input, as explained in 'textual_feedback'. Empty if none apply.

---
**Instructions:**
1. Evaluate the response based on:
    - **Accuracy**: Does it correctly address the input?
    - **Relevance**: Does it stay on topic?
    - **Clarity**: Is it easy to understand?
    - **Completeness**: Does it cover the input's core needs?
2. Select the most significant phrases for 'good_spans' and 'poor_spans', keeping them impactful, and essence-focused, with brief justifications. Include none if no phrases stand out.
3. Ensure 'good_spans' and 'poor_spans' are directly supported by the analysis in 'textual_feedback'.

---
**Output Format:**
{{
"textual_feedback": "Your critique here summarizing key strengths and weaknesses in one paragraph.",
"good_spans": ["phrase1", "phrase2",...],
"poor_spans": ["phrase1", "phrase2",...]
}}'''

        critique = item.get("critique", "")
        good_spans = item.get("good_spans", [])
        poor_spans = item.get("poor_spans", [])

        input_text = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"

        answer = json.dumps({
            "textual_feedback": critique,
            "good_spans": good_spans,
            "poor_spans": poor_spans
        }, ensure_ascii=False)

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


def main():
    """Main training function."""
    MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
    dataset_file = "./data/ultrafeedback/RM/train_processed_span_v3.json"
    valid_dataset_file = "./data/ultrafeedback/RM/test_processed_span_v3.json"
    BATCH_SIZE = 1
    EPOCHS = 3
    prompt_max_length = 950
    max_length = 1250
    exp = "ckpt/text2grad_ultrafeedback_RM"

    os.makedirs(exp, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"
    
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    model.pad_token_id = tokenizer.eos_token_id
    
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, lora_config)
    print("LoRA layers have been added.")

    ds_config = {
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
                "lr": 1e-5,  
                "betas": [0.9, 0.999],
                "eps": 1e-8,
                "weight_decay": 3e-7
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
        "train_batch_size": 1 * BATCH_SIZE * torch.cuda.device_count(),
        "gradient_accumulation_steps": 1,
        "train_micro_batch_size_per_gpu": BATCH_SIZE,
        "gradient_clipping": 1.0,
        "steps_per_print": 2000,
        "wall_clock_breakdown": False
    }

    trainable_parameters = [p for p in model.parameters() if p.requires_grad]
    print(f"Trainable parameters: {sum(p.numel() for p in trainable_parameters)}")
    
    model_engine, optimizer, _, _ = deepspeed.initialize(
        model=model,
        model_parameters=trainable_parameters,
        config=ds_config
    )

    train_dataset = QACDataset(dataset_file, tokenizer=tokenizer, prompt_max_length=prompt_max_length,
                               max_length=max_length)
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    valid_dataset = QACDataset(valid_dataset_file, tokenizer=tokenizer, prompt_max_length=prompt_max_length,
                               max_length=max_length)
    valid_dataloader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)

    project_name = "UltraFeedback-RM"
    wandb.init(project=project_name, name=project_name)
    
    for epoch in range(EPOCHS):
        model_engine.train()
        for step, batch in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
            if epoch == 0 and step == 0:
                sample_input_ids = batch["input_ids"][0].cpu().numpy()
                decoded_text = tokenizer.decode(sample_input_ids, skip_special_tokens=False)
                print("\n\n===== FIRST PROMPT SAMPLE =====")
                print(decoded_text)
                print("===== END OF PROMPT SAMPLE =====\n\n")

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
            })

            if step % 400 == 0 and step > 1000:
                get_accelerator().empty_cache()
                eval_losses = []

                eval_subset_size = 200
                total_eval_samples = len(valid_dataset)

                random_indices = torch.randperm(total_eval_samples)[:eval_subset_size]
                eval_subset = torch.utils.data.Subset(valid_dataset, random_indices)

                eval_subset_loader = DataLoader(
                    eval_subset,
                    batch_size=BATCH_SIZE,
                    shuffle=True
                )

                with torch.no_grad():
                    table = defaultdict(list)
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
                wandb.log({
                    "eval/loss": avg_eval_loss,
                    "eval/samples": eval_subset_size
                })
                print(f"Epoch {epoch}, Step {step}, Loss: {loss.item()}, "
                      f"Eval Loss (on {eval_subset_size} samples): {avg_eval_loss:.4f}")

                get_accelerator().empty_cache()

            if step % 400 == 0 and step > 3000:
                if model_engine.local_rank == 0:
                    checkpoint_dir = os.path.join(exp, f'{epoch}_{step}')
                    os.makedirs(checkpoint_dir, exist_ok=True)
                    model_engine.save_pretrained(checkpoint_dir)
                    get_accelerator().empty_cache()


if __name__ == "__main__":
    main()