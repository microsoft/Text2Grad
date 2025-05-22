import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.utils.data import DataLoader, Dataset
import json
from tqdm import tqdm
import os
import argparse

# Add this to handle the LOCAL_RANK error
if "LOCAL_RANK" not in os.environ:
    os.environ["LOCAL_RANK"] = "0"

class QACDataset(Dataset):
    def __init__(self, json_path, tokenizer, prompt_max_length, max_length):
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

        # Extract only question and solution, omit answer
        question = item["question"]
        solution = item["solution"]

        # Format the template without exposing answer
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
        # Construct input format with chat template
        input_text = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"

        # Tokenize and process
        tokens = self.tokenizer.tokenize(input_text)
        if len(tokens) > self.max_length:
            tokens = tokens[:self.max_length]

        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)

        # Add padding
        padding_len = self.max_length - len(input_ids)
        input_ids = [self.pad_token_id] * padding_len + input_ids  # Left padding
        attention_mask = [0] * padding_len + [1] * len(tokens)  # Corresponding mask

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long)
        }


def parse_args():
    parser = argparse.ArgumentParser(description="Run inference with a language model on KodCode dataset")
    
    # Model and data parameters
    parser.add_argument("--model_path", type=str, default="../ckpt/llama31-8B-span2span-v2/0_4000_merge",
                        help="Path to the model checkpoint")
    parser.add_argument("--dataset_path", type=str, 
                        default="../data/KodCode/kodcode_test.json",
                        help="Path to the dataset file")
    parser.add_argument("--output_file", type=str, default="inference_results_0_4000.json",
                        help="Path to save the inference results")
    
    # Inference parameters
    parser.add_argument("--batch_size", type=int, default=40,
                        help="Batch size for inference")
    parser.add_argument("--prompt_max_length", type=int, default=1000,
                        help="Maximum length for the prompt")
    parser.add_argument("--max_length", type=int, default=1600,
                        help="Maximum total sequence length")
    parser.add_argument("--max_new_tokens", type=int, default=350,
                        help="Maximum number of new tokens to generate")
    
    # Hardware parameters
    parser.add_argument("--gpu_ids", type=str, default="0,1,2,3,4,5",
                        help="Comma-separated list of GPU IDs to use")
    parser.add_argument("--gpu_memory", type=str, default="30GiB",
                        help="Memory limit per GPU")
    
    return parser.parse_args()


def main():
    # Parse command-line arguments
    args = parse_args()
    
    # Set GPU devices
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
    
    # Set device - this is still useful for non-model operations
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    # Create a custom device map to explicitly distribute across all GPUs
    num_gpus = torch.cuda.device_count()
    print(f"Number of available GPUs: {num_gpus}")

    if num_gpus > 1:
        # Load model with explicit device mapping to distribute across all GPUs
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            device_map="balanced",  # Use "balanced" instead of "auto" for better distribution
            max_memory={i: args.gpu_memory for i in range(num_gpus)},  # Allocate memory per GPU
            torch_dtype=torch.float16
        )
    else:
        # Fallback to single GPU
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            device_map="auto",
            torch_dtype=torch.float16
        )

    print(f"Model device map: {model.hf_device_map}")  # Print the device map to verify distribution

    # Load dataset
    valid_dataset = QACDataset(
        args.dataset_path,
        tokenizer=tokenizer,
        prompt_max_length=args.prompt_max_length,
        max_length=args.max_length
    )
    valid_dataloader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False)

    # Store results
    results = []

    # Inference
    with torch.no_grad():
        for batch_idx, batch in tqdm(enumerate(valid_dataloader), desc="Inferencing"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            # Generate responses
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=args.max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

            # Decode outputs
            for i, output in enumerate(outputs):
                # Find input length
                input_length = input_ids[i].size(0)
                assistant_response = tokenizer.decode(output[input_length:], skip_special_tokens=True)

                try:
                    json_start = assistant_response.find('{')
                    json_end = assistant_response.rfind('}') + 1
                    if json_start != -1 and json_end != -1:
                        json_str = assistant_response[json_start:json_end]
                        response_dict = json.loads(json_str)

                        original_idx = i + args.batch_size * batch_idx
                        if original_idx < len(valid_dataset.json_data):
                            original_item = valid_dataset.json_data[original_idx]

                            results.append({
                                "generated_response": assistant_response,
                                "improvement_code": original_item.get("improvement_code", []),
                                "wrong_code": original_item.get("wrong_code", []),
                                "feedback": original_item.get("code_feedback", ""),
                                "generated_improvement_code": response_dict.get("improvement_code", []),
                                "generated_wrong_code": response_dict.get("wrong_code", []),
                                "generated_feedback": response_dict.get("code_feedback", "")
                            })
                        else:
                            results.append({
                                "generated_response": assistant_response,
                                "improvement_code": [],
                                "wrong_code": [],
                                "feedback": "",
                                "generated_improvement_code": [],
                                "generated_wrong_code": [],
                                "generated_feedback": ""
                            })
                    else:
                        results.append({
                            "generated_response": assistant_response,
                            "improvement_code": [],
                            "wrong_code": [],
                            "feedback": "",
                            "generated_improvement_code": [],
                            "generated_wrong_code": [],
                            "generated_feedback": ""
                        })
                except json.JSONDecodeError:
                    # If JSON parsing fails, save original response
                    results.append({
                        "generated_response": assistant_response,
                        "improvement_code": [],
                        "wrong_code": [],
                        "feedback": "",
                        "generated_improvement_code": [],
                        "generated_wrong_code": [],
                        "generated_feedback": ""
                    })

    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()