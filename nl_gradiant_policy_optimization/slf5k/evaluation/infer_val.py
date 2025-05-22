import json
import random
import os
import torch
import argparse
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from functools import partial

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

class QACDataset(Dataset):
    def __init__(self, data, tokenizer, prompt_max_length, max_length):
        if isinstance(data, dict):
            self.json_data = data
            self.keys = list(self.json_data.keys())
        elif isinstance(data, list):
            self.json_data = {str(i): item for i, item in enumerate(data) if 'post' in item and 'ideal_human_summary' in item}
            self.keys = list(self.json_data.keys())
        else:
            raise ValueError("Input data must be a dictionary or a list of dictionaries.")

        self.prompt_max_length = prompt_max_length
        self.max_length = max_length
        self.answer_max_length = max_length - prompt_max_length - 1
        self.pad_token_id = tokenizer.pad_token_id
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, ind):
        key = self.keys[ind]
        value = self.json_data[key]
        post = value["post"]
        ideal_human_summary = value["ideal_human_summary"]
        question = post
        answer = ideal_human_summary
        cur_example = self.build_dataset(question, answer)
        return cur_example

    def build_dataset(self, question, answer):
        def preprocess_function(question, answer):
            new_question = "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n" + \
                         "Please provide a comprehensive and concise summary of the following text. " + \
                         "Focus on the main ideas and key points while maintaining clarity and coherence.\n\n" + \
                         "Text: " + question + "\n<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
            query_num = len(answer.split(","))

            input_ids = torch.tensor(self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(new_question)),
                                     dtype=torch.long)
            attention_mask = torch.ones(input_ids.shape)
            new_examples = {
                "query": question,
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "query_nums": query_num,
                "answer": answer,
                "ideal_human_summary": answer,
            }
            return new_examples

        new_examples = preprocess_function(question, answer)
        return new_examples

# Define the collate function globally or before sample_and_infer
def collate_fn_with_tokenizer(batch, tokenizer): # Add tokenizer as an argument
    # Extract all elements
    queries = [item["query"] for item in batch]
    input_ids = [item["input_ids"] for item in batch]
    attention_masks = [item["attention_mask"] for item in batch]
    ideal_human_summaries = [item["ideal_human_summary"] for item in batch]

    # Find the maximum length in the batch
    max_length = max(len(ids) for ids in input_ids)

    # For left padding, we need to handle it manually
    padded_input_ids = []
    padded_attention_masks = []

    for ids, mask in zip(input_ids, attention_masks):
        padding_length = max_length - len(ids)
        if padding_length > 0:

            pad_ids = torch.full((padding_length,), tokenizer.pad_token_id, dtype=torch.long)
            pad_mask = torch.zeros(padding_length, dtype=torch.long)

            padded_ids = torch.cat([pad_ids, ids])
            padded_mask = torch.cat([pad_mask, mask])
        else:
            padded_ids = ids
            padded_mask = mask

        padded_input_ids.append(padded_ids)
        padded_attention_masks.append(padded_mask)

    # Stack tensors
    input_ids_tensor = torch.stack(padded_input_ids)
    attention_masks_tensor = torch.stack(padded_attention_masks)

    return {
        "query": queries,
        "input_ids": input_ids_tensor,
        "attention_mask": attention_masks_tensor,
        "ideal_human_summary": ideal_human_summaries
    }

def sample_and_infer(args):
    """Run inference on samples from the dataset using the specified model.
    
    Args:
        args: ArgumentParser namespace containing:
            - model_path: Path to the model
            - input_path: Path to input JSON
            - output_dir: Directory for output
            - sample_size: Number of samples (0 for all)
            - batch_size: Batch size for inference
    """
    # Paths from arguments
    input_json_path = args.input_path
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # Load model and tokenizer
    model_name = args.model_path
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # Set padding side before any tokenization happens
    tokenizer.padding_side = 'left'
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        if tokenizer.pad_token_id is None:
             tokenizer.pad_token_id = tokenizer.eos_token_id

    print(f"Tokenizer padding side: {tokenizer.padding_side}")

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="balanced"
    )

    with open(input_json_path, "r") as f:
        data = json.load(f)

    sample_size = args.sample_size
    keys_or_indices = list(range(len(data))) if isinstance(data, list) else list(data.keys())

    if sample_size > 0 and len(keys_or_indices) > sample_size:
        interval = len(keys_or_indices) // sample_size
        sampled_keys_or_indices = [keys_or_indices[i] for i in range(0, len(keys_or_indices), interval)][:sample_size]
        if isinstance(data, list):
            sampled_data = [data[i] for i in sampled_keys_or_indices]
        else: 
            sampled_data = {key: data[key] for key in sampled_keys_or_indices}
        print(f"Selecting {len(sampled_data)} examples at interval of {interval} from {len(keys_or_indices)} total examples")
    else:
        sampled_data = data # Use all data
        print(f"Using all available data ({len(keys_or_indices)} items).")

    prompt_max_length = 2048-1024
    max_length = 4096-1024
    dataset = QACDataset(sampled_data, tokenizer, prompt_max_length, max_length)

    collate_fn = partial(collate_fn_with_tokenizer, tokenizer=tokenizer)

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    GENERATION_CONFIG = {
        "temperature": 0.6,
        "do_sample": True,
        "top_p": 0.9,
        "max_new_tokens": 200,
        "min_new_tokens": 50
    }

    results = []
    for i, batch in enumerate(dataloader):
        input_ids = batch["input_ids"].to(model.device)
        attention_mask = batch["attention_mask"].to(model.device)
        query = batch["query"]  
        ideal_human_summary = batch["ideal_human_summary"]

        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                **GENERATION_CONFIG
            )

        for j in range(len(outputs)):
            generated_text = tokenizer.decode(outputs[j], skip_special_tokens=True)

            assistant_response = generated_text.split("<|end_header_id|>\n")[-1].strip()
            print(assistant_response)
            results.append({
                "query": query[j],
                "response": assistant_response.split("assistant\n\n")[-1].split("assistant\n")[-1],
                "ideal_human_summary": ideal_human_summary[j]
            })

        if (i + 1) % 5 == 0:
            print(f"Processed {(i + 1) * len(batch['query'])}/{len(dataset)} examples")

    # Save results
    model_name_short = os.path.basename(model_name.rstrip("/"))
    output_path = os.path.join(output_dir, f"{model_name_short}_samples.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"Results saved to {output_path}")

if __name__ == "__main__":    
    parser = argparse.ArgumentParser(
        description="Run inference with a specified model on a dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--model_path", type=str,
                        default="../ckpt/span_ppo_nofeedback/epoch_1_step_180-merge",
                        help="Path to the model or model name on HuggingFace")
    parser.add_argument("--input_path", type=str,
                        default="../slf5k_data/valid_score_all_processed.json",
                        help="Path to the input JSON file")
    parser.add_argument("--output_dir", type=str,
                        default="./result",
                        help="Directory to save the output JSON file")
    parser.add_argument("--sample_size", type=int, default=500,
                        help="Number of samples to select (0 for all)")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for inference")

    args = parser.parse_args()
    sample_and_infer(args)
