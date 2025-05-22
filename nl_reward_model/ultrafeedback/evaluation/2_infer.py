import torch
import os
import json
import re
import argparse
import pandas as pd
from tqdm import tqdm
from vllm import LLM, SamplingParams

def parse_args():
    parser = argparse.ArgumentParser(description="Run inference with VLLM")
    parser.add_argument("--model_path", type=str,
                        default="../ckpt/llama31-8B-span2span-v2-nofeedback/0_10000_merge")
    parser.add_argument("--valid_dataset_file", type=str,
                        default="../data/ultrafeedback/RM/test_annotated_v4_concurrent_2step.json")
    parser.add_argument("--output_file", type=str,
                        default="validation_results.json")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_new_tokens", type=int, default=300)
    parser.add_argument("--num_samples", type=int, default=1000, help="Number of samples to evaluate, -1 for all")
    parser.add_argument("--tensor_parallel_size", type=int, default=1, help="Number of GPUs for tensor parallelism")
    parser.add_argument("--gpu_ids", type=str, default="0,1,2,3,4,5,6,7", help="GPU IDs to use, comma separated")
    return parser.parse_args()

def format_prompt(user_prompt, assistant_response):
    """Format the prompt for the model"""
    prompt = f'''Please critique the following response to a user input and provide word-level list of good and poor spans:

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

---
**Output Format:**
{{
"good_spans": ["phrase1", "phrase2",...],
"poor_spans": ["phrase1", "phrase2",...]
}}'''

    # Construct input format
    return f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"

def parse_response(assistant_part):
    """Parse the model's response to extract good and poor spans"""
    try:
        # First try to find and parse JSON directly
        json_start = assistant_part.find("{")
        json_end = assistant_part.rfind("}") + 1

        if json_start >= 0 and json_end > json_start:
            json_str = assistant_part[json_start:json_end]
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                # If JSON parsing fails, use regex to extract spans
                return extract_spans_with_regex(assistant_part)
        else:
            # If no JSON-like structure found, try regex directly
            return extract_spans_with_regex(assistant_part)

    except Exception as e:
        return {"error": f"Failed to parse response: {str(e)}", "raw_response": assistant_part}

def extract_spans_with_regex(text):
    """Extract spans using regex when JSON parsing fails"""
    # Extract good spans using regex
    good_spans_pattern = r'"good_spans"\s*:\s*\[(.*?)\]'
    good_spans_match = re.search(good_spans_pattern, text, re.DOTALL)
    good_spans = []
    if good_spans_match:
        good_spans_text = good_spans_match.group(1)
        # Extract individual spans (handling both single and double quotes)
        span_pattern = r'(?:"([^"]*?)"|\'([^\']*?)\')'
        good_spans = [m[0] or m[1] for m in re.findall(span_pattern, good_spans_text)]

    # Extract poor spans using regex
    poor_spans_pattern = r'"poor_spans"\s*:\s*\[(.*?)\]'
    poor_spans_match = re.search(poor_spans_pattern, text, re.DOTALL)
    poor_spans = []
    if poor_spans_match:
        poor_spans_text = poor_spans_match.group(1)
        # Extract individual spans (handling both single and double quotes)
        span_pattern = r'(?:"([^"]*?)"|\'([^\']*?)\')'
        poor_spans = [m[0] or m[1] for m in re.findall(span_pattern, poor_spans_text)]

    return {
        "good_spans": good_spans,
        "poor_spans": poor_spans
    }

def main():
    args = parse_args()
    
    # Set GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids

    # Load model with VLLM
    print(f"Loading model from {args.model_path} with VLLM")
    model = LLM(
        model=args.model_path,
        tensor_parallel_size=args.tensor_parallel_size,
        dtype="float16",
        trust_remote_code=True,
        max_model_len=8192,    
    )

    # Define sampling parameters
    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.9,
        max_tokens=args.max_new_tokens,
    )

    # Load validation data
    print(f"Loading validation data from {args.valid_dataset_file}")
    with open(args.valid_dataset_file, "r") as f:
        valid_data = json.load(f)

    if args.num_samples > 0:
        valid_data = valid_data[:args.num_samples]

    results = []

    # Process validation data in batches
    for i in tqdm(range(0, len(valid_data), args.batch_size), desc="Processing batches"):
        batch_items = valid_data[i:i+args.batch_size]
        batch_prompts = []
        prompt_ids = []

        # Format prompts for the batch
        for idx, item in enumerate(batch_items):
            user_prompt = item["prompt"]
            assistant_response = item["response"]
            
            # Format the prompt
            input_text = format_prompt(user_prompt, assistant_response)
            batch_prompts.append(input_text)
            prompt_ids.append(i + idx)

        # Generate responses for the batch using VLLM
        outputs = model.generate(batch_prompts, sampling_params)

        # Process each response in the batch
        for output, prompt_id in zip(outputs, prompt_ids):
            item = valid_data[prompt_id]
            generated_text = output.outputs[0].text

            # Extract the model's response
            assistant_part = generated_text.strip()

            # Parse the response
            parsed_response = parse_response(assistant_part)

            # Store results
            result = {
                "prompt": item["prompt"],
                "response": item["response"],
                "ground_truth": {
                    "good_spans": item.get("good_spans", []),
                    "poor_spans": item.get("poor_spans", [])
                },
                "model_output": parsed_response,
                "raw_model_output": assistant_part
            }

            results.append(result)

            # Print only the first result from each batch to avoid excessive output
            if prompt_id == i:
                print(f"Sample output from batch {i//args.batch_size + 1}:")
                print(generated_text[:200] + "..." if len(generated_text) > 200 else generated_text)
                print("-" * 50)

    # Save results
    print(f"Saving results to {args.output_file}")
    with open(args.output_file, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    main()