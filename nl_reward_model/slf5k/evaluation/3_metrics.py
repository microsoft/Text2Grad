import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.utils.data import DataLoader, Dataset
import json
import random
from tqdm import tqdm
from peft import PeftModel
import rouge_score
import re
from rouge_score import rouge_scorer
import os
import argparse

SYSTEM_PROMPT = '''You are a human annotator specializing in linguistics. Evaluate the generated summary against the original post using word-level scoring based on span-based quality assessment.

# Objective
- Critique the summary first, then assign scores to words based on identified spans.
- Focus on quality over quantity: identify concise, meaningful phrases (spans) rather than excessive breakdowns.

# Scoring Rules
- Score 1: Words in "good spans" (accurate, helpful phrases that capture key details from the original post).
- Score -1: Words in "poor spans" (inaccurate, redundant, or misleading phrases that detract from quality).
- Score 0: Neutral words (not part of good or poor spans).

# Evaluation Steps
1. Critique the summary:
   - Identify "good spans": concise phrases that accurately and helpfully reflect the original post's key points.
   - Identify "poor spans": concise phrases that are inaccurate, redundant, or misleading.
   - Keep spans meaningful and minimal; avoid over-segmentation.
2. Assign scores:
   - 1 for each word in good spans.
   - -1 for each word in poor spans.
   - 0 for all other words.

# Input Format
{
  "original_post": "Text of the original Reddit post.",
  "generated_summary": "Text of the model-generated summary."
}

# Output Format in json
{
  "word_score_list": [
    ("word1", "Score (-1, 0, or 1)"),
    ("word2", "Score (-1, 0, or 1)"),
    ...
  ]
}

# Note
- Scores apply to words only, not punctuation.
- Directly output json and no other words
'''

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"


def load_and_prepare_model():
    model_path = "../ckpt/text2grad_slf5k_RM/0_4000_merge"
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, padding_side="right")
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="balanced",
        trust_remote_code=True
    )
    model.eval()

    return model, tokenizer


def prepare_input(post, summary):
    question = f"""# User Input
{{
  "original_post": "{post}",
  "generated_summary": "{summary}"
}}
"""
    question = f"""# Input
{{
  "original_post": "{post}",
  "generated_summary": "{summary}"
}}
"""

    prompt = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n{SYSTEM_PROMPT}\n{question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"

    return prompt


def process_batch(model, tokenizer, batch_data):
    prompts = [prepare_input(data["post"], data["generated_summary"]) for data in batch_data]
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.1,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id
        )

    responses = []
    for i, output in enumerate(outputs):
        input_length = inputs['input_ids'].shape[1]

        response_tokens = output[input_length:]
        response = tokenizer.decode(response_tokens, skip_special_tokens=True).strip()

        print(response)

        responses.append(response)

    return responses


def extract_word_scores(response):
    try:
        response_dict = json.loads(response)
        word_score_list = response_dict.get("word_score_list", [])

        # Handle different formats
        if isinstance(word_score_list, str):
            # Parse string representation of tuples
            tuples = re.findall(r'\([\"\']?([^\"\',]+)[\"\']?,\s*(-?\d+)\)', word_score_list)
            return [(word, int(score)) for word, score in tuples]
        elif isinstance(word_score_list, list):
            # Handle list of dictionaries
            if word_score_list and isinstance(word_score_list[0], dict):
                # Try different key combinations
                if "word" in word_score_list[0] and "score" in word_score_list[0]:
                    return [(item["word"], int(item["score"])) for item in word_score_list]
                elif "word" in word_score_list[0] and "Score" in word_score_list[0]:
                    return [(item["word"], int(item["Score"])) for item in word_score_list]
            # Handle list of lists/tuples
            elif word_score_list and isinstance(word_score_list[0], (list, tuple)):
                return [(str(item[0]), int(item[1])) for item in word_score_list]

        # If we get here, try a more general approach
        print(f"Warning: Using fallback extraction for format: {type(word_score_list)}")
        if isinstance(word_score_list, list):
            result = []
            for item in word_score_list:
                if isinstance(item, (list, tuple)) and len(item) >= 2:
                    result.append((str(item[0]), int(item[1])))
                elif isinstance(item, dict) and len(item) >= 2:
                    # Try to find word and score keys
                    word_key = next((k for k in item.keys() if k.lower() in ['word', 'token']), None)
                    score_key = next((k for k in item.keys() if k.lower() in ['score', 'value']), None)
                    if word_key and score_key:
                        result.append((str(item[word_key]), int(item[score_key])))
            return result

        return []
    except json.JSONDecodeError:
        # If JSON parsing fails, try using regex to extract word scores
        patterns = [
            r'\([\"\']?([^\"\',]+)[\"\']?,\s*(-?\d+)\)',  # ("word", 1) or ('word', 1)
            r'\[[\"\']?([^\"\',]+)[\"\']?,\s*(-?\d+)\]',  # ["word", 1] or ['word', 1]
            r'{\s*[\"\']?word[\"\']?\s*:\s*[\"\']?([^\"\',]+)[\"\']?\s*,\s*[\"\']?score[\"\']?\s*:\s*(-?\d+)\s*}'  # {"word": "word", "score": 1}
        ]

        for pattern in patterns:
            tuples = re.findall(pattern, response)
            if tuples:
                return [(word, int(score)) for word, score in tuples]

        return []


def extract_standard_word_scores(data):
    """Extract word scores from validation data standard format"""
    try:
        # Check for different possible keys
        for key in ["word_score_list", "word_scores", "scores"]:
            if key in data:
                word_score_list = data[key]
                break
        else:
            # If no known keys found
            word_score_list = []

            # Try to find a key that might contain the scores
            for key, value in data.items():
                if isinstance(value, list) and value and (
                    isinstance(value[0], (list, tuple, dict)) or
                    key.lower() in ["word_score_list", "word_scores", "scores"]
                ):
                    word_score_list = value
                    break

        # Process based on format
        if isinstance(word_score_list, list):
            # If list is empty, return empty list
            if not word_score_list:
                return []

            # Dictionary format
            if isinstance(word_score_list[0], dict):
                # Try different key combinations
                if "word" in word_score_list[0] and "score" in word_score_list[0]:
                    return [(item["word"], item["score"]) for item in word_score_list]
                elif "word" in word_score_list[0] and "Score" in word_score_list[0]:
                    return [(item["word"], item["Score"]) for item in word_score_list]
                else:
                    # Try to find word and score keys
                    word_key = next((k for k in word_score_list[0].keys() if k.lower() in ['word', 'token']), None)
                    score_key = next((k for k in word_score_list[0].keys() if k.lower() in ['score', 'value']), None)
                    if word_key and score_key:
                        return [(item[word_key], item[score_key]) for item in word_score_list]

            # List/tuple format
            elif isinstance(word_score_list[0], (list, tuple)):
                if len(word_score_list[0]) >= 2:
                    return [(item[0], item[1]) for item in word_score_list]

        print(f"Warning: Unexpected word_score_list format: {type(word_score_list)}")
        if word_score_list:
            print(f"First item type: {type(word_score_list[0])}")
            print(f"Sample content: {word_score_list[:2]}")

    except (KeyError, TypeError, IndexError) as e:
        print(f"Error extracting word scores: {str(e)}")
        if data:
            print(f"Data keys available: {list(data.keys())}")

    return []


def calculate_rouge_scores(predictions, references):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = {
        'rouge1': {'precision': 0, 'recall': 0, 'fmeasure': 0},
        'rouge2': {'precision': 0, 'recall': 0, 'fmeasure': 0},
        'rougeL': {'precision': 0, 'recall': 0, 'fmeasure': 0}
    }

    valid_pairs = 0
    for pred, ref in zip(predictions, references):
        if pred and ref:  # Check if both are non-empty
            score = scorer.score(ref, pred)
            for metric in scores.keys():
                scores[metric]['precision'] += score[metric].precision
                scores[metric]['recall'] += score[metric].recall
                scores[metric]['fmeasure'] += score[metric].fmeasure
            valid_pairs += 1

    # Calculate averages
    if valid_pairs > 0:
        for metric in scores.keys():
            for key in scores[metric].keys():
                scores[metric][key] /= valid_pairs

    return scores


def calculate_score_metrics(predicted_scores, standard_scores, target_score):
    """Calculate precision, recall, and F1 for a specific score"""
    # Convert lists to word-score dictionaries for easier comparison
    pred_dict = {word: score for word, score in predicted_scores}
    std_dict = {word: score for word, score in standard_scores}

    # Find common words
    common_words = set(pred_dict.keys()) & set(std_dict.keys())

    # Calculate metrics
    true_positives = sum(1 for word in common_words
                        if pred_dict[word] == target_score and std_dict[word] == target_score)
    predicted_positives = sum(1 for score in pred_dict.values() if score == target_score)
    actual_positives = sum(1 for score in std_dict.values() if score == target_score)

    return {
        'true_positives': true_positives,
        'predicted_positives': predicted_positives,
        'actual_positives': actual_positives
    }


def calculate_overall_metrics(results):
    """Calculate overall metrics for all samples"""
    total_metrics = {
        'score_specific': {score: {
            'true_positives': 0,
            'predicted_positives': 0,
            'actual_positives': 0
        } for score in [-1, 0, 1]},
        'word_matching': {
            'total_std_words': 0,
            'total_pred_words': 0,
            'missing_words': 0,
            'matching_words': 0
        }
    }

    valid_samples = 0
    for sample in results['samples']:
        pred_scores = sample['predicted_word_scores']
        std_scores = sample['standard_word_scores']

        if not pred_scores or not std_scores:
            continue

        valid_samples += 1

        # Update word matching metrics
        pred_words = set(word for word, _ in pred_scores)
        std_words = set(word for word, _ in std_scores)

        total_metrics['word_matching']['total_std_words'] += len(std_words)
        total_metrics['word_matching']['total_pred_words'] += len(pred_words)
        total_metrics['word_matching']['missing_words'] += len(std_words - pred_words)
        total_metrics['word_matching']['matching_words'] += len(std_words & pred_words)

        # Calculate score-specific metrics
        for score in [-1, 0, 1]:
            metrics = calculate_score_metrics(pred_scores, std_scores, score)
            for key, value in metrics.items():
                total_metrics['score_specific'][score][key] += value

    # Calculate precision, recall, and F1 for each score
    for score in [-1, 0, 1]:
        tp = total_metrics['score_specific'][score]['true_positives']
        pp = total_metrics['score_specific'][score]['predicted_positives']
        ap = total_metrics['score_specific'][score]['actual_positives']

        precision = tp / pp if pp > 0 else 0
        recall = tp / ap if ap > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        total_metrics['score_specific'][score]['precision'] = precision
        total_metrics['score_specific'][score]['recall'] = recall
        total_metrics['score_specific'][score]['f1'] = f1

    return total_metrics


def parse_args():
    parser = argparse.ArgumentParser(description="Calculate metrics for reward model evaluation")
    parser.add_argument("--results_path", type=str, 
                        default="../Evaluation/0_4800_res_span",
                        help="Path to the inference results from 2_infer.py")
    parser.add_argument("--output_path", type=str,
                        default="../Evaluation/0_4800_metrics_span",
                        help="Path to save the metrics results")
    parser.add_argument("--data_path", type=str,
                        default="../SLF5K_label/validation_critique_processed.json",
                        help="Path to the validation data with ground truth")
    return parser.parse_args()

def extract_textual_feedback(response):
    try:
        # Try to parse JSON response
        response_dict = json.loads(response)
        return response_dict.get("textual_feedback", "")
    except json.JSONDecodeError:
        # If JSON parsing fails, try using regex to extract
        match = re.search(r'"textual_feedback"\s*:\s*"([^"]*)"', response)
        if match:
            return match.group(1)
        return ""

def main():
    # Parse command line arguments
    args = parse_args()
    
    print("\nLoading inference results...")
    with open(args.results_path, "r") as f:
        inference_results = json.load(f)
    print(f"Loaded {len(inference_results)} inference results")
    
    # Load validation data for ground truth
    print("\nLoading validation data...")
    with open(args.data_path, "r") as f:
        val_data = json.load(f)
    print(f"Loaded validation data with {len(val_data)} samples")
    
    # Process results and calculate metrics
    print("\nProcessing results and calculating metrics...")
    
    # Extract predictions and references
    predictions = []
    references = []
    
    final_results = {
        "samples": []
    }
    
    for result in inference_results:
        try:
            # Get original data index
            data_index = result["index"]
            
            # Extract model response
            model_response = result["model_response"]
            
            # Extract predicted word scores
            predicted_word_scores = extract_word_scores(model_response)
            
            # Extract textual feedback
            predicted_feedback = extract_textual_feedback(model_response)
            
            # Find corresponding ground truth data
            # This depends on how your data is structured
            original_data = None
            if isinstance(val_data, dict):
                # Try to find by matching post and summary
                for key, item in val_data.items():
                    if (item.get("post") == result["original_post"] and 
                        item.get("generated_summary") == result["generated_summary"]):
                        original_data = item
                        break
            else:
                # Try to find in list by index
                for item in val_data:
                    if (item.get("post") == result["original_post"] and 
                        item.get("generated_summary") == result["generated_summary"]):
                        original_data = item
                        break
            
            if not original_data:
                print(f"Warning: Could not find ground truth for result at index {data_index}")
                continue
                
            # Extract ground truth feedback and word scores
            standard_feedback = original_data.get("textual_feedback", "")
            standard_word_scores = extract_standard_word_scores(original_data)
            
            # Add to lists for ROUGE calculation
            if predicted_feedback and standard_feedback:
                predictions.append(predicted_feedback)
                references.append(standard_feedback)
            
            # Add to final results
            final_results["samples"].append({
                "index": data_index,
                "predicted_word_scores": predicted_word_scores,
                "standard_word_scores": standard_word_scores,
                "predicted_feedback": predicted_feedback,
                "standard_feedback": standard_feedback
            })
            
        except Exception as e:
            print(f"Error processing result: {str(e)}")
            continue
    
    print(f"\nProcessed {len(final_results['samples'])} samples with valid data")
    
    # Calculate ROUGE scores for textual feedback
    print("\nCalculating ROUGE scores for textual feedback...")
    rouge_scores = calculate_rouge_scores(predictions, references)
    
    # Calculate word score metrics
    print("\nCalculating word score metrics...")
    metrics = calculate_overall_metrics(final_results)
    
    # Combine all metrics
    final_results['rouge_scores'] = rouge_scores
    final_results['word_score_metrics'] = metrics
    
    # Save results
    with open(args.output_path, "w") as f:
        json.dump(final_results, f, indent=2)
    
    # Print simplified metrics summary
    print("\n" + "=" * 50)
    print("📊 EVALUATION METRICS SUMMARY 📊".center(50))
    print("=" * 50)

    # Print ROUGE scores for textual feedback
    print("\n🔍 TEXTUAL FEEDBACK - ROUGE SCORES:")
    print("-" * 50)
    for metric, values in rouge_scores.items():
        print(f"  {metric.upper()}:")
        print(f"    • F1: {values['fmeasure']:.4f}")

    # Print simplified score-specific metrics for word scores
    print("\n🔍 WORD SCORE LIST - METRICS:")
    print("-" * 50)

    for score in [-1, 0, 1]:
        score_label = "Negative (-1)" if score == -1 else "Neutral (0)" if score == 0 else "Positive (1)"
        print(f"\n  {score_label}:")
        print(f"    • Precision: {metrics['score_specific'][score]['precision']:.4f}")
        print(f"    • Recall:    {metrics['score_specific'][score]['recall']:.4f}")
        print(f"    • F1 Score:  {metrics['score_specific'][score]['f1']:.4f}")

    print("\n" + "=" * 50)
    print(f"✅ Results saved to: {args.output_path}")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    main()