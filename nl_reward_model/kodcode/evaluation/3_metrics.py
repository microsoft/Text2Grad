import json
import numpy as np
import argparse
from typing import List, Dict, Any, Tuple
from difflib import SequenceMatcher

def load_data(file_path: str) -> List[Dict[str, Any]]:
    """Load the evaluation data from a JSON file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def calculate_code_similarity(generated: List[str], reference: List[str]) -> Tuple[float, float, float]:
    """
    Calculate precision, recall, and F1 score for code segments using sequence matching.
    This approach considers the position and order of code lines.
    
    Args:
        generated: List of generated code snippets
        reference: List of reference code snippets
        
    Returns:
        Tuple of (precision, recall, F1 score)
    """
    if not reference and not generated:
        return 1.0, 1.0, 1.0
    if not reference:
        return 0.0, 1.0, 0.0
    if not generated:
        return 1.0, 0.0, 0.0

    gen_text = "\n".join(generated)
    ref_text = "\n".join(reference)

    matcher = SequenceMatcher(None, gen_text, ref_text)
    matching_blocks = matcher.get_matching_blocks()

    matched_chars = sum(block.size for block in matching_blocks if block.size > 0)

    precision = matched_chars / len(gen_text) if gen_text else 1.0
    recall = matched_chars / len(ref_text) if ref_text else 1.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return precision, recall, f1

def calculate_text_similarity(generated: str, reference: str) -> Tuple[float, float, float]:
    """
    Calculate precision, recall, and F1 score for textual feedback using sequence matching.
    
    Args:
        generated: Generated text feedback
        reference: Reference text feedback
        
    Returns:
        Tuple of (precision, recall, F1 score)
    """
    if not reference and not generated:
        return 1.0, 1.0, 1.0
    if not reference:
        return 0.0, 1.0, 0.0
    if not generated:
        return 1.0, 0.0, 0.0

    matcher = SequenceMatcher(None, generated, reference)
    matching_blocks = matcher.get_matching_blocks()

    matched_chars = sum(block.size for block in matching_blocks if block.size > 0)

    precision = matched_chars / len(generated) if generated else 1.0
    recall = matched_chars / len(reference) if reference else 1.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return precision, recall, f1

def evaluate_dataset(data: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    """
    Evaluate the entire dataset and calculate aggregate metrics.
    
    Args:
        data: List of evaluation data items
        
    Returns:
        Dictionary containing evaluation metrics
    """
    wrong_metrics = []
    improvement_metrics = []
    feedback_metrics = []

    for item in data:
        gen_wrong = item.get("generated_wrong_code", [])
        gen_improvement = item.get("generated_improvement_code", [])
        gen_feedback = item.get("generated_feedback", "")

        ref_wrong = item.get("wrong_code", [])
        ref_improvement = item.get("improvement_code", [])
        ref_feedback = item.get("feedback", "")

        wrong_precision, wrong_recall, wrong_f1 = calculate_code_similarity(gen_wrong, ref_wrong)
        wrong_metrics.append({
            "precision": wrong_precision,
            "recall": wrong_recall,
            "f1": wrong_f1
        })

        imp_precision, imp_recall, imp_f1 = calculate_code_similarity(gen_improvement, ref_improvement)
        improvement_metrics.append({
            "precision": imp_precision,
            "recall": imp_recall,
            "f1": imp_f1
        })

        feedback_precision, feedback_recall, feedback_f1 = calculate_text_similarity(gen_feedback, ref_feedback)
        feedback_metrics.append({
            "precision": feedback_precision,
            "recall": feedback_recall,
            "f1": feedback_f1
        })

    def average_metrics(metrics_list: List[Dict[str, float]]) -> Dict[str, float]:
        """Calculate average precision, recall, and F1 scores from a list of metrics."""
        avg = {
            "precision": np.mean([m["precision"] for m in metrics_list]),
            "recall": np.mean([m["recall"] for m in metrics_list]),
            "f1": np.mean([m["f1"] for m in metrics_list])
        }
        return avg

    sample_metrics = []
    for w, i, f in zip(wrong_metrics, improvement_metrics, feedback_metrics):
        sample_metrics.append({
            "precision": (w["precision"] + i["precision"] + f["precision"]) / 3,
            "recall": (w["recall"] + i["recall"] + f["recall"]) / 3,
            "f1": (w["f1"] + i["f1"] + f["f1"]) / 3 
        })

    return {
        "wrong_code": average_metrics(wrong_metrics),
        "improvement_code": average_metrics(improvement_metrics),
        "feedback": average_metrics(feedback_metrics),
        "overall": average_metrics(sample_metrics),
        "samples": len(data)
    }

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate model outputs against reference data")
    
    parser.add_argument("--input_file", type=str, 
                        default="../ckpt/text2grad_kodcode_RM/EVAL/inference_results_0_4400_nofeedback.json",
                        help="Path to the inference results file")
    
    parser.add_argument("--output_file", type=str, 
                        default="word_level_evaluation_results.json",
                        help="Path to save the evaluation results")
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    data = load_data(args.input_file)

    results = evaluate_dataset(data)

    print(f"Evaluated {results['samples']} samples")
    print("\nWord-Level Evaluation Results:")

    print("\nWrong Code Detection:")
    print(f"Precision: {results['wrong_code']['precision']:.4f}")
    print(f"Recall: {results['wrong_code']['recall']:.4f}")
    print(f"F1 Score: {results['wrong_code']['f1']:.4f}")

    print("\nImprovement Code Suggestions:")
    print(f"Precision: {results['improvement_code']['precision']:.4f}")
    print(f"Recall: {results['improvement_code']['recall']:.4f}")
    print(f"F1 Score: {results['improvement_code']['f1']:.4f}")

    print("\nTextual Feedback:")
    print(f"Precision: {results['feedback']['precision']:.4f}")
    print(f"Recall: {results['feedback']['recall']:.4f}")
    print(f"F1 Score: {results['feedback']['f1']:.4f}")

    print("\nOverall Performance:")
    print(f"Precision: {results['overall']['precision']:.4f}")
    print(f"Recall: {results['overall']['recall']:.4f}")
    print(f"F1 Score: {results['overall']['f1']:.4f}")

    with open(args.output_file, "w") as f:
        json.dump(results, f, indent=4)

    print(f"\nResults saved to '{args.output_file}'")

if __name__ == "__main__":
    main()