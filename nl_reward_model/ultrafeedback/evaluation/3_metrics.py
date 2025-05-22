#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Calculate precision, recall, and F1 scores for word-level scoring
Consider word positions and calculate metrics separately for each score value (1, -1, 0)
Also calculate ROUGE metrics for textual_feedback
"""

import json
import argparse
import numpy as np
from tqdm import tqdm
import logging
import os
from rouge_score import rouge_scorer

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("word_metrics.log")
    ]
)
logger = logging.getLogger(__name__)

def calculate_metrics(ground_truth_list, extracted_list):
    """
    Calculate precision, recall, and F1 scores for word-level scoring, separately for each score value

    Args:
        ground_truth_list: word_score_list from ground truth
        extracted_list: word_score_list from extraction

    Returns:
        dict: Dictionary containing precision, recall, and F1 scores for each score value
    """
    # Initialize counters for each score value (1, -1, 0)
    metrics_by_score = {
        1: {"tp": 0, "fp": 0, "fn": 0},  # Positive score
        -1: {"tp": 0, "fp": 0, "fn": 0},  # Negative score
        0: {"tp": 0, "fp": 0, "fn": 0}   # Neutral score
    }

    # Ensure both lists have the same length
    min_len = min(len(ground_truth_list), len(extracted_list))

    # Iterate through each position
    for i in range(min_len):
        gt_word, gt_score = ground_truth_list[i]
        ex_word, ex_score = extracted_list[i]

        # Check if words match
        if gt_word != ex_word:
            logger.warning(f"Words at position {i} don't match: ground truth '{gt_word}' vs extracted '{ex_word}'")
            continue

        # Update metrics for each score value
        for score in [1, -1, 0]:
            # True Positive: both predicted and actual values are the current score
            if ex_score == score and gt_score == score:
                metrics_by_score[score]["tp"] += 1
            # False Positive: predicted value is the current score but actual value is not
            elif ex_score == score and gt_score != score:
                metrics_by_score[score]["fp"] += 1
            # False Negative: actual value is the current score but predicted value is not
            elif gt_score == score and ex_score != score:
                metrics_by_score[score]["fn"] += 1

    # Calculate precision, recall, and F1 for each score value
    result = {}
    total_tp = 0
    total_fp = 0
    total_fn = 0

    for score in [1, -1, 0]:
        tp = metrics_by_score[score]["tp"]
        fp = metrics_by_score[score]["fp"]
        fn = metrics_by_score[score]["fn"]

        # Accumulate overall metrics
        total_tp += tp
        total_fp += fp
        total_fn += fn

        # Calculate precision and recall for current score value
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0

        # Calculate F1 score
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        # Save results
        score_label = "positive" if score == 1 else "negative" if score == -1 else "neutral"
        result[score_label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "tp": tp,
            "fp": fp,
            "fn": fn
        }

    # Calculate overall metrics (unweighted average)
    overall_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    overall_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    overall_f1 = 2 * (overall_precision * overall_recall) / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0

    result["overall"] = {
        "precision": overall_precision,
        "recall": overall_recall,
        "f1": overall_f1,
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn
    }

    return result

def calculate_rouge_scores(reference_text, candidate_text):
    """
    Calculate ROUGE scores

    Args:
        reference_text: Reference text
        candidate_text: Candidate text

    Returns:
        dict: Contains F1, precision, and recall for ROUGE-1, ROUGE-2, and ROUGE-L
    """
    if not reference_text or not candidate_text:
        return {
            "rouge1": {"precision": 0, "recall": 0, "fmeasure": 0},
            "rouge2": {"precision": 0, "recall": 0, "fmeasure": 0},
            "rougeL": {"precision": 0, "recall": 0, "fmeasure": 0}
        }

    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference_text, candidate_text)

    result = {}
    for metric, score in scores.items():
        result[metric] = {
            "precision": score.precision,
            "recall": score.recall,
            "fmeasure": score.fmeasure
        }

    return result

def calculate_span_overlap_metrics(ground_truth_list, extracted_list):
    """
    Calculate overlap metrics for good and poor spans, including IOU (Intersection over Union)

    Args:
        ground_truth_list: word_score_list from ground truth
        extracted_list: word_score_list from extraction

    Returns:
        dict: Contains overlap metrics for good and poor spans
    """
    # Helper function to extract spans, avoiding code duplication
    def extract_spans(word_score_list):
        good_spans = []
        poor_spans = []
        current_good_span = []
        current_poor_span = []

        for i, (word, score) in enumerate(word_score_list):
            if score == 1:  # good span
                current_good_span.append((i, word))
                if current_poor_span:  # end current poor span
                    if len(current_poor_span) > 0:
                        poor_spans.append(current_poor_span)
                    current_poor_span = []
            elif score == -1:  # poor span
                current_poor_span.append((i, word))
                if current_good_span:  # end current good span
                    if len(current_good_span) > 0:
                        good_spans.append(current_good_span)
                    current_good_span = []
            else:  # neutral span
                if current_good_span:  # end current good span
                    if len(current_good_span) > 0:
                        good_spans.append(current_good_span)
                    current_good_span = []
                if current_poor_span:  # end current poor span
                    if len(current_poor_span) > 0:
                        poor_spans.append(current_poor_span)
                    current_poor_span = []

        # Handle potentially remaining spans
        if current_good_span:
            good_spans.append(current_good_span)
        if current_poor_span:
            poor_spans.append(current_poor_span)
            
        return good_spans, poor_spans
    
    # Extract spans from ground truth and extracted lists
    gt_good_spans, gt_poor_spans = extract_spans(ground_truth_list)
    ex_good_spans, ex_poor_spans = extract_spans(extracted_list)
    
    # Calculate span-level metrics
    results = {
        "good_span": {
            "gt_count": len(gt_good_spans),
            "ex_count": len(ex_good_spans),
            "exact_match": 0,
            "partial_match": 0,
            "avg_iou": 0.0,
        },
        "poor_span": {
            "gt_count": len(gt_poor_spans),
            "ex_count": len(ex_poor_spans),
            "exact_match": 0,
            "partial_match": 0,
            "avg_iou": 0.0,
        }
    }
    
    # Helper function to calculate span metrics
    def calculate_span_metrics(gt_spans, ex_spans, span_type):
        ious = []
        exact_match = 0
        partial_match = 0
        
        for gt_span in gt_spans:
            gt_indices = set([idx for idx, _ in gt_span])
            best_iou = 0.0
            has_match = False
            
            for ex_span in ex_spans:
                ex_indices = set([idx for idx, _ in ex_span])
                intersection = len(gt_indices.intersection(ex_indices))
                union = len(gt_indices.union(ex_indices))
                iou = intersection / union if union > 0 else 0

                if iou > best_iou:
                    best_iou = iou

                if iou == 1.0:  # exact match
                    exact_match += 1
                    has_match = True
                    break
                elif iou > 0 and not has_match:  # partial match, count only once
                    partial_match += 1
                    has_match = True
            
            if best_iou > 0:
                ious.append(best_iou)
        
        return {
            "exact_match": exact_match,
            "partial_match": partial_match,
            "avg_iou": np.mean(ious) if ious else 0.0
        }
    
    # Calculate metrics for good spans and poor spans
    good_metrics = calculate_span_metrics(gt_good_spans, ex_good_spans, "good_span")
    poor_metrics = calculate_span_metrics(gt_poor_spans, ex_poor_spans, "poor_span")
    
    # Update results
    results["good_span"].update(good_metrics)
    results["poor_span"].update(poor_metrics)

    # Calculate recall and precision
    for span_type in ["good_span", "poor_span"]:
        gt_count = results[span_type]["gt_count"]
        ex_count = results[span_type]["ex_count"]
        partial_match = results[span_type]["partial_match"]
        
        results[span_type]["recall"] = partial_match / gt_count if gt_count > 0 else 0.0
        results[span_type]["precision"] = partial_match / ex_count if ex_count > 0 else 0.0
        results[span_type]["f1"] = 2 * (results[span_type]["precision"] * results[span_type]["recall"]) / (results[span_type]["precision"] + results[span_type]["recall"]) if (results[span_type]["precision"] + results[span_type]["recall"]) > 0 else 0.0
        
        # Add stricter metrics - using exact_match instead of partial_match
        results[span_type]["exact_recall"] = results[span_type]["exact_match"] / gt_count if gt_count > 0 else 0.0
        results[span_type]["exact_precision"] = results[span_type]["exact_match"] / ex_count if ex_count > 0 else 0.0
        results[span_type]["exact_f1"] = 2 * (results[span_type]["exact_precision"] * results[span_type]["exact_recall"]) / (results[span_type]["exact_precision"] + results[span_type]["exact_recall"]) if (results[span_type]["exact_precision"] + results[span_type]["exact_recall"]) > 0 else 0.0

    return results

def evaluate_file(input_file, output_file):
    """Analyze word-level score metrics in the file"""
    logger.info(f"Starting analysis of file: {input_file}")

    try:
        # Read input file
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        logger.info(f"Successfully loaded data with {len(data)} records")

        # Store metrics for each record
        all_metrics = []
        all_rouge_scores = []
        all_span_metrics = []  # New: store span metrics

        # Track overall metrics
        total_counts = {
            "positive": {"tp": 0, "fp": 0, "fn": 0},
            "negative": {"tp": 0, "fp": 0, "fn": 0},
            "neutral": {"tp": 0, "fp": 0, "fn": 0},
            "overall": {"tp": 0, "fp": 0, "fn": 0}
        }

        # Track ROUGE metrics
        total_rouge = {
            "rouge1": {"precision": 0, "recall": 0, "fmeasure": 0},
            "rouge2": {"precision": 0, "recall": 0, "fmeasure": 0},
            "rougeL": {"precision": 0, "recall": 0, "fmeasure": 0}
        }

        # Track span metrics
        total_span_metrics = {
            "good_span": {
                "gt_count": 0, "ex_count": 0,
                "exact_match": 0, "partial_match": 0,
                "recall": 0, "precision": 0, "f1": 0
            },
            "poor_span": {
                "gt_count": 0, "ex_count": 0,
                "exact_match": 0, "partial_match": 0,
                "recall": 0, "precision": 0, "f1": 0
            }
        }

        # Process each record
        for i, result in enumerate(tqdm(data, desc="Calculating metrics")):
            # Ensure record contains necessary fields
            if "ground_truth" not in result or "model_output" not in result:
                logger.warning(f"Record {i} missing required fields, skipping")
                continue

            ground_truth = result["ground_truth"]
            extracted = result["model_output"]  # Using model_output instead of extracted

            # Ensure word_score_list exists
            if "word_score_list" not in ground_truth or "word_score_list" not in extracted:
                logger.warning(f"Record {i} missing word_score_list, skipping")
                continue

            ground_truth_list = ground_truth["word_score_list"]
            extracted_list = extracted["word_score_list"]

            metrics = calculate_metrics(ground_truth_list, extracted_list)

            span_metrics = calculate_span_overlap_metrics(ground_truth_list, extracted_list)

            for key in total_counts:
                for metric in ["tp", "fp", "fn"]:
                    total_counts[key][metric] += metrics[key][metric]

            for span_type in ["good_span", "poor_span"]:
                total_span_metrics[span_type]["gt_count"] += span_metrics[span_type]["gt_count"]
                total_span_metrics[span_type]["ex_count"] += span_metrics[span_type]["ex_count"]
                total_span_metrics[span_type]["exact_match"] += span_metrics[span_type]["exact_match"]
                total_span_metrics[span_type]["partial_match"] += span_metrics[span_type]["partial_match"]

            rouge_scores = {}
            if "textual_feedback" in ground_truth and "textual_feedback" in extracted:
                gt_feedback = ground_truth["textual_feedback"]
                ex_feedback = extracted["textual_feedback"]

                rouge_scores = calculate_rouge_scores(gt_feedback, ex_feedback)

                for metric in total_rouge:
                    for score_type in ["precision", "recall", "fmeasure"]:
                        total_rouge[metric][score_type] += rouge_scores[metric][score_type]

                all_rouge_scores.append({
                    "index": i,
                    "rouge_scores": rouge_scores
                })

            result["detailed_metrics"] = metrics
            result["rouge_scores"] = rouge_scores
            result["span_metrics"] = span_metrics  # New: add span metrics

            # Add to all metrics list
            all_metrics.append({
                "index": i,
                "metrics": metrics,
                "rouge_scores": rouge_scores,
                "span_metrics": span_metrics,  # New: add span metrics
                "post_length": len(result.get("prompt", "")),  # Using prompt instead of post
                "response_length": len(result.get("response", ""))  # Using response instead of generated_summary
            })

            # Add to span metrics list
            all_span_metrics.append({
                "index": i,
                "span_metrics": span_metrics
            })

        # Calculate overall metrics
        overall_metrics = {}
        for key in total_counts:
            tp = total_counts[key]["tp"]
            fp = total_counts[key]["fp"]
            fn = total_counts[key]["fn"]

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

            overall_metrics[key] = {
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "tp": tp,
                "fp": fp,
                "fn": fn
            }

        # Calculate overall span metrics
        overall_span_metrics = {}
        for span_type in ["good_span", "poor_span"]:
            gt_count = total_span_metrics[span_type]["gt_count"]
            ex_count = total_span_metrics[span_type]["ex_count"]
            partial_match = total_span_metrics[span_type]["partial_match"]

            precision = partial_match / ex_count if ex_count > 0 else 0
            recall = partial_match / gt_count if gt_count > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

            overall_span_metrics[span_type] = {
                "gt_count": gt_count,
                "ex_count": ex_count,
                "exact_match": total_span_metrics[span_type]["exact_match"],
                "partial_match": partial_match,
                "precision": precision,
                "recall": recall,
                "f1": f1
            }

        # Calculate average ROUGE scores
        avg_rouge = {
            "rouge1": {"precision": 0, "recall": 0, "fmeasure": 0},
            "rouge2": {"precision": 0, "recall": 0, "fmeasure": 0},
            "rougeL": {"precision": 0, "recall": 0, "fmeasure": 0}
        }

        if all_rouge_scores:
            for metric in avg_rouge:
                for score_type in ["precision", "recall", "fmeasure"]:
                    avg_rouge[metric][score_type] = total_rouge[metric][score_type] / len(all_rouge_scores)

        # Calculate average metrics
        average_metrics = {
            "positive": {"precision": [], "recall": [], "f1": []},
            "negative": {"precision": [], "recall": [], "f1": []},
            "neutral": {"precision": [], "recall": [], "f1": []},
            "overall": {"precision": [], "recall": [], "f1": []}
        }

        average_rouge = {
            "rouge1": {"precision": [], "recall": [], "fmeasure": []},
            "rouge2": {"precision": [], "recall": [], "fmeasure": []},
            "rougeL": {"precision": [], "recall": [], "fmeasure": []}
        }

        # Calculate average span metrics
        average_span_metrics = {
            "good_span": {"precision": [], "recall": [], "f1": [], "avg_iou": []},
            "poor_span": {"precision": [], "recall": [], "f1": [], "avg_iou": []}
        }

        for m in all_metrics:
            for key in average_metrics:
                for metric in ["precision", "recall", "f1"]:
                    average_metrics[key][metric].append(m["metrics"][key][metric])

            if m["rouge_scores"]:
                for metric in average_rouge:
                    for score_type in ["precision", "recall", "fmeasure"]:
                        average_rouge[metric][score_type].append(m["rouge_scores"][metric][score_type])

            # Add span metrics
            for span_type in ["good_span", "poor_span"]:
                for metric in ["precision", "recall", "f1", "avg_iou"]:
                    if metric in m["span_metrics"][span_type]:
                        average_span_metrics[span_type][metric].append(m["span_metrics"][span_type][metric])

        # Calculate average values for each category
        for key in average_metrics:
            for metric in ["precision", "recall", "f1"]:
                average_metrics[key][metric] = np.mean(average_metrics[key][metric])

        for metric in average_rouge:
            for score_type in ["precision", "recall", "fmeasure"]:
                if average_rouge[metric][score_type]:
                    average_rouge[metric][score_type] = np.mean(average_rouge[metric][score_type])

        # Calculate average span metrics
        for span_type in average_span_metrics:
            for metric in average_span_metrics[span_type]:
                if average_span_metrics[span_type][metric]:
                    average_span_metrics[span_type][metric] = np.mean(average_span_metrics[span_type][metric])

        # Create summary
        summary = {
            "total_records": len(data),
            "valid_records": len(all_metrics),
            "valid_rouge_records": len(all_rouge_scores),
            "overall_metrics": overall_metrics,
            "average_metrics": average_metrics,
            "average_rouge": average_rouge,
            "overall_span_metrics": overall_span_metrics,  # New: overall span metrics
            "average_span_metrics": average_span_metrics   # New: average span metrics
        }

        # Save original data (with metrics)
        output_data_file = f"{os.path.splitext(output_file)[0]}_data.json"
        with open(output_data_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        # Save summary
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        logger.info(f"Analysis completed, summary saved to: {output_file}")
        logger.info(f"Data with metrics saved to: {output_data_file}")

        # Print summary
        logger.info("\nSummary:")
        logger.info(f"Total records: {summary['total_records']}")
        logger.info(f"Valid records: {summary['valid_records']}")
        logger.info(f"Valid ROUGE records: {summary['valid_rouge_records']}")

        # Print metrics for each category
        for category in ["positive", "negative", "neutral", "overall"]:
            logger.info(f"\n{category.capitalize()} scoring:")
            logger.info("Overall metrics:")
            logger.info(f"  Precision: {overall_metrics[category]['precision']:.4f}")
            logger.info(f"  Recall: {overall_metrics[category]['recall']:.4f}")
            logger.info(f"  F1: {overall_metrics[category]['f1']:.4f}")
            logger.info("Average metrics:")
            logger.info(f"  Precision: {average_metrics[category]['precision']:.4f}")
            logger.info(f"  Recall: {average_metrics[category]['recall']:.4f}")
            logger.info(f"  F1: {average_metrics[category]['f1']:.4f}")

        # Print span metrics
        logger.info("\nSpan metrics:")
        for span_type in ["good_span", "poor_span"]:
            logger.info(f"\n{span_type.replace('_', ' ').capitalize()}:")
            logger.info("Overall metrics:")
            logger.info(f"   Total count: GT={overall_span_metrics[span_type]['gt_count']}, Predicted={overall_span_metrics[span_type]['ex_count']}")
            logger.info(f"   Exact match: {overall_span_metrics[span_type]['exact_match']}")
            logger.info(f"   Partial match: {overall_span_metrics[span_type]['partial_match']}")
            logger.info(f"  Precision: {overall_span_metrics[span_type]['precision']:.4f}")
            logger.info(f"  Recall: {overall_span_metrics[span_type]['recall']:.4f}")
            logger.info(f"  F1: {overall_span_metrics[span_type]['f1']:.4f}")
            logger.info("Average metrics:")
            logger.info(f"  Precision: {average_span_metrics[span_type]['precision']:.4f}")
            logger.info(f"  Recall: {average_span_metrics[span_type]['recall']:.4f}")
            logger.info(f"  F1: {average_span_metrics[span_type]['f1']:.4f}")
            logger.info(f"   Average IOU: {average_span_metrics[span_type]['avg_iou']:.4f}")

        # Print ROUGE metrics
        logger.info("\nROUGE metrics:")
        for metric in ["rouge1", "rouge2", "rougeL"]:
            logger.info(f"\n{metric.upper()}:")
            logger.info(f"  Precision: {average_rouge[metric]['precision']:.4f}")
            logger.info(f"  Recall: {average_rouge[metric]['recall']:.4f}")
            logger.info(f"  F-measure: {average_rouge[metric]['fmeasure']:.4f}")

        # Sort records by overall F1 value
        sorted_metrics = sorted(all_metrics, key=lambda x: x["metrics"]["overall"]["f1"])

        # Print indices of lowest 5 records with lowest F1
        logger.info("\nIndices of lowest 5 records with lowest F1:")
        for m in sorted_metrics[:min(5, len(sorted_metrics))]:
            logger.info(f"   Record {m['index']}: F1 = {m['metrics']['overall']['f1']:.4f}")

        # Print indices of highest 5 records with highest F1
        logger.info("\nIndices of highest 5 records with highest F1:")
        for m in sorted_metrics[-min(5, len(sorted_metrics)):]:
            logger.info(f"   Record {m['index']}: F1 = {m['metrics']['overall']['f1']:.4f}")

        # Sort records by span F1 value
        if all_span_metrics:
            # Sort by good span F1
            sorted_good_span = sorted(all_span_metrics, key=lambda x: x["span_metrics"]["good_span"]["f1"] if "f1" in x["span_metrics"]["good_span"] else 0)

            logger.info("\nGood Span F1 values of lowest 5 records:")
            for m in sorted_good_span[:min(5, len(sorted_good_span))]:
                f1 = m["span_metrics"]["good_span"].get("f1", 0)
                logger.info(f"   Record {m['index']}: Good Span F1 = {f1:.4f}")

            logger.info("\nGood Span F1 values of highest 5 records:")
            for m in sorted_good_span[-min(5, len(sorted_good_span)):]:
                f1 = m["span_metrics"]["good_span"].get("f1", 0)
                logger.info(f"   Record {m['index']}: Good Span F1 = {f1:.4f}")

            # Sort by poor span F1
            sorted_poor_span = sorted(all_span_metrics, key=lambda x: x["span_metrics"]["poor_span"]["f1"] if "f1" in x["span_metrics"]["poor_span"] else 0)

            logger.info("\nPoor Span F1 values of lowest 5 records:")
            for m in sorted_poor_span[:min(5, len(sorted_poor_span))]:
                f1 = m["span_metrics"]["poor_span"].get("f1", 0)
                logger.info(f"   Record {m['index']}: Poor Span F1 = {f1:.4f}")

            logger.info("\nPoor Span F1 values of highest 5 records:")
            for m in sorted_poor_span[-min(5, len(sorted_poor_span)):]:
                f1 = m["span_metrics"]["poor_span"].get("f1", 0)
                logger.info(f"   Record {m['index']}: Poor Span F1 = {f1:.4f}")

        # If there are ROUGE records, sort by ROUGE-L F-measure and print highest/lowest records
        if all_rouge_scores:
            sorted_rouge = sorted(all_rouge_scores, key=lambda x: x["rouge_scores"]["rougeL"]["fmeasure"])

            logger.info("\nROUGE-L F-measure lowest 5 records:")
            for m in sorted_rouge[:min(5, len(sorted_rouge))]:
                logger.info(f"   Record {m['index']}: ROUGE-L = {m['rouge_scores']['rougeL']['fmeasure']:.4f}")

            logger.info("\nROUGE-L F-measure highest 5 records:")
            for m in sorted_rouge[-min(5, len(sorted_rouge)):]:
                logger.info(f"   Record {m['index']}: ROUGE-L = {m['rouge_scores']['rougeL']['fmeasure']:.4f}")

    except Exception as e:
        logger.error(f"Error analyzing file: {str(e)}", exc_info=True)
        raise

def main():
    parser = argparse.ArgumentParser(description="Calculate word-level score and textual feedback evaluation metrics")
    parser.add_argument("--input", default="abc.json", help="Input file path")
    parser.add_argument("--output", default="word_metrics_summary.json", help="Output file path")
    args = parser.parse_args()

    evaluate_file(args.input, args.output)

if __name__ == "__main__":
    main()