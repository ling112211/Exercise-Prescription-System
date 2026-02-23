import os
import json
import numpy as np
from typing import Dict, List
from utils import load_run_results, calculate_metrics

# ---------------- Configuration ----------------

# Define the models and their corresponding result file paths.
# The paths should be relative to the project root or absolute paths.
# Assuming a directory structure like: data/results/{model_category}/{model_name}/{benchmark}/result.json
# You can adjust the base path logic below.

RESULTS_BASE_DIR = "data/results"  # Placeholder: Update this to your actual results directory

MODELS = {
    "Base": [
        "DeepSeek-R1-8B", 
        "Qwen3-8B", 
        "DeepSeek-R1-14B", 
        "Qwen3-14B"
    ],
    "EPS_no_D1": [
        "EPS-noD1-DeepSeek-R1-8B", 
        "EPS-noD1-Qwen3-8B", 
        "EPS-noD1-DeepSeek-R1-14B", 
        "EPS-noD1-Qwen3-14B"
    ],
    "EPS": [
        "EPS-DeepSeek-R1-8B", 
        "EPS-Qwen3-8B", 
        "EPS-DeepSeek-R1-14B", 
        "EPS-Qwen3-14B"
    ],
    "Flagship": [
        "ChatGPT-5", 
        "DeepSeek-R1", 
        "Gemini-2.5-Flash", 
        "Grok-4-Fast"
    ]
}

BENCHMARKS = ["CMB", "CMExam", "MedMCQA", "MedQA"]

def get_result_path(model_category: str, model_name: str, benchmark: str) -> str:
    """
    Constructs the file path for a specific model and benchmark result.
    Adjust this function based on your actual file organization.
    """
    # Example path: data/results/Base/Qwen3-14B/CMB/result.json
    return os.path.join(RESULTS_BASE_DIR, model_category, model_name, benchmark, "result.json")

def evaluate_all():
    print(f"{'Category':<12} | {'Model':<25} | {'Benchmark':<10} | {'Accuracy':<8} | {'95% CI':<15} | {'Runs':<4}")
    print("-" * 85)

    summary_data = []

    for category, model_list in MODELS.items():
        for model_name in model_list:
            for benchmark in BENCHMARKS:
                # Construct the path to the result file
                result_path = get_result_path(category, model_name, benchmark)
                
                # Load accuracies from the result file
                accuracies = load_run_results(result_path)
                
                if not accuracies:
                    # print(f"{category:<12} | {model_name:<25} | {benchmark:<10} | {'N/A':<8} | {'N/A':<15} | {0:<4}")
                    continue

                # Calculate metrics
                mean_acc, ci_lo, ci_hi, n_runs, std_dev = calculate_metrics(accuracies)
                
                # Print formatted row
                ci_str = f"[{ci_lo:.4f}, {ci_hi:.4f}]"
                print(f"{category:<12} | {model_name:<25} | {benchmark:<10} | {mean_acc:.4f}   | {ci_str:<15} | {n_runs:<4}")

                summary_data.append({
                    "category": category,
                    "model": model_name,
                    "benchmark": benchmark,
                    "accuracy": mean_acc,
                    "ci_lower": ci_lo,
                    "ci_upper": ci_hi,
                    "n_runs": n_runs,
                    "std_dev": std_dev
                })

    # Optionally save the summary to a CSV or JSON
    save_summary(summary_data)

def save_summary(data: List[Dict], filename="benchmark_summary.json"):
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"\nSummary saved to {filename}")

if __name__ == "__main__":
    evaluate_all()
