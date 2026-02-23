import json
import math
import re
from statistics import stdev
from typing import Dict, List, Tuple

def parse_answer_from_text(text: str) -> str:
    """
    Extracts the multiple choice answer (A-E) from the model's output text.
    Prioritizes '答案：X' format, falls back to searching for the last valid character.
    """
    # Match "答案：A" or "答案: A"
    m = re.search(r"答案[:：]\s*([A-E])\b", text, flags=re.IGNORECASE)
    if m:
        return m.group(1).upper()
    
    # Fallback: look for the last occurrence of A-E
    for ch in reversed(text):
        if ch in "ABCDE":
            return ch
    return ""

def format_options(option_dict: Dict[str, str]) -> str:
    """
    Formats a dictionary of options into a string.
    Example: {'A': 'Option 1', 'B': 'Option 2'} -> "A. Option 1\nB. Option 2"
    """
    return "\n".join([f"{key}. {val}" for key, val in option_dict.items()])

def calculate_metrics(run_accuracies: List[float], alpha: float = 0.05) -> Tuple[float, float, float, int, float]:
    """
    Calculates Mean, 95% CI (Lower, Upper), N, and Standard Deviation.
    
    Args:
        run_accuracies: List of accuracy scores (floats) from multiple runs.
        alpha: Significance level (default 0.05 for 95% CI).
        
    Returns:
        (mean, ci_lower, ci_upper, n_runs, std_dev)
    """
    n = len(run_accuracies)
    if n == 0:
        return 0.0, 0.0, 0.0, 0, 0.0
        
    m = sum(run_accuracies) / n
    
    if n > 1:
        s = stdev(run_accuracies)
        se = s / math.sqrt(n)
        
        try:
            from scipy.stats import t
            # Two-tailed t-test critical value
            tcrit = t.ppf(1 - alpha / 2, df=n - 1)
        except ImportError:
            # Fallback if scipy is not installed, approx for large N or use 1.96 for Z
            # For small N (e.g., 5 or 10), 1.96 is an underestimate, but acceptable if scipy missing
            tcrit = 1.96
            
        lo = m - tcrit * se
        hi = m + tcrit * se
    else:
        s = 0.0
        lo = m
        hi = m
        
    return m, lo, hi, n, s

def load_run_results(json_path: str) -> List[float]:
    """
    Loads the 'accuracy' field from each run in the result JSON file.
    """
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # Check if it's the summary format
        if "runs" in data:
            return [r["accuracy"] for r in data["runs"]]
        
        # Fallback or other formats could be handled here
        return []
    except FileNotFoundError:
        print(f"Warning: File not found: {json_path}")
        return []
