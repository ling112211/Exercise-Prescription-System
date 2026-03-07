"""
Expert pilot evaluation: compute mean scores with 95% CI and draw grouped bar chart.

Expected inputs are three CSV files corresponding to:
- Base model
- EPS without D2
- EPS

Each CSV should contain 7 question columns (Q1..Q7) with A/B/C/D grades (or 0-3 scores).
Optionally, a rater identifier column can be provided for paired tests.
"""

from __future__ import annotations

import argparse
import math
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


DIMENSIONS: List[str] = [
    "consensus",
    "correctness",
    "completeness",
    "unbiasedness",
    "clarity",
    "empathy",
    "actionability",
]
QNUM_TO_DIM: Dict[int, str] = {i + 1: dim for i, dim in enumerate(DIMENSIONS)}
LETTER_TO_SCORE: Dict[str, int] = {"A": 0, "B": 1, "C": 2, "D": 3}

LETTER_RE = re.compile(r"^\s*([ABCD])\s*[\.:\)\]]?\s*$", flags=re.IGNORECASE)
NUM_RE = re.compile(r"^\s*([0-3])(\.0+)?\s*$")


def read_csv_robust(path: Path) -> pd.DataFrame:
    for enc in ("utf-8-sig", "utf-8", "gb18030"):
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path)


def parse_score(x) -> float:
    if pd.isna(x):
        return np.nan

    if isinstance(x, (int, float)) and not (isinstance(x, float) and math.isnan(x)):
        v = float(x)
        if 0.0 <= v <= 3.0:
            return v

    s = str(x).strip()
    m = LETTER_RE.match(s)
    if m:
        return float(LETTER_TO_SCORE[m.group(1).upper()])

    m = NUM_RE.match(s)
    if m:
        return float(m.group(1))

    return np.nan


def pick_q1_to_q7_cols(df: pd.DataFrame) -> Dict[int, str]:
    """
    Find columns for questions 1..7.

    Supported header patterns (case-insensitive):
    - "1." "2." ... "7." (also supports full-width dot)
    - "Q1" "Q2" ... "Q7"
    - "1" "2" ... "7"
    """
    mapping: Dict[int, str] = {}

    for c in df.columns:
        s = str(c).strip()

        m = re.match(r"^([1-7])\s*\.\s*", s)
        if m:
            q = int(m.group(1))
            mapping.setdefault(q, c)
            continue

        m = re.match(r"^[Qq]\s*([1-7])\s*$", s)
        if m:
            q = int(m.group(1))
            mapping.setdefault(q, c)
            continue

        m = re.match(r"^([1-7])\s*$", s)
        if m:
            q = int(m.group(1))
            mapping.setdefault(q, c)
            continue

    missing = [q for q in range(1, 8) if q not in mapping]
    if missing:
        raise ValueError(
            f"Cannot find question columns for Q{missing}. "
            "Expected headers like '1.'..'7.' or 'Q1'..'Q7'."
        )
    return mapping


def t_critical(alpha: float, df: int) -> float:
    try:
        from scipy.stats import t  # type: ignore
        return float(t.ppf(1.0 - alpha / 2.0, df=df))
    except Exception:
        return 1.959963984540054


def mean_ci_t(x: np.ndarray, alpha: float = 0.05) -> Tuple[float, float, float, int, float]:
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n = int(x.size)
    if n < 2:
        return np.nan, np.nan, np.nan, n, np.nan

    mean = float(np.mean(x))
    sd = float(np.std(x, ddof=1))
    se = sd / math.sqrt(n)
    tcrit = t_critical(alpha=alpha, df=n - 1)
    lo = mean - tcrit * se
    hi = mean + tcrit * se
    return mean, lo, hi, n, sd


def holm_adjust(pvals: Iterable[float]) -> List[float]:
    p = np.asarray(list(pvals), dtype=float)
    m = p.size
    order = np.argsort(p)
    adj = np.empty(m, dtype=float)
    prev = 0.0
    for k, idx in enumerate(order):
        rank = k + 1
        val = (m - rank + 1) * p[idx]
        val = min(1.0, val)
        val = max(prev, val)
        adj[idx] = val
        prev = val
    return adj.tolist()


def infer_pair_key(dfs: Dict[str, pd.DataFrame], user_key: Optional[str]) -> Optional[str]:
    if user_key:
        return user_key if all(user_key in df.columns for df in dfs.values()) else None

    candidates = ["rater_id", "id", "ID", "RaterID", "participant_id"]
    for key in candidates:
        if all(key in df.columns for df in dfs.values()):
            return key
    return None


def summarize_model(df: pd.DataFrame, label: str) -> pd.DataFrame:
    qcols = pick_q1_to_q7_cols(df)
    rows = []
    for q in range(1, 8):
        dim = QNUM_TO_DIM[q]
        col = qcols[q]
        scores = df[col].map(parse_score).astype(float)
        mean, lo, hi, n, sd = mean_ci_t(scores.values, alpha=0.05)
        rows.append(
            {
                "model": label,
                "dimension": dim,
                "question": q,
                "n": n,
                "mean": mean,
                "sd": sd,
                "ci_low": lo,
                "ci_high": hi,
            }
        )
    return pd.DataFrame(rows).sort_values("question")


def aligned_scores_for_dim(
    dfs: Dict[str, pd.DataFrame],
    labels: List[str],
    dim: str,
    pair_key: Optional[str],
) -> Dict[str, np.ndarray]:
    q = DIMENSIONS.index(dim) + 1
    out: Dict[str, np.ndarray] = {}

    if pair_key and all(pair_key in dfs[l].columns for l in labels):
        frames = []
        for lab in labels:
            df = dfs[lab].copy()
            qcol = pick_q1_to_q7_cols(df)[q]
            df = df[[pair_key, qcol]].copy()
            df.rename(columns={qcol: lab}, inplace=True)
            df[lab] = df[lab].map(parse_score).astype(float)
            frames.append(df)

        merged = frames[0]
        for f in frames[1:]:
            merged = merged.merge(f, on=pair_key, how="inner")

        for lab in labels:
            out[lab] = merged[lab].to_numpy(dtype=float)
        return out

    for lab in labels:
        df = dfs[lab]
        qcol = pick_q1_to_q7_cols(df)[q]
        out[lab] = df[qcol].map(parse_score).to_numpy(dtype=float)
    return out


def pairwise_tests(
    dfs: Dict[str, pd.DataFrame],
    labels: List[str],
    pair_key: Optional[str],
) -> pd.DataFrame:
    pairs = [
        ("EPS", "EPS without D2"),
        ("EPS without D2", "Base model"),
        ("EPS", "Base model"),
    ]

    rows = []
    for dim in DIMENSIONS:
        scores = aligned_scores_for_dim(dfs=dfs, labels=labels, dim=dim, pair_key=pair_key)

        for a, b in pairs:
            xa = scores[a]
            xb = scores[b]

            if pair_key:
                mask = ~np.isnan(xa) & ~np.isnan(xb)
                xa2 = xa[mask]
                xb2 = xb[mask]
                n = int(mask.sum())
                p = np.nan
                stat = np.nan
                test_name = "paired_t"
                if n >= 2:
                    try:
                        from scipy.stats import ttest_rel  # type: ignore
                        stat, p = ttest_rel(xa2, xb2, nan_policy="omit")
                    except Exception:
                        stat, p = np.nan, np.nan
                diff = float(np.nanmean(xa2) - np.nanmean(xb2)) if n > 0 else np.nan
            else:
                xa2 = xa[~np.isnan(xa)]
                xb2 = xb[~np.isnan(xb)]
                n = int(min(xa2.size, xb2.size))
                p = np.nan
                stat = np.nan
                test_name = "welch_t"
                if xa2.size >= 2 and xb2.size >= 2:
                    try:
                        from scipy.stats import ttest_ind  # type: ignore
                        stat, p = ttest_ind(xa2, xb2, equal_var=False, nan_policy="omit")
                    except Exception:
                        stat, p = np.nan, np.nan
                diff = float(np.nanmean(xa2) - np.nanmean(xb2)) if (xa2.size > 0 and xb2.size > 0) else np.nan

            rows.append(
                {
                    "dimension": dim,
                    "comparison": f"{a} - {b}",
                    "test": test_name,
                    "n_used": n,
                    "mean_diff": diff,
                    "t_stat": float(stat) if stat is not None else np.nan,
                    "p_value": float(p) if p is not None else np.nan,
                }
            )

    out = pd.DataFrame(rows)
    out["p_holm"] = np.nan

    for comp in out["comparison"].unique():
        m = out["comparison"] == comp
        pvals = out.loc[m, "p_value"].to_list()
        if all(pd.notna(pvals)):
            out.loc[m, "p_holm"] = holm_adjust(pvals)
    return out


def configure_matplotlib_fonts() -> None:
    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.size": 10,
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "legend.fontsize": 10,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "mathtext.fontset": "custom",
            "mathtext.rm": "Times New Roman",
            "mathtext.it": "Times New Roman:italic",
            "mathtext.bf": "Times New Roman:bold",
        }
    )


def plot_grouped_bars(
    summary: pd.DataFrame,
    out_pdf: Path,
    out_png: Path,
    show: bool,
) -> None:
    dim_order = DIMENSIONS
    bar_order = ["Base model", "EPS without D2", "EPS"]

    pivot_mean = summary.pivot(index="dimension", columns="model", values="mean").loc[dim_order, bar_order]
    pivot_lo = summary.pivot(index="dimension", columns="model", values="ci_low").loc[dim_order, bar_order]
    pivot_hi = summary.pivot(index="dimension", columns="model", values="ci_high").loc[dim_order, bar_order]

    means = pivot_mean.values
    yerr_lower = means - pivot_lo.values
    yerr_upper = pivot_hi.values - means
    yerr = np.stack([yerr_lower, yerr_upper], axis=0)

    facecolors = ["#E59693", "#FCD7AF", "#989CC8"]
    errcolors = ["#C93735", "#F8AB61", "#333A8C"]

    n_dims = len(dim_order)
    n_models = len(bar_order)
    x = np.arange(n_dims)
    width = 0.24

    fig, ax = plt.subplots(figsize=(10.5, 4.2), dpi=300)

    for j, model in enumerate(bar_order):
        ax.bar(
            x + (j - (n_models - 1) / 2) * width,
            pivot_mean[model].values,
            width=width,
            label=model,
            color=facecolors[j],
            edgecolor="none",
            linewidth=0.0,
            yerr=yerr[:, :, j],
            capsize=3,
            error_kw={"elinewidth": 1.0, "capthick": 1.0, "ecolor": errcolors[j]},
        )

    ax.set_ylabel("Evaluation scores")
    ax.set_ylim(0, 3.05)
    ax.set_xticks(x)
    ax.set_xticklabels([d.capitalize() for d in dim_order], rotation=25, ha="right")

    ax.set_axisbelow(True)
    ax.yaxis.grid(True, linewidth=0.6, alpha=0.30)
    ax.xaxis.grid(False)

    handles, labels = ax.get_legend_handles_labels()
    legend_order = [labels.index("EPS"), labels.index("EPS without D2"), labels.index("Base model")]
    ax.legend(
        [handles[i] for i in legend_order],
        [labels[i] for i in legend_order],
        frameon=False,
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.18),
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Plot expert pilot evaluation (grouped bar chart with 95% CI).")
    p.add_argument("--base", type=str, default=None, help="Path to base model CSV.")
    p.add_argument("--d1", type=str, default=None, help="Path to EPS without D2 CSV.")
    p.add_argument("--eps", type=str, default=None, help="Path to EPS CSV.")
    p.add_argument(
        "--input-dir",
        type=str,
        default=None,
        help="Directory containing base_model.csv, eps_without_d2.csv, eps.csv (used if explicit paths are not set).",
    )
    p.add_argument("--pair-key", type=str, default=None, help="Column name used to align raters for paired tests.")
    p.add_argument("--no-tests", action="store_true", help="Disable pairwise significance tests.")
    p.add_argument("--no-show", action="store_true", help="Do not display the plot window.")
    p.add_argument("--outdir", type=str, default=".", help="Output directory.")
    p.add_argument("--prefix", type=str, default="expert_pilot", help="Output filename prefix.")
    return p


def resolve_inputs(args: argparse.Namespace) -> Dict[str, Path]:
    repo_root = Path(__file__).resolve().parents[1]
    input_dir = Path(args.input_dir) if args.input_dir else (repo_root / "data" / "expert_pilot")

    base = Path(args.base) if args.base else (input_dir / "base_model.csv")
    d1 = Path(args.d1) if args.d1 else (input_dir / "eps_without_d2.csv")
    eps = Path(args.eps) if args.eps else (input_dir / "eps.csv")

    missing = [str(p) for p in (base, d1, eps) if not p.exists()]
    if missing:
        msg = (
            "Missing input file(s):\n- " + "\n- ".join(missing) + "\n\n"
            "Provide explicit paths via --base/--d1/--eps, or place files under data/expert_pilot/:\n"
            "  base_model.csv\n  eps_without_d2.csv\n  eps.csv\n"
        )
        raise FileNotFoundError(msg)

    return {"Base model": base, "EPS without D2": d1, "EPS": eps}


def main() -> int:
    args = build_parser().parse_args()
    configure_matplotlib_fonts()

    try:
        paths = resolve_inputs(args)
    except Exception as e:
        print(str(e), file=sys.stderr)
        return 2

    dfs: Dict[str, pd.DataFrame] = {}
    for label, path in paths.items():
        df = read_csv_robust(path)
        dfs[label] = df

    labels = ["Base model", "EPS without D2", "EPS"]
    pair_key = infer_pair_key(dfs=dfs, user_key=args.pair_key)

    summaries = []
    for label in labels:
        df = dfs[label].copy()
        if pair_key and pair_key in df.columns:
            df = df.dropna(subset=[pair_key]).copy()
        summaries.append(summarize_model(df, label))
    summary = pd.concat(summaries, ignore_index=True)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    out_summary = outdir / f"{args.prefix}_means_ci.csv"
    summary.to_csv(out_summary, index=False, encoding="utf-8-sig")

    if not args.no_tests:
        tests = pairwise_tests(dfs=dfs, labels=labels, pair_key=pair_key)
        out_tests = outdir / f"{args.prefix}_pairwise_tests.csv"
        tests.to_csv(out_tests, index=False, encoding="utf-8-sig")
    else:
        out_tests = None

    out_pdf = outdir / f"{args.prefix}_bar_mean_ci.pdf"
    out_png = outdir / f"{args.prefix}_bar_mean_ci.png"
    plot_grouped_bars(summary=summary, out_pdf=out_pdf, out_png=out_png, show=(not args.no_show))

    print(f"Saved summary: {out_summary}")
    if out_tests:
        print(f"Saved tests:   {out_tests}")
        if pair_key:
            print(f"Tests used paired t-test aligned by '{pair_key}'.")
        else:
            print("Tests used Welch's t-test (unpaired).")
    print(f"Saved figure:  {out_pdf}")
    print(f"Saved figure:  {out_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())