import argparse
import math
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =========================
# Plot style
# =========================
def configure_matplotlib() -> None:
    plt.rcParams.update({
        "font.family": "Times New Roman",
        "font.size": 10,
        "axes.labelsize": 10,
        "legend.fontsize": 10,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "mathtext.fontset": "custom",
        "mathtext.rm": "Times New Roman",
        "mathtext.it": "Times New Roman:italic",
        "mathtext.bf": "Times New Roman:bold",
    })


# =========================
# Column detection
# =========================
Q1_PATTERN = re.compile(r"^\s*1\s*[\.．、。:：\)]")
Q2_15_PATTERN = re.compile(r"^\s*(?:[2-9]|1[0-5])\s*[\.．、。:：\)]")


def normalize_colname(x: object) -> str:
    return str(x).strip()


def pick_question_cols(df: pd.DataFrame, pattern: re.Pattern) -> List[str]:
    cols = []
    for c in df.columns:
        s = normalize_colname(c)
        if pattern.match(s):
            cols.append(c)
    return cols


def qnum(colname: object) -> int:
    s = normalize_colname(colname)
    m = re.match(r"^\s*(\d+)\s*[\.．、。:：\)]", s)
    return int(m.group(1)) if m else 10**9


def find_single_col(df: pd.DataFrame, pattern: re.Pattern) -> str:
    cols = pick_question_cols(df, pattern)
    if len(cols) != 1:
        raise ValueError(f"Expected exactly one column matching pattern, got {len(cols)}: {cols}")
    return cols[0]


# =========================
# Parsing helpers
# =========================
def parse_yes_no(x: object) -> float:
    """
    Parse screening item (Q1).
    Returns 1.0 for "A" (yes), 0.0 for "B" (no), otherwise NaN.
    """
    if pd.isna(x):
        return np.nan
    s = str(x).strip()
    m = re.match(r"^([AB])\s*[\.\．、。:：\)\）]?", s)
    if m:
        return 1.0 if m.group(1) == "A" else 0.0

    # Heuristic for common Chinese yes/no tokens (kept as data parsing rules).
    if ("是" in s) and ("否" not in s):
        return 1.0
    if ("否" in s) and ("是" not in s):
        return 0.0
    return np.nan


def parse_likert_1_7(x: object) -> float:
    """
    Parse a 1–7 Likert response.
    Accepts raw numeric values or strings like "5", "5.XXX".
    """
    if pd.isna(x):
        return np.nan
    if isinstance(x, (int, float, np.integer, np.floating)):
        v = float(x)
        return v if 1.0 <= v <= 7.0 else np.nan

    s = str(x).strip()
    m = re.match(r"^\s*(-?\d+(?:\.\d+)?)", s)
    if not m:
        return np.nan
    v = float(m.group(1))
    return v if 1.0 <= v <= 7.0 else np.nan


# =========================
# Stats helpers
# =========================
def mean_ci_t(x: np.ndarray, alpha: float = 0.05) -> Tuple[float, float, float, int, float]:
    """
    Mean and two-sided (1-alpha) CI using Student t distribution.
    Returns (mean, ci_low, ci_high, n, sd).
    """
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n = int(x.size)
    if n < 2:
        return np.nan, np.nan, np.nan, n, np.nan

    mean = float(np.mean(x))
    sd = float(np.std(x, ddof=1))
    se = sd / float(np.sqrt(n))

    try:
        from scipy.stats import t as t_dist
        tcrit = float(t_dist.ppf(1.0 - alpha / 2.0, df=n - 1))
    except Exception:
        tcrit = 1.959963984540054

    ci_low = mean - tcrit * se
    ci_high = mean + tcrit * se
    return mean, ci_low, ci_high, n, sd


def welch_t_pvalue(x: np.ndarray, y: np.ndarray) -> float:
    """
    Two-sided Welch's t-test p-value.
    Uses scipy if available; otherwise a normal approximation fallback.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x = x[~np.isnan(x)]
    y = y[~np.isnan(y)]
    if x.size < 2 or y.size < 2:
        return np.nan

    try:
        from scipy.stats import ttest_ind
        return float(ttest_ind(x, y, equal_var=False, nan_policy="omit").pvalue)
    except Exception:
        mx, my = float(np.mean(x)), float(np.mean(y))
        vx, vy = float(np.var(x, ddof=1)), float(np.var(y, ddof=1))
        denom = math.sqrt(vx / x.size + vy / y.size)
        if denom <= 0:
            return np.nan
        t = (mx - my) / denom
        p = math.erfc(abs(t) / math.sqrt(2.0))
        return float(p)


def holm_adjust(pvals: np.ndarray) -> np.ndarray:
    """
    Holm step-down adjustment.
    Preserves NaNs: NaN inputs map to NaN outputs.
    """
    pvals = np.asarray(pvals, dtype=float)
    out = np.full_like(pvals, np.nan, dtype=float)

    mask = ~np.isnan(pvals)
    if not np.any(mask):
        return out

    pv = pvals[mask]
    m = int(pv.size)
    order = np.argsort(pv)

    adj = np.empty(m, dtype=float)
    running_max = 0.0
    for k, idx in enumerate(order):
        p = float(pv[idx])
        adj_p = (m - k) * p
        running_max = max(running_max, adj_p)
        adj[idx] = min(running_max, 1.0)

    out[mask] = adj
    return out


# =========================
# QC helpers
# =========================
def try_find_id_col(df: pd.DataFrame) -> Optional[str]:
    candidates = ["编号", "ID", "Id", "id", "participant_id", "user_id"]
    for c in df.columns:
        if normalize_colname(c) in candidates:
            return c
    return None


def try_find_time_col(df: pd.DataFrame) -> Optional[str]:
    """
    Tries to locate a completion-time column by common name patterns.
    """
    patterns = [
        r"(?i)\b(duration|time\s*spent|elapsed)\b",
        r"用时",
        r"耗时",
        r"完成时间",
        r"答题时间",
        r"填写时间",
    ]
    for c in df.columns:
        name = normalize_colname(c)
        for pat in patterns:
            if re.search(pat, name):
                return c
    return None


def infer_time_seconds(values: pd.Series, colname: str) -> pd.Series:
    """
    Convert a time column to seconds using a conservative heuristic.
    If units are ambiguous, assumes seconds.
    """
    s = pd.to_numeric(values, errors="coerce")
    name = normalize_colname(colname).lower()

    if ("min" in name) or ("minute" in name) or ("分钟" in name):
        return s * 60.0
    if ("sec" in name) or ("second" in name) or ("秒" in name):
        return s

    # Heuristic by magnitude: large medians are likely seconds.
    med = float(np.nanmedian(s.to_numpy(dtype=float))) if np.isfinite(np.nanmedian(s.to_numpy(dtype=float))) else np.nan
    if np.isnan(med):
        return s
    if med <= 30.0:
        # Could be minutes; keep conservative and treat as minutes.
        return s * 60.0
    return s


def drop_straightliners(df: pd.DataFrame, q_cols: List[str], min_answered: int = 8) -> Tuple[pd.DataFrame, int]:
    """
    Drops rows that select the same option for all answered items (Q2..Q15).
    Only considers rows with at least min_answered non-missing parsed responses.
    """
    parsed = df[q_cols].applymap(parse_likert_1_7)
    answered = parsed.notna().sum(axis=1)
    same = parsed.nunique(axis=1, dropna=True)
    mask_keep = ~((answered >= min_answered) & (same <= 1))
    removed = int((~mask_keep).sum())
    return df.loc[mask_keep].copy(), removed


def apply_time_filter(
    df: pd.DataFrame,
    time_col: str,
    min_time_s: float,
    max_time_s: float
) -> Tuple[pd.DataFrame, int]:
    """
    Drops rows outside [min_time_s, max_time_s] based on inferred seconds.
    """
    tsec = infer_time_seconds(df[time_col], time_col)
    mask_keep = (tsec >= min_time_s) & (tsec <= max_time_s)
    removed = int((~mask_keep).sum())
    return df.loc[mask_keep].copy(), removed


# =========================
# Core analysis
# =========================
def summarize_group(df: pd.DataFrame, q_cols: List[str], group_name: str) -> pd.DataFrame:
    rows = []
    for c in q_cols:
        vals = df[c].map(parse_likert_1_7).to_numpy(dtype=float)
        mean, lo, hi, n, sd = mean_ci_t(vals, alpha=0.05)
        rows.append({
            "group": group_name,
            "question": qnum(c),
            "col": normalize_colname(c),
            "n": n,
            "mean": mean,
            "sd": sd,
            "ci_low": lo,
            "ci_high": hi,
        })
    return pd.DataFrame(rows).sort_values("question").reset_index(drop=True)


def compute_item_pvalues(
    df_h: pd.DataFrame, df_e: pd.DataFrame,
    q_cols_h: List[str], q_cols_e: List[str]
) -> np.ndarray:
    pvals = []
    for ch, ce in zip(q_cols_h, q_cols_e):
        x = df_h[ch].map(parse_likert_1_7).to_numpy(dtype=float)
        y = df_e[ce].map(parse_likert_1_7).to_numpy(dtype=float)
        pvals.append(welch_t_pvalue(x, y))
    return np.asarray(pvals, dtype=float)


def make_wide_table(sum_h: pd.DataFrame, sum_e: pd.DataFrame, pvals: np.ndarray) -> pd.DataFrame:
    p_holm = holm_adjust(pvals)

    wide = (
        sum_h[["question", "n", "mean", "ci_low", "ci_high"]]
        .rename(columns={
            "n": "n_human",
            "mean": "mean_human",
            "ci_low": "ci_low_human",
            "ci_high": "ci_high_human",
        })
        .merge(
            sum_e[["question", "n", "mean", "ci_low", "ci_high"]]
            .rename(columns={
                "n": "n_eps",
                "mean": "mean_eps",
                "ci_low": "ci_low_eps",
                "ci_high": "ci_high_eps",
            }),
            on="question",
            how="inner",
        )
    )
    wide["p_value_welch"] = pvals
    wide["p_holm_14tests"] = p_holm
    return wide


# =========================
# Plotting
# =========================
DEFAULT_LABELS: Dict[int, str] = {
    2:  "Satisfaction",
    3:  "Pleasant interaction",
    4:  "Happiness after feedback",
    5:  "Calm & stable",
    6:  "Steady under pressure",
    7:  "Not easily triggered",
    8:  "Low emotional swings",
    9:  "Not anxious",
    10: "Consistent tone",
    11: "Often agrees",
    12: "Rarely disagrees",
    13: "Emphasizes agreement",
    14: "Often praises",
    15: "Highlights strengths",
}


def radar_plot_with_ci(
    sum_h: pd.DataFrame,
    sum_e: pd.DataFrame,
    out_pdf: Path,
    out_png: Path,
    labels: Dict[int, str],
    group_h_name: str,
    group_e_name: str,
) -> None:
    dim_qnums = list(range(2, 16))
    cat_labels = [labels[i] for i in dim_qnums]

    def grab(series_name: str, q: int) -> float:
        v = sum_h.loc[sum_h["question"] == q, series_name].values
        if v.size != 1:
            return np.nan
        return float(v[0])

    def grab_e(series_name: str, q: int) -> float:
        v = sum_e.loc[sum_e["question"] == q, series_name].values
        if v.size != 1:
            return np.nan
        return float(v[0])

    h_means = [grab("mean", q) for q in dim_qnums]
    h_lo = [grab("ci_low", q) for q in dim_qnums]
    h_hi = [grab("ci_high", q) for q in dim_qnums]

    e_means = [grab_e("mean", q) for q in dim_qnums]
    e_lo = [grab_e("ci_low", q) for q in dim_qnums]
    e_hi = [grab_e("ci_high", q) for q in dim_qnums]

    def close(arr: List[float]) -> List[float]:
        return arr + [arr[0]]

    N = len(dim_qnums)
    angles = np.linspace(0.0, 2.0 * np.pi, N, endpoint=False).tolist()
    angles_c = angles + [angles[0]]

    h_means_c = close(h_means)
    e_means_c = close(e_means)

    fig = plt.figure(figsize=(8.6, 8.6), dpi=200)
    ax = plt.subplot(111, polar=True)

    ax.set_theta_offset(np.pi / 2.0)
    ax.set_theta_direction(-1)

    ax.set_ylim(1, 7)
    ax.set_yticks([1, 2, 3, 4, 5, 6, 7])
    ax.set_yticklabels([])

    ax.yaxis.grid(True, linewidth=0.8, alpha=0.7)
    ax.xaxis.grid(True, linewidth=0.8, alpha=0.7)

    ax.set_xticks(angles)
    ax.set_xticklabels(cat_labels, fontfamily="Times New Roman")

    # Colors kept consistent with common publication defaults.
    col_h = "#1f77b4"
    col_e = "#d55e00"

    ax.plot(angles_c, h_means_c, color=col_h, linewidth=2.0, label=group_h_name)
    ax.fill(angles_c, h_means_c, color=col_h, alpha=0.12)

    ax.plot(angles_c, e_means_c, color=col_e, linewidth=2.0, label=group_e_name)
    ax.fill(angles_c, e_means_c, color=col_e, alpha=0.12)

    ax.scatter(angles, h_means, color=col_h, s=18, zorder=3)
    ax.scatter(angles, e_means, color=col_e, s=18, zorder=3)

    def draw_errorbars(
        angles_: List[float],
        lo_: List[float],
        hi_: List[float],
        color: str,
        cap: float = 0.03,
        lw: float = 1.3,
        alpha: float = 0.95
    ) -> None:
        for th, l, u in zip(angles_, lo_, hi_):
            if np.isnan(l) or np.isnan(u):
                continue
            ax.plot([th, th], [l, u], color=color, linewidth=lw, alpha=alpha, zorder=2)
            ax.plot([th - cap, th + cap], [l, l], color=color, linewidth=lw, alpha=alpha, zorder=2)
            ax.plot([th - cap, th + cap], [u, u], color=color, linewidth=lw, alpha=alpha, zorder=2)

    draw_errorbars(angles, h_lo, h_hi, col_h)
    draw_errorbars(angles, e_lo, e_hi, col_e)

    handles, names = ax.get_legend_handles_labels()
    ax.legend(handles, names, loc="upper right", bbox_to_anchor=(1.18, 1.10), frameon=False)

    fig.tight_layout()
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


# =========================
# CLI
# =========================
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Participant-reported outcomes (Phase 2) analysis and radar plot.")
    p.add_argument("--human-xlsx", type=str, required=True, help="Path to Human summary.xlsx")
    p.add_argument("--eps-xlsx", type=str, required=True, help="Path to EPS-human clean_responses.xlsx")
    p.add_argument("--sheet", type=str, default="0", help="Excel sheet name or index (default: 0)")

    p.add_argument("--outdir", type=str, default="outputs/questionnaire", help="Output directory")
    p.add_argument("--prefix", type=str, default="phase2", help="Output filename prefix")

    p.add_argument("--filter-used-only", action="store_true", help="Keep only respondents with Q1 == yes")
    p.add_argument("--no-filter-used-only", dest="filter_used_only", action="store_false")
    p.set_defaults(filter_used_only=True)

    p.add_argument("--time-filter", action="store_true", help="Enable completion-time QC if a time column is detected")
    p.add_argument("--no-time-filter", dest="time_filter", action="store_false")
    p.set_defaults(time_filter=True)
    p.add_argument("--min-time-sec", type=float, default=60.0, help="Minimum completion time in seconds")
    p.add_argument("--max-time-sec", type=float, default=3600.0, help="Maximum completion time in seconds")

    p.add_argument("--min-answered", type=int, default=8, help="Minimum answered items required to evaluate straight-lining")
    p.add_argument("--group-human", type=str, default="Human", help="Human group label for tables/plots")
    p.add_argument("--group-eps", type=str, default="EPS–human", help="EPS-human group label for tables/plots")

    return p.parse_args()


def read_sheet_arg(sheet_arg: str):
    if sheet_arg.isdigit():
        return int(sheet_arg)
    return sheet_arg


def main() -> None:
    args = parse_args()
    configure_matplotlib()

    human_xlsx = Path(args.human_xlsx)
    eps_xlsx = Path(args.eps_xlsx)
    if not human_xlsx.exists():
        raise FileNotFoundError(f"Human file not found: {human_xlsx}")
    if not eps_xlsx.exists():
        raise FileNotFoundError(f"EPS-human file not found: {eps_xlsx}")

    sheet = read_sheet_arg(args.sheet)

    df_h = pd.read_excel(human_xlsx, sheet_name=sheet)
    df_e = pd.read_excel(eps_xlsx, sheet_name=sheet)

    # Drop empty ID rows if possible.
    id_h = try_find_id_col(df_h)
    id_e = try_find_id_col(df_e)
    if id_h is not None:
        df_h = df_h.dropna(subset=[id_h]).copy()
        df_h = df_h.drop_duplicates(subset=[id_h], keep="first").copy()
    if id_e is not None:
        df_e = df_e.dropna(subset=[id_e]).copy()
        df_e = df_e.drop_duplicates(subset=[id_e], keep="first").copy()

    # Identify question columns.
    q1_h = find_single_col(df_h, Q1_PATTERN)
    q1_e = find_single_col(df_e, Q1_PATTERN)

    q_cols_h = sorted(pick_question_cols(df_h, Q2_15_PATTERN), key=qnum)
    q_cols_e = sorted(pick_question_cols(df_e, Q2_15_PATTERN), key=qnum)

    target = list(range(2, 16))
    if [qnum(c) for c in q_cols_h] != target:
        raise ValueError(f"Human file: expected Q2..Q15 columns, got {[qnum(c) for c in q_cols_h]}")
    if [qnum(c) for c in q_cols_e] != target:
        raise ValueError(f"EPS file: expected Q2..Q15 columns, got {[qnum(c) for c in q_cols_e]}")

    # Q1 screening filter.
    if args.filter_used_only:
        use_h = df_h[q1_h].map(parse_yes_no)
        use_e = df_e[q1_e].map(parse_yes_no)
        df_h = df_h.loc[use_h == 1.0].copy()
        df_e = df_e.loc[use_e == 1.0].copy()

    # Completion-time QC (optional, only if detected).
    removed_time_h = removed_time_e = 0
    if args.time_filter:
        tcol_h = try_find_time_col(df_h)
        tcol_e = try_find_time_col(df_e)
        if (tcol_h is not None) and (tcol_e is not None):
            df_h, removed_time_h = apply_time_filter(df_h, tcol_h, args.min_time_sec, args.max_time_sec)
            df_e, removed_time_e = apply_time_filter(df_e, tcol_e, args.min_time_sec, args.max_time_sec)

    # Straight-lining QC.
    df_h, removed_sl_h = drop_straightliners(df_h, q_cols_h, min_answered=args.min_answered)
    df_e, removed_sl_e = drop_straightliners(df_e, q_cols_e, min_answered=args.min_answered)

    # Summaries.
    sum_h = summarize_group(df_h, q_cols_h, args.group_human)
    sum_e = summarize_group(df_e, q_cols_e, args.group_eps)

    # P-values and Holm adjustment.
    pvals = compute_item_pvalues(df_h, df_e, q_cols_h, q_cols_e)
    wide = make_wide_table(sum_h, sum_e, pvals)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    out_wide_csv = outdir / f"{args.prefix}_radar_means_ci_with_p.csv"
    out_long_csv = outdir / f"{args.prefix}_radar_means_ci_long.csv"
    out_pdf = outdir / f"{args.prefix}_radar_mean_ci.pdf"
    out_png = outdir / f"{args.prefix}_radar_mean_ci.png"

    wide.to_csv(out_wide_csv, index=False, encoding="utf-8-sig")
    pd.concat([sum_h, sum_e], ignore_index=True).to_csv(out_long_csv, index=False, encoding="utf-8-sig")

    # Plot.
    radar_plot_with_ci(
        sum_h=sum_h,
        sum_e=sum_e,
        out_pdf=out_pdf,
        out_png=out_png,
        labels=DEFAULT_LABELS,
        group_h_name=args.group_human,
        group_e_name=args.group_eps,
    )

    # Console report.
    print("=== Participant-reported outcomes (Phase 2) ===")
    print(f"Human: n={len(df_h)} | removed_time={removed_time_h} | removed_straightline={removed_sl_h}")
    print(f"EPS  : n={len(df_e)} | removed_time={removed_time_e} | removed_straightline={removed_sl_e}")
    print(f"Saved: {out_wide_csv}")
    print(f"Saved: {out_long_csv}")
    print(f"Saved: {out_pdf}")
    print(f"Saved: {out_png}")
    print()
    print(wide[["question", "mean_human", "mean_eps", "p_value_welch", "p_holm_14tests"]].to_string(index=False))


if __name__ == "__main__":
    main()