from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


def read_excel(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path)
    df.columns = df.columns.astype(str).str.strip()
    return df


def pick_first_existing(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    cols = {str(c).strip(): str(c).strip() for c in df.columns}
    for c in candidates:
        if c in cols:
            return cols[c]
    return None


def pick_case_insensitive(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    mapping = {str(c).strip().lower(): str(c).strip() for c in df.columns}
    for c in candidates:
        key = str(c).strip().lower()
        if key in mapping:
            return mapping[key]
    return None


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def clean_numeric_keepna(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def t_crit_975(df: float) -> float:
    try:
        from scipy import stats
        return float(stats.t.ppf(0.975, df))
    except Exception:
        return 1.959963984540054


def mean_ci_95(series: pd.Series) -> Tuple[int, float, float, float, float]:
    s = pd.Series(series).dropna().astype(float)
    n = int(len(s))
    if n == 0:
        return 0, np.nan, np.nan, np.nan, np.nan
    m = float(s.mean())
    sd = float(s.std(ddof=1)) if n > 1 else 0.0
    if n <= 1:
        return n, m, sd, np.nan, np.nan
    se = sd / math.sqrt(n)
    tcrit = t_crit_975(n - 1)
    low = m - tcrit * se
    high = m + tcrit * se
    return n, m, sd, float(low), float(high)


def welch_ttest_pvalue(x: pd.Series, y: pd.Series) -> float:
    x = pd.Series(x).dropna().astype(float).values
    y = pd.Series(y).dropna().astype(float).values
    if len(x) < 2 or len(y) < 2:
        return np.nan
    try:
        from scipy import stats
        _, p = stats.ttest_ind(x, y, equal_var=False, nan_policy="omit")
        return float(p)
    except Exception:
        return np.nan


def welch_mean_diff_ci(x_eps: pd.Series, x_hum: pd.Series) -> Tuple[float, float, float]:
    x = pd.Series(x_eps).dropna().astype(float).values
    y = pd.Series(x_hum).dropna().astype(float).values
    n1, n2 = len(x), len(y)
    if n1 < 2 or n2 < 2:
        return np.nan, np.nan, np.nan

    m1, m2 = float(np.mean(x)), float(np.mean(y))
    s1, s2 = float(np.std(x, ddof=1)), float(np.std(y, ddof=1))
    diff = m1 - m2

    se2 = (s1 * s1) / n1 + (s2 * s2) / n2
    if se2 <= 0:
        return float(diff), np.nan, np.nan

    num = se2 * se2
    den = ((s1 * s1 / n1) ** 2) / (n1 - 1) + ((s2 * s2 / n2) ** 2) / (n2 - 1)
    df = num / den if den > 0 else (n1 + n2 - 2)

    tcrit = t_crit_975(df)
    se = math.sqrt(se2)
    low = diff - tcrit * se
    high = diff + tcrit * se
    return float(diff), float(low), float(high)


def format_p(p: object) -> str:
    if p is None:
        return r"$\it{P}$ = NA"
    if isinstance(p, str) and p.strip().startswith("<"):
        thr = p.strip()[1:]
        return rf"$\it{{P}}$ < {thr}"
    try:
        pv = float(p)
    except Exception:
        return r"$\it{P}$ = NA"
    if np.isnan(pv):
        return r"$\it{P}$ = NA"
    if pv < 0.0001:
        return r"$\it{P}$ < 0.0001"
    return rf"$\it{{P}}$ = {pv:.4f}"


def apply_plot_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "Times New Roman",
            "mathtext.fontset": "custom",
            "mathtext.rm": "Times New Roman",
            "mathtext.it": "Times New Roman:italic",
            "mathtext.bf": "Times New Roman:bold",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def style_axis(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.1)
    ax.spines["bottom"].set_linewidth(1.1)
    ax.tick_params(axis="both", direction="out", length=4, width=1.0, labelsize=12)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, linewidth=0.6, alpha=0.25)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))


def set_ylim_with_headroom(ax: plt.Axes, lows: Sequence[float], highs: Sequence[float], top_pad_frac: float = 0.18, bottom_pad_frac: float = 0.12) -> None:
    lo = np.nanmin(np.array(lows, dtype=float))
    hi = np.nanmax(np.array(highs, dtype=float))
    if not (np.isfinite(lo) and np.isfinite(hi)) or lo == hi:
        return
    rng = hi - lo
    ax.set_ylim(lo - rng * bottom_pad_frac, hi + rng * top_pad_frac)


def plot_glycemic_bars(
    out_dir: Path,
    human_mean_red: float,
    human_low_red: float,
    human_high_red: float,
    eps_mean_red: float,
    eps_low_red: float,
    eps_high_red: float,
    human_mean_pct: float,
    human_low_pct: float,
    human_high_pct: float,
    eps_mean_pct: float,
    eps_low_pct: float,
    eps_high_pct: float,
    p_red: float,
    p_pct: float,
) -> None:
    apply_plot_style()

    col_eps = "#6CBAD8"
    col_hum = "#BAD2E1"

    groups = ["Human", "EPS-human"]
    xticklabels = ["Human", "EPS–human"]
    x = np.arange(len(groups))
    bar_width = 0.62

    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.2), dpi=300)

    for ax in axes:
        style_axis(ax)

    # Panel 1
    m1 = [human_mean_red, eps_mean_red]
    lo1 = [human_low_red, eps_low_red]
    hi1 = [human_high_red, eps_high_red]
    yerr1 = np.vstack([np.array(m1) - np.array(lo1), np.array(hi1) - np.array(m1)])

    axes[0].bar(
        x,
        m1,
        width=bar_width,
        color=[col_hum, col_eps],
        edgecolor="none",
        linewidth=0,
        yerr=yerr1,
        capsize=4,
        error_kw=dict(ecolor="black", elinewidth=1.4, capthick=1.4),
        zorder=3,
    )
    axes[0].set_xticks(x, xticklabels)
    axes[0].set_title("Fasting glucose reduction", fontsize=16, pad=10)
    axes[0].set_ylabel("Mean (95% CI), mmol/L", fontsize=14)
    axes[0].set_xlim(-0.55, 1.55)
    set_ylim_with_headroom(axes[0], lo1, hi1)
    axes[0].text(0.5, 0.93, format_p(p_red), transform=axes[0].transAxes, ha="center", va="top", fontsize=12)

    # Panel 2
    m2 = [human_mean_pct, eps_mean_pct]
    lo2 = [human_low_pct, eps_low_pct]
    hi2 = [human_high_pct, eps_high_pct]
    yerr2 = np.vstack([np.array(m2) - np.array(lo2), np.array(hi2) - np.array(m2)])

    axes[1].bar(
        x,
        m2,
        width=bar_width,
        color=[col_hum, col_eps],
        edgecolor="none",
        linewidth=0,
        yerr=yerr2,
        capsize=4,
        error_kw=dict(ecolor="black", elinewidth=1.4, capthick=1.4),
        zorder=3,
    )
    axes[1].set_xticks(x, xticklabels)
    axes[1].set_title("Fasting glucose reduction ratio", fontsize=16, pad=10)
    axes[1].set_ylabel("Mean (95% CI), %", fontsize=14)
    axes[1].set_xlim(-0.55, 1.55)
    set_ylim_with_headroom(axes[1], lo2, hi2)
    axes[1].text(0.5, 0.93, format_p(p_pct), transform=axes[1].transAxes, ha="center", va="top", fontsize=12)

    fig.tight_layout()

    out_png = out_dir / "glycemic_fpg_reduction_bars.png"
    out_pdf = out_dir / "glycemic_fpg_reduction_bars.pdf"
    ensure_parent_dir(out_png)
    fig.savefig(out_png, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fasting glucose outcomes (Fig. 7).")
    p.add_argument("--gly_human", type=str, default=None, help="Path to Human glycemic-control Excel.")
    p.add_argument("--gly_eps", type=str, default=None, help="Path to EPS-human glycemic-control Excel.")
    p.add_argument("--out_dir", type=str, default="outputs/clinical_trial", help="Output directory.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)

    if args.gly_human is None or args.gly_eps is None:
        raise SystemExit("Missing input paths. Provide --gly_human and --gly_eps.")

    df_h = read_excel(Path(args.gly_human))
    df_e = read_excel(Path(args.gly_eps))

    baseline_fpg_candidates = ["baseline_fpg_mmol", "baseline_fpg", "FPG0", "Baseline fasting glucose", "Fasting glucose"]
    end_fpg_candidates = ["endpoint_fpg_mmol", "endpoint_fpg", "FPG1", "Endpoint fasting glucose"]

    col_fpg0_h = pick_first_existing(df_h, baseline_fpg_candidates)
    col_fpg1_h = pick_first_existing(df_h, end_fpg_candidates)
    col_fpg0_e = pick_first_existing(df_e, baseline_fpg_candidates)
    col_fpg1_e = pick_first_existing(df_e, end_fpg_candidates)

    if col_fpg0_h is None or col_fpg1_h is None or col_fpg0_e is None or col_fpg1_e is None:
        raise ValueError(
            "Fasting glucose columns not found in both arms. "
            f"Baseline_candidates={baseline_fpg_candidates}, End_candidates={end_fpg_candidates}. "
            f"Human_found=({col_fpg0_h},{col_fpg1_h}), EPS_found=({col_fpg0_e},{col_fpg1_e})"
        )

    df_h = df_h.copy()
    df_e = df_e.copy()

    df_h["fpg0"] = clean_numeric_keepna(df_h[col_fpg0_h])
    df_h["fpg1"] = clean_numeric_keepna(df_h[col_fpg1_h])
    df_e["fpg0"] = clean_numeric_keepna(df_e[col_fpg0_e])
    df_e["fpg1"] = clean_numeric_keepna(df_e[col_fpg1_e])

    df_h["fpg_red"] = df_h["fpg0"] - df_h["fpg1"]
    df_e["fpg_red"] = df_e["fpg0"] - df_e["fpg1"]

    df_h["fpg_red_pct"] = np.where(df_h["fpg0"] > 0, df_h["fpg_red"] / df_h["fpg0"] * 100.0, np.nan)
    df_e["fpg_red_pct"] = np.where(df_e["fpg0"] > 0, df_e["fpg_red"] / df_e["fpg0"] * 100.0, np.nan)

    # Arm summaries
    n_h_red, m_h_red, sd_h_red, lo_h_red, hi_h_red = mean_ci_95(df_h["fpg_red"])
    n_e_red, m_e_red, sd_e_red, lo_e_red, hi_e_red = mean_ci_95(df_e["fpg_red"])

    n_h_pct, m_h_pct, sd_h_pct, lo_h_pct, hi_h_pct = mean_ci_95(df_h["fpg_red_pct"])
    n_e_pct, m_e_pct, sd_e_pct, lo_e_pct, hi_e_pct = mean_ci_95(df_e["fpg_red_pct"])

    # Between-arm Welch
    p_red = welch_ttest_pvalue(df_e["fpg_red"], df_h["fpg_red"])
    diff_red, diff_red_lo, diff_red_hi = welch_mean_diff_ci(df_e["fpg_red"], df_h["fpg_red"])

    p_pct = welch_ttest_pvalue(df_e["fpg_red_pct"], df_h["fpg_red_pct"])
    diff_pct, diff_pct_lo, diff_pct_hi = welch_mean_diff_ci(df_e["fpg_red_pct"], df_h["fpg_red_pct"])

    results = pd.DataFrame(
        [
            {
                "Metric": "Fasting glucose reduction",
                "Human_N": n_h_red,
                "Human_Mean": m_h_red,
                "Human_SD": sd_h_red,
                "Human_Mean_CI_low": lo_h_red,
                "Human_Mean_CI_high": hi_h_red,
                "EPS_N": n_e_red,
                "EPS_Mean": m_e_red,
                "EPS_SD": sd_e_red,
                "EPS_Mean_CI_low": lo_e_red,
                "EPS_Mean_CI_high": hi_e_red,
                "MeanDiff": diff_red,
                "MeanDiff_CI_low": diff_red_lo,
                "MeanDiff_CI_high": diff_red_hi,
                "P_value": p_red,
            },
            {
                "Metric": "Fasting glucose reduction ratio (%)",
                "Human_N": n_h_pct,
                "Human_Mean": m_h_pct,
                "Human_SD": sd_h_pct,
                "Human_Mean_CI_low": lo_h_pct,
                "Human_Mean_CI_high": hi_h_pct,
                "EPS_N": n_e_pct,
                "EPS_Mean": m_e_pct,
                "EPS_SD": sd_e_pct,
                "EPS_Mean_CI_low": lo_e_pct,
                "EPS_Mean_CI_high": hi_e_pct,
                "MeanDiff": diff_pct,
                "MeanDiff_CI_low": diff_pct_lo,
                "MeanDiff_CI_high": diff_pct_hi,
                "P_value": p_pct,
            },
        ]
    )

    out_xlsx = out_dir / "glycemic_control_analysis_results.xlsx"
    ensure_parent_dir(out_xlsx)
    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
        results.to_excel(writer, index=False, sheet_name="Results")

    # Plot (publication-style)
    plot_glycemic_bars(
        out_dir=out_dir,
        human_mean_red=m_h_red,
        human_low_red=lo_h_red,
        human_high_red=hi_h_red,
        eps_mean_red=m_e_red,
        eps_low_red=lo_e_red,
        eps_high_red=hi_e_red,
        human_mean_pct=m_h_pct,
        human_low_pct=lo_h_pct,
        human_high_pct=hi_h_pct,
        eps_mean_pct=m_e_pct,
        eps_low_pct=lo_e_pct,
        eps_high_pct=hi_e_pct,
        p_red=p_red,
        p_pct=p_pct,
    )


if __name__ == "__main__":
    main()
