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


def parse_percent_keepna(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip()
    out = pd.Series(np.nan, index=series.index, dtype=float)
    if s.empty:
        return out
    tmp = s.str.replace("%", "", regex=False).str.replace(" ", "", regex=False)
    tmp_num = pd.to_numeric(tmp, errors="coerce")
    if tmp_num.notna().any():
        v = tmp_num.astype(float)
        if v.abs().max(skipna=True) < 1.0:
            v = v * 100.0
        out.loc[v.index] = v
    return out


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


def normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2)))


def two_proportion_ztest_pvalue(x1: int, n1: int, x2: int, n2: int) -> float:
    if n1 == 0 or n2 == 0:
        return np.nan
    p1 = x1 / n1
    p2 = x2 / n2
    p_pool = (x1 + x2) / (n1 + n2)
    denom = math.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
    if denom == 0:
        return np.nan
    z = (p1 - p2) / denom
    return float(2 * (1 - normal_cdf(abs(z))))


def clopper_pearson_ci(x: int, n: int, alpha: float = 0.05) -> Tuple[float, float]:
    if n == 0:
        return np.nan, np.nan
    try:
        from scipy.stats import beta
        if x == 0:
            low = 0.0
        else:
            low = float(beta.ppf(alpha / 2, x, n - x + 1))
        if x == n:
            high = 1.0
        else:
            high = float(beta.ppf(1 - alpha / 2, x + 1, n - x))
        return low, high
    except Exception:
        p = x / n
        z = 1.959963984540054
        se = math.sqrt(p * (1 - p) / n)
        return max(0.0, p - z * se), min(1.0, p + z * se)


def rd_ci_pp(x1: int, n1: int, x2: int, n2: int) -> Tuple[float, float, float]:
    if n1 == 0 or n2 == 0:
        return np.nan, np.nan, np.nan
    p1 = x1 / n1
    p2 = x2 / n2
    rd = p1 - p2
    se = math.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    z = 1.959963984540054
    low = rd - z * se
    high = rd + z * se
    return float(rd * 100), float(low * 100), float(high * 100)


def rr_ci(x1: int, n1: int, x2: int, n2: int, cc: float = 0.5) -> Tuple[float, float, float]:
    if n1 == 0 or n2 == 0:
        return np.nan, np.nan, np.nan

    if x1 == 0 or x2 == 0:
        x1a = x1 + cc
        x2a = x2 + cc
        n1a = n1 + 2 * cc
        n2a = n2 + 2 * cc
    else:
        x1a, x2a, n1a, n2a = float(x1), float(x2), float(n1), float(n2)

    p1 = x1a / n1a
    p2 = x2a / n2a
    if p2 == 0:
        return np.nan, np.nan, np.nan

    rr = p1 / p2
    se_log = math.sqrt((1 / x1a) - (1 / n1a) + (1 / x2a) - (1 / n2a))
    z = 1.959963984540054
    log_rr = math.log(rr)
    low = math.exp(log_rr - z * se_log)
    high = math.exp(log_rr + z * se_log)
    return float(rr), float(low), float(high)


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


def asym_yerr_from_ci(means: Sequence[float], lows: Sequence[float], highs: Sequence[float]) -> Tuple[np.ndarray, list, list]:
    low_err, high_err = [], []
    for m, lo, hi in zip(means, lows, highs):
        if np.isfinite(m) and np.isfinite(lo):
            low_err.append(max(0.0, m - lo))
        else:
            low_err.append(0.0)
        if np.isfinite(m) and np.isfinite(hi):
            high_err.append(max(0.0, hi - m))
        else:
            high_err.append(0.0)
    return np.vstack([low_err, high_err]), low_err, high_err


def set_ylim_with_headroom(
    ax: plt.Axes,
    means: Sequence[float],
    low_errs: Sequence[float],
    high_errs: Sequence[float],
    top_pad_frac: float = 0.12,
    bottom_pad_frac: float = 0.08,
    force_zero: bool = True,
) -> None:
    m = np.array(means, dtype=float)
    loe = np.array(low_errs, dtype=float)
    hie = np.array(high_errs, dtype=float)

    y_max = np.nanmax(m + hie)
    y_min = np.nanmin(m - loe)
    if not np.isfinite(y_max):
        return

    pad_top = max(1e-6, abs(y_max) * top_pad_frac)
    pad_bottom = max(1e-6, abs(y_max) * bottom_pad_frac)

    if force_zero:
        lo = 0.0
    else:
        lo = (y_min - pad_bottom) if np.isfinite(y_min) else 0.0

    hi = y_max + pad_top
    if np.isfinite(lo) and np.isfinite(hi) and lo < hi:
        ax.set_ylim(lo, hi)


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
            "axes.titlesize": 13,
            "axes.labelsize": 11.5,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "axes.linewidth": 1.1,
        }
    )


def style_axis(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.1)
    ax.spines["bottom"].set_linewidth(1.1)
    ax.tick_params(direction="out", length=3.5, width=1.0)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.grid(axis="y", linewidth=0.6, alpha=0.22)
    ax.set_axisbelow(True)


def plot_weight_loss_bars(
    out_dir: Path,
    human_mean_wkg: float,
    human_low_wkg: float,
    human_high_wkg: float,
    eps_mean_wkg: float,
    eps_low_wkg: float,
    eps_high_wkg: float,
    human_mean_wpct: float,
    human_low_wpct: float,
    human_high_wpct: float,
    eps_mean_wpct: float,
    eps_low_wpct: float,
    eps_high_wpct: float,
    human_pct_ge2: float,
    human_low_ge2: float,
    human_high_ge2: float,
    eps_pct_ge2: float,
    eps_low_ge2: float,
    eps_high_ge2: float,
    p_wkg: float,
    p_wpct: float,
    p_ge2: float,
) -> None:
    apply_plot_style()

    color_eps = "#6CBAD8"
    color_human = "#BAD2E1"
    err_color = "black"

    bar_width = 0.72
    capsize = 4
    eline = 1.3
    capthick = 1.3

    x = np.arange(2)
    xticklabels = ["Human", "EPS–human"]

    fig, axes = plt.subplots(1, 3, figsize=(10.4, 3.15), dpi=300)
    fig.patch.set_facecolor("white")

    for ax in axes:
        style_axis(ax)

    def draw_panel(
        ax: plt.Axes,
        means: Sequence[float],
        lows: Sequence[float],
        highs: Sequence[float],
        title: str,
        ylabel: str,
        pval: float,
        force_zero: bool = True,
    ) -> None:
        yerr, low_err, high_err = asym_yerr_from_ci(means, lows, highs)
        ax.bar(
            x,
            means,
            width=bar_width,
            color=[color_human, color_eps],
            edgecolor="none",
            yerr=yerr,
            capsize=capsize,
            error_kw={"ecolor": err_color, "elinewidth": eline, "capthick": capthick},
            zorder=3,
        )
        ax.set_xticks(x, xticklabels)
        ax.set_title(title, pad=8)
        ax.set_ylabel(ylabel)
        set_ylim_with_headroom(ax, means, low_err, high_err, force_zero=force_zero)
        ax.margins(x=0.10)
        ax.text(0.5, 0.95, format_p(pval), transform=ax.transAxes, ha="center", va="top")

    draw_panel(
        axes[0],
        means=[human_mean_wkg, eps_mean_wkg],
        lows=[human_low_wkg, eps_low_wkg],
        highs=[human_high_wkg, eps_high_wkg],
        title="Weight loss",
        ylabel="Mean (95% CI), kg",
        pval=p_wkg,
        force_zero=True,
    )

    draw_panel(
        axes[1],
        means=[human_mean_wpct, eps_mean_wpct],
        lows=[human_low_wpct, eps_low_wpct],
        highs=[human_high_wpct, eps_high_wpct],
        title="Weight loss ratio",
        ylabel="Mean (95% CI), %",
        pval=p_wpct,
        force_zero=True,
    )

    draw_panel(
        axes[2],
        means=[human_pct_ge2, eps_pct_ge2],
        lows=[human_low_ge2, eps_low_ge2],
        highs=[human_high_ge2, eps_high_ge2],
        title="≥2% weight loss",
        ylabel="Proportion (95% CI), %",
        pval=p_ge2,
        force_zero=True,
    )

    fig.tight_layout(w_pad=2.2)

    out_png = out_dir / "weight_loss_bars.png"
    out_pdf = out_dir / "weight_loss_bars.pdf"
    ensure_parent_dir(out_png)
    fig.savefig(out_png, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Weight-loss outcomes (Fig. 4).")
    p.add_argument("--weight_human", type=str, default=None, help="Path to Human weight-loss Excel.")
    p.add_argument("--weight_eps", type=str, default=None, help="Path to EPS-human weight-loss Excel.")
    p.add_argument("--out_dir", type=str, default="outputs/clinical_trial", help="Output directory.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)

    if args.weight_human is None or args.weight_eps is None:
        raise SystemExit("Missing input paths. Provide --weight_human and --weight_eps.")

    df_h = read_excel(Path(args.weight_human))
    df_e = read_excel(Path(args.weight_eps))

    col_wl = pick_first_existing(df_h, ["weight_loss_kg", "weight_loss"]) or "weight_loss_kg"
    col_ratio = pick_first_existing(df_h, ["weight_loss_pct", "weight_loss_ratio"]) or "weight_loss_pct"

    baseline_weight_candidates = ["baseline_weight_kg", "baseline_weight", "weight_baseline"]
    col_w0_h = pick_first_existing(df_h, baseline_weight_candidates)
    col_w0_e = pick_first_existing(df_e, baseline_weight_candidates)
    if col_w0_h is None or col_w0_e is None:
        raise ValueError(
            "Baseline weight column not found in both arms. "
            f"Candidates={baseline_weight_candidates}. Human_found={col_w0_h}, EPS_found={col_w0_e}"
        )

    df_h = df_h.copy()
    df_e = df_e.copy()

    df_h["wl_kg"] = clean_numeric_keepna(df_h[col_wl])
    df_e["wl_kg"] = clean_numeric_keepna(df_e[col_wl])

    df_h["w0"] = clean_numeric_keepna(df_h[col_w0_h])
    df_e["w0"] = clean_numeric_keepna(df_e[col_w0_e])

    ratio_h = parse_percent_keepna(df_h[col_ratio]) if col_ratio in df_h.columns else pd.Series(np.nan, index=df_h.index)
    ratio_e = parse_percent_keepna(df_e[col_ratio]) if col_ratio in df_e.columns else pd.Series(np.nan, index=df_e.index)

    ratio_h = ratio_h.where(ratio_h.notna(), (df_h["wl_kg"] / df_h["w0"]) * 100.0)
    ratio_e = ratio_e.where(ratio_e.notna(), (df_e["wl_kg"] / df_e["w0"]) * 100.0)

    df_h["wl_pct"] = ratio_h
    df_e["wl_pct"] = ratio_e

    df_h["ge2"] = np.where(df_h["wl_pct"].notna(), (df_h["wl_pct"] >= 2.0).astype(int), np.nan)
    df_e["ge2"] = np.where(df_e["wl_pct"].notna(), (df_e["wl_pct"] >= 2.0).astype(int), np.nan)

    # Continuous outcomes (mean and t-based 95% CI)
    n_h_kg, m_h_kg, sd_h_kg, lo_h_kg, hi_h_kg = mean_ci_95(df_h["wl_kg"])
    n_e_kg, m_e_kg, sd_e_kg, lo_e_kg, hi_e_kg = mean_ci_95(df_e["wl_kg"])

    n_h_pct, m_h_pct, sd_h_pct, lo_h_pct, hi_h_pct = mean_ci_95(df_h["wl_pct"])
    n_e_pct, m_e_pct, sd_e_pct, lo_e_pct, hi_e_pct = mean_ci_95(df_e["wl_pct"])

    # Binary outcome (exact CI)
    ge2_h = pd.Series(df_h["ge2"]).dropna().astype(int)
    ge2_e = pd.Series(df_e["ge2"]).dropna().astype(int)

    x_h = int(ge2_h.sum())
    n_h = int(len(ge2_h))
    x_e = int(ge2_e.sum())
    n_e = int(len(ge2_e))

    p_h = x_h / n_h if n_h else np.nan
    p_e = x_e / n_e if n_e else np.nan
    lo_h_p, hi_h_p = clopper_pearson_ci(x_h, n_h)
    lo_e_p, hi_e_p = clopper_pearson_ci(x_e, n_e)

    # Between-arm tests/effects
    p_kg = welch_ttest_pvalue(df_e["wl_kg"], df_h["wl_kg"])
    diff_kg, diff_kg_lo, diff_kg_hi = welch_mean_diff_ci(df_e["wl_kg"], df_h["wl_kg"])

    p_pct = welch_ttest_pvalue(df_e["wl_pct"], df_h["wl_pct"])
    diff_pct, diff_pct_lo, diff_pct_hi = welch_mean_diff_ci(df_e["wl_pct"], df_h["wl_pct"])

    p_ge2 = two_proportion_ztest_pvalue(x_e, n_e, x_h, n_h)
    rd_pp, rd_lo, rd_hi = rd_ci_pp(x_e, n_e, x_h, n_h)  # EPS - Human
    rr, rr_lo, rr_hi = rr_ci(x_e, n_e, x_h, n_h)

    # Output table (add compatible column names used by common plotting scripts)
    results = pd.DataFrame(
        [
            {
                "Metric": "Weight change (kg)",
                "Human_N": n_h_kg,
                "Human_Mean": m_h_kg,
                "Human_SD": sd_h_kg,
                "Human_Mean_CI_low": lo_h_kg,
                "Human_Mean_CI_high": hi_h_kg,
                "EPS_N": n_e_kg,
                "EPS_Mean": m_e_kg,
                "EPS_SD": sd_e_kg,
                "EPS_Mean_CI_low": lo_e_kg,
                "EPS_Mean_CI_high": hi_e_kg,
                "MeanDiff": diff_kg,
                "MeanDiff_CI_low": diff_kg_lo,
                "MeanDiff_CI_high": diff_kg_hi,
                "P_value": p_kg,
            },
            {
                "Metric": "Percent weight change (%)",
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
            {
                "Metric": "At least 2% weight loss",
                "Human_N": n_h,
                "Human_%": (p_h * 100.0) if np.isfinite(p_h) else np.nan,
                "Human_%_CI_low": (lo_h_p * 100.0) if np.isfinite(lo_h_p) else np.nan,
                "Human_%_CI_high": (hi_h_p * 100.0) if np.isfinite(hi_h_p) else np.nan,
                "EPS_N": n_e,
                "EPS_%": (p_e * 100.0) if np.isfinite(p_e) else np.nan,
                "EPS_%_CI_low": (lo_e_p * 100.0) if np.isfinite(lo_e_p) else np.nan,
                "EPS_%_CI_high": (hi_e_p * 100.0) if np.isfinite(hi_e_p) else np.nan,
                "RD_pp": rd_pp,
                "RD_CI_low": rd_lo,
                "RD_CI_high": rd_hi,
                "RR": rr,
                "RR_CI_low": rr_lo,
                "RR_CI_high": rr_hi,
                "P_value": p_ge2,
            },
        ]
    )

    out_xlsx = out_dir / "weight_loss_analysis_results.xlsx"
    ensure_parent_dir(out_xlsx)
    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
        results.to_excel(writer, index=False, sheet_name="Results")

    # Plot (publication-style)
    plot_weight_loss_bars(
        out_dir=out_dir,
        human_mean_wkg=m_h_kg,
        human_low_wkg=lo_h_kg,
        human_high_wkg=hi_h_kg,
        eps_mean_wkg=m_e_kg,
        eps_low_wkg=lo_e_kg,
        eps_high_wkg=hi_e_kg,
        human_mean_wpct=m_h_pct,
        human_low_wpct=lo_h_pct,
        human_high_wpct=hi_h_pct,
        eps_mean_wpct=m_e_pct,
        eps_low_wpct=lo_e_pct,
        eps_high_wpct=hi_e_pct,
        human_pct_ge2=(p_h * 100.0) if np.isfinite(p_h) else np.nan,
        human_low_ge2=(lo_h_p * 100.0) if np.isfinite(lo_h_p) else np.nan,
        human_high_ge2=(hi_h_p * 100.0) if np.isfinite(hi_h_p) else np.nan,
        eps_pct_ge2=(p_e * 100.0) if np.isfinite(p_e) else np.nan,
        eps_low_ge2=(lo_e_p * 100.0) if np.isfinite(lo_e_p) else np.nan,
        eps_high_ge2=(hi_e_p * 100.0) if np.isfinite(hi_e_p) else np.nan,
        p_wkg=p_kg,
        p_wpct=p_pct,
        p_ge2=p_ge2,
    )


if __name__ == "__main__":
    main()
