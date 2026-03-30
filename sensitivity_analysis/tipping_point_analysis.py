"""
Tipping-Point Sensitivity Analysis
=================================
Core question: How much worse would missing EPS-arm outcomes need to be,
relative to MAR imputation, before the EPS-vs-Human treatment effect is
no longer statistically persuasive?

Primary scenario:
  Delta applied to missing EPS-arm outcomes only.

Secondary scenario:
  Differential delta for weight loss, where missing EPS-arm outcomes are
  made worse while missing Human-arm outcomes are made better.

Reference workflow:
  If missing-baseline Excel files are supplied, those records are used as
  the real baseline data for non-completers. If they are not supplied, the
  repository falls back to within-arm resampling from completers so the
  bundled example data still run end-to-end.

This script is the single source of MNAR delta-adjustment and tipping-point
results. The ITT scripts handle MAR MI, available-case analyses, BOCF, and
diagnostics only.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="statsmodels")
warnings.filterwarnings("ignore", message=".*convergence.*", category=RuntimeWarning)

import statsmodels.api as sm


def clean_numeric(series):
    return pd.to_numeric(series, errors="coerce")


def clean_sex(series):
    raw = series.astype(str).str.strip().str.lower()
    out = pd.Series(np.nan, index=raw.index, dtype=float)
    out[raw.isin({"女", "female", "f", "0", "2"})] = 1.0
    out[raw.isin({"男", "male", "m", "1"})] = 0.0
    return out


def pick_col(df, candidates):
    mapping = {str(col).strip().lower(): str(col).strip() for col in df.columns}
    for candidate in candidates:
        key = candidate.lower()
        if key in mapping:
            return mapping[key]
    return None


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def fmt_p(value):
    if pd.isna(value):
        return ""
    return "<0.0001" if value < 1e-4 else f"{value:.4f}"


def describe_source(source):
    if source == "resampled_from_completers":
        return "Within-arm resampling from completers (repository fallback)"
    if source.startswith("file:"):
        return f"Missing baseline file: {source[5:]}"
    return source


def rubins_rules(estimates, variances):
    from scipy.stats import t as tdist

    est = np.array(estimates, dtype=float)
    var = np.array(variances, dtype=float)
    valid = ~(np.isnan(est) | np.isnan(var))
    est, var = est[valid], var[valid]

    m = len(est)
    if m < 2:
        return np.nan, np.nan, np.nan, np.nan, np.nan

    q_bar = np.mean(est)
    u_bar = np.mean(var)
    b = np.var(est, ddof=1)
    total_var = u_bar + (1 + 1 / m) * b
    se = math.sqrt(total_var) if total_var > 0 else np.nan

    if b > 0 and u_bar > 0:
        r = (1 + 1 / m) * b / u_bar
        df = (m - 1) * (1 + 1 / r) ** 2
    else:
        df = 1e6

    tcrit = float(tdist.ppf(0.975, df))
    lo = q_bar - tcrit * se if not np.isnan(se) else np.nan
    hi = q_bar + tcrit * se if not np.isnan(se) else np.nan
    p_value = 2 * float(tdist.sf(abs(q_bar / se), df)) if not np.isnan(se) and se > 0 else np.nan
    return q_bar, lo, hi, p_value, se


def ancova_effect(df_c, outcome_col, covars):
    covars = [col for col in covars if col in df_c.columns]
    cols = ["group_num"] + covars + [outcome_col]
    data = df_c[cols].dropna()
    if len(data) < len(cols) + 3:
        return np.nan, np.nan, np.nan

    y = data[outcome_col].astype(float)
    x = sm.add_constant(data[["group_num"] + covars].astype(float))
    try:
        model = sm.OLS(y, x).fit()
        return float(model.params["group_num"]), float(model.bse["group_num"]), float(model.pvalues["group_num"])
    except Exception:
        return np.nan, np.nan, np.nan


def load_weight_arm(path, label):
    df = pd.read_excel(path)
    df.columns = df.columns.astype(str).str.strip()

    col_age = pick_col(df, ["age", "年龄"])
    col_sex = pick_col(df, ["sex", "性别"])
    col_bmi = pick_col(df, ["bmi", "BMI"])
    col_height = pick_col(df, ["height", "身高"])
    col_baseline_wt = pick_col(
        df,
        ["baseline_weight_kg", "入营体重", "入营体重kg", "入营体重（档案）", "初始体重（档案）"],
    )
    col_hba1c = pick_col(df, ["hba1c", "HbA1c", "糖化血红蛋白"])
    col_weight_loss = pick_col(df, ["weight_loss_kg", "减重数"])

    arm = pd.DataFrame(
        {
            "age": clean_numeric(df[col_age]).values if col_age else np.nan,
            "sex": clean_sex(df[col_sex]).values if col_sex else np.nan,
            "bmi": clean_numeric(df[col_bmi]).values if col_bmi else np.nan,
            "height": clean_numeric(df[col_height]).values if col_height else np.nan,
            "baseline_wt": clean_numeric(df[col_baseline_wt]).values if col_baseline_wt else np.nan,
            "hba1c": clean_numeric(df[col_hba1c]).values if col_hba1c else np.nan,
            "wl_kg": clean_numeric(df[col_weight_loss]).values if col_weight_loss else np.nan,
        }
    )
    arm["group"] = label
    arm["completer"] = 1
    arm["wl_pct"] = np.where(arm["baseline_wt"] > 0, arm["wl_kg"] / arm["baseline_wt"] * 100, np.nan)
    return arm


def load_weight_missing(path, label):
    df = pd.read_excel(path)
    df.columns = df.columns.astype(str).str.strip()

    col_age = pick_col(df, ["age", "年龄"])
    col_sex = pick_col(df, ["sex", "性别"])
    col_bmi = pick_col(df, ["bmi", "BMI"])
    col_height = pick_col(df, ["height", "身高"])
    col_baseline_wt = pick_col(
        df,
        ["baseline_weight_kg", "入营体重", "入营体重kg", "入营体重（档案）", "初始体重（档案）"],
    )
    col_hba1c = pick_col(df, ["hba1c", "HbA1c", "糖化血红蛋白"])

    hba1c = clean_numeric(df[col_hba1c]) if col_hba1c else pd.Series(np.nan, index=df.index, dtype=float)
    hba1c = hba1c.copy()
    hba1c[hba1c == 0] = np.nan

    missing = pd.DataFrame(
        {
            "age": clean_numeric(df[col_age]).values if col_age else np.nan,
            "sex": clean_sex(df[col_sex]).values if col_sex else np.nan,
            "bmi": clean_numeric(df[col_bmi]).values if col_bmi else np.nan,
            "height": clean_numeric(df[col_height]).values if col_height else np.nan,
            "baseline_wt": clean_numeric(df[col_baseline_wt]).values if col_baseline_wt else np.nan,
            "hba1c": hba1c.values,
            "wl_kg": np.nan,
            "wl_pct": np.nan,
        }
    )
    missing["group"] = label
    missing["completer"] = 0
    return missing


def load_glycemic_arm(path, label):
    df = pd.read_excel(path)
    df.columns = df.columns.astype(str).str.strip()

    col_age = pick_col(df, ["age", "年龄"])
    col_sex = pick_col(df, ["sex", "性别"])
    col_bmi = pick_col(df, ["bmi", "BMI"])
    col_height = pick_col(df, ["height", "身高"])
    col_bw = pick_col(df, ["baseline_weight_kg", "入营体重", "入营体重kg", "初始体重（档案）"])
    col_fpg0 = pick_col(df, ["baseline_fpg_mmol", "入营空腹"])
    col_ppg0 = pick_col(df, ["baseline_ppg_mmol", "入营餐后2小时"])
    col_fpg1 = pick_col(df, ["endpoint_fpg_mmol", "结营空腹"])

    arm = pd.DataFrame(
        {
            "age": clean_numeric(df[col_age]).values if col_age else np.nan,
            "sex": clean_sex(df[col_sex]).values if col_sex else np.nan,
            "bmi": clean_numeric(df[col_bmi]).values if col_bmi else np.nan,
            "height": clean_numeric(df[col_height]).values if col_height else np.nan,
            "bw": clean_numeric(df[col_bw]).values if col_bw else np.nan,
            "fpg0": clean_numeric(df[col_fpg0]).values if col_fpg0 else np.nan,
            "ppg0": clean_numeric(df[col_ppg0]).values if col_ppg0 else np.nan,
            "fpg1": clean_numeric(df[col_fpg1]).values if col_fpg1 else np.nan,
        }
    )
    arm["group"] = label
    arm["completer"] = 1
    arm["fpg_change"] = arm["fpg0"] - arm["fpg1"]
    arm["fpg_change_pct"] = np.where(arm["fpg0"] > 0, arm["fpg_change"] / arm["fpg0"] * 100, np.nan)
    return arm


def build_glycemic_missing_frame(df, label):
    col_age = pick_col(df, ["age", "年龄"])
    col_sex = pick_col(df, ["sex", "性别"])
    col_bmi = pick_col(df, ["bmi", "BMI"])
    col_height = pick_col(df, ["height", "身高"])
    col_bw = pick_col(df, ["baseline_weight_kg", "入营体重", "入营体重kg", "初始体重（档案）"])
    col_fpg0 = pick_col(df, ["baseline_fpg_mmol", "入营空腹"])
    col_ppg0 = pick_col(df, ["baseline_ppg_mmol", "入营餐后2小时"])

    missing = pd.DataFrame(
        {
            "age": clean_numeric(df[col_age]).values if col_age else np.nan,
            "sex": clean_sex(df[col_sex]).values if col_sex else np.nan,
            "bmi": clean_numeric(df[col_bmi]).values if col_bmi else np.nan,
            "height": clean_numeric(df[col_height]).values if col_height else np.nan,
            "bw": clean_numeric(df[col_bw]).values if col_bw else np.nan,
            "fpg0": clean_numeric(df[col_fpg0]).values if col_fpg0 else np.nan,
            "ppg0": clean_numeric(df[col_ppg0]).values if col_ppg0 else np.nan,
            "fpg1": np.nan,
            "fpg_change": np.nan,
            "fpg_change_pct": np.nan,
        }
    )
    missing["group"] = label
    missing["completer"] = 0
    return missing


def eps_missing_needs_shift_correction(df):
    col_age = pick_col(df, ["年龄"])
    if not col_age:
        return False
    observed = df[col_age]
    non_null = observed.notna()
    if non_null.sum() == 0:
        return False
    numeric = clean_numeric(observed[non_null]).notna().sum()
    return numeric < non_null.sum()


def load_shifted_eps_missing(path):
    df_raw = pd.read_excel(path, header=None)
    headers = df_raw.iloc[0].tolist()

    col_meas = headers.index("最新测量数据时间")
    col_age = headers.index("年龄")
    col_sex = headers.index("性别")
    col_ht = headers.index("身高")
    col_wt = headers.index("初始体重（档案）")
    col_bmi = headers.index("bmi")
    col_fpg0 = headers.index("入营空腹")

    records = []
    for _, row in df_raw.iloc[1:].iterrows():
        age_val = row.iloc[col_age]
        try:
            age = float(age_val)
            sex_raw = str(row.iloc[col_sex])
            height = float(row.iloc[col_ht]) if pd.notna(row.iloc[col_ht]) else np.nan
            weight = float(row.iloc[col_wt]) if pd.notna(row.iloc[col_wt]) else np.nan
            bmi = float(row.iloc[col_bmi]) if pd.notna(row.iloc[col_bmi]) else np.nan
        except (TypeError, ValueError):
            age = float(row.iloc[col_meas]) if pd.notna(row.iloc[col_meas]) else np.nan
            sex_raw = str(row.iloc[col_age])
            height = float(row.iloc[col_sex]) if pd.notna(row.iloc[col_sex]) else np.nan
            weight = float(row.iloc[col_ht]) if pd.notna(row.iloc[col_ht]) else np.nan
            bmi = float(row.iloc[col_wt]) if pd.notna(row.iloc[col_wt]) else np.nan

        fpg0 = float(row.iloc[col_fpg0]) if pd.notna(row.iloc[col_fpg0]) else np.nan
        records.append({"age": age, "sex_raw": sex_raw, "height": height, "bw": weight, "bmi": bmi, "fpg0": fpg0})

    missing = pd.DataFrame(records)
    missing["sex"] = clean_sex(missing["sex_raw"].astype(str))
    missing["ppg0"] = np.nan
    missing["fpg1"] = np.nan
    missing["ppg1"] = np.nan
    missing["fpg_change"] = np.nan
    missing["fpg_change_pct"] = np.nan
    missing["group"] = "EPS"
    missing["completer"] = 0
    return missing[
        ["age", "sex", "bmi", "height", "bw", "fpg0", "ppg0", "fpg1", "ppg1", "group", "completer", "fpg_change", "fpg_change_pct"]
    ]


def load_glycemic_missing(path, label):
    df = pd.read_excel(path)
    df.columns = df.columns.astype(str).str.strip()

    if label == "EPS" and eps_missing_needs_shift_correction(df):
        return load_shifted_eps_missing(path), "Applied EPS one-column shift correction to demographic fields"

    return build_glycemic_missing_frame(df, label), ""


def generate_missing(comp_df, n_missing, rng, baseline_cols):
    if n_missing <= 0:
        return pd.DataFrame(columns=comp_df.columns)

    available_cols = [col for col in baseline_cols if col in comp_df.columns]
    sampled = comp_df[available_cols].sample(n=n_missing, replace=True, random_state=rng).reset_index(drop=True)
    for col in [col for col in available_cols if col != "group"]:
        valid = sampled[col].notna()
        if not valid.any():
            continue
        sd = comp_df[col].std() * 0.1
        if pd.notna(sd) and sd > 0:
            sampled[col] = sampled[col].astype(float)
            sampled.loc[valid, col] += rng.normal(0, sd, size=valid.sum())
    return sampled


def generate_missing_wl(comp_df, n_missing, rng):
    baseline_cols = ["group", "age", "sex", "bmi", "height", "baseline_wt", "hba1c"]
    sampled = generate_missing(comp_df, n_missing, rng, baseline_cols)
    if len(sampled) > 0:
        sampled["wl_kg"] = np.nan
        sampled["wl_pct"] = np.nan
        sampled["completer"] = 0
    return sampled


def generate_missing_gl(comp_df, n_missing, rng):
    baseline_cols = ["group", "age", "sex", "bmi", "height", "bw", "fpg0", "ppg0"]
    sampled = generate_missing(comp_df, n_missing, rng, baseline_cols)
    if len(sampled) > 0:
        sampled["fpg1"] = np.nan
        sampled["ppg1"] = np.nan
        sampled["fpg_change"] = np.nan
        sampled["fpg_change_pct"] = np.nan
        sampled["completer"] = 0
    return sampled


def resolve_weight_missing(comp_df, label, n_expected, rng, missing_path=None):
    if missing_path:
        return load_weight_missing(missing_path, label), f"file:{missing_path}", ""
    return generate_missing_wl(comp_df, n_expected, rng), "resampled_from_completers", ""


def resolve_glycemic_missing(comp_df, label, n_expected, rng, missing_path=None):
    if missing_path:
        missing, note = load_glycemic_missing(missing_path, label)
        return missing, f"file:{missing_path}", note
    return generate_missing_gl(comp_df, n_expected, rng), "resampled_from_completers", ""


def mice_wl(df_full, m=20, seed=42):
    from sklearn.experimental import enable_iterative_imputer  # noqa: F401
    from sklearn.impute import IterativeImputer
    from sklearn.linear_model import BayesianRidge

    candidate_cols = ["group_num", "age", "sex", "bmi", "height", "baseline_wt", "hba1c", "wl_kg"]
    cols = [col for col in candidate_cols if col in df_full.columns and df_full[col].notna().any()]
    base = df_full[cols].copy()

    datasets = []
    for index in range(m):
        imputer = IterativeImputer(
            estimator=BayesianRidge(),
            max_iter=50,
            random_state=seed + index,
            sample_posterior=True,
            skip_complete=True,
        )
        filled = pd.DataFrame(imputer.fit_transform(base.values), columns=cols)
        dataset = df_full.copy()
        dataset["wl_kg"] = filled["wl_kg"]
        baseline_wt = filled["baseline_wt"] if "baseline_wt" in filled.columns else dataset["baseline_wt"]
        dataset["wl_pct"] = np.where(baseline_wt > 0, dataset["wl_kg"] / baseline_wt * 100, np.nan)
        datasets.append(dataset)
    return datasets


def mice_gl(df_full, m=20, seed=42):
    from sklearn.experimental import enable_iterative_imputer  # noqa: F401
    from sklearn.impute import IterativeImputer
    from sklearn.linear_model import BayesianRidge

    candidate_cols = ["group_num", "age", "sex", "bmi", "fpg0", "ppg0", "fpg1"]
    cols = [col for col in candidate_cols if col in df_full.columns and df_full[col].notna().any()]
    base = df_full[cols].copy()

    datasets = []
    for index in range(m):
        imputer = IterativeImputer(
            estimator=BayesianRidge(),
            max_iter=50,
            random_state=seed + index,
            sample_posterior=True,
            skip_complete=True,
        )
        filled = pd.DataFrame(imputer.fit_transform(base.values), columns=cols)
        dataset = df_full.copy()
        dataset["fpg1"] = filled["fpg1"]
        dataset["fpg_change"] = dataset["fpg0"] - dataset["fpg1"]
        dataset["fpg_change_pct"] = np.where(
            dataset["fpg0"] > 0, dataset["fpg_change"] / dataset["fpg0"] * 100, np.nan
        )
        datasets.append(dataset)
    return datasets


def tipping_search(df_itt, mice_func, outcome_col, delta_range, covars, apply_to="eps_only", delta_target="wl_kg", m=20, seed=42):
    base_datasets = mice_func(df_itt, m=m, seed=seed)
    results = []

    for delta in delta_range:
        coefs, variances = [], []
        for dataset in base_datasets:
            shifted = dataset.copy()
            missing_mask = df_itt["completer"] == 0

            if apply_to == "eps_only":
                target_mask = missing_mask & (df_itt["group"] == "EPS")
            elif apply_to == "human_only":
                target_mask = missing_mask & (df_itt["group"] == "Human")
            else:
                target_mask = missing_mask

            if delta_target == "wl_kg":
                shifted.loc[target_mask, "wl_kg"] += delta
                baseline_wt = shifted["baseline_wt"]
                shifted["wl_pct"] = np.where(baseline_wt > 0, shifted["wl_kg"] / baseline_wt * 100, np.nan)
            elif delta_target == "fpg1":
                shifted.loc[target_mask, "fpg1"] += delta
                shifted["fpg_change"] = shifted["fpg0"] - shifted["fpg1"]
                shifted["fpg_change_pct"] = np.where(
                    shifted["fpg0"] > 0, shifted["fpg_change"] / shifted["fpg0"] * 100, np.nan
                )

            coef, se, _ = ancova_effect(shifted, outcome_col, covars)
            coefs.append(coef)
            variances.append(se**2 if not np.isnan(se) else np.nan)

        q_bar, lo, hi, p_value, _ = rubins_rules(coefs, variances)
        results.append(
            {
                "delta": delta,
                "estimate": q_bar,
                "ci_lo": lo,
                "ci_hi": hi,
                "p": p_value,
                "significant": p_value < 0.05 if not np.isnan(p_value) else None,
                "ci_excludes_zero": (lo > 0 or hi < 0) if not (np.isnan(lo) or np.isnan(hi)) else None,
            }
        )

    return results


def find_tipping(results):
    for index, row in enumerate(results):
        if row["significant"] is False or row["ci_excludes_zero"] is False:
            return row["delta"], row["p"], row["ci_lo"], row["ci_hi"], index == 0
    return None, None, None, None, False


def detail_df(results):
    rows = []
    for row in results:
        rows.append(
            {
                "Delta": row["delta"],
                "ANCOVA Estimate (EPS-Human)": f"{row['estimate']:.4f}" if not np.isnan(row["estimate"]) else "",
                "CI low": f"{row['ci_lo']:.4f}" if not np.isnan(row["ci_lo"]) else "",
                "CI high": f"{row['ci_hi']:.4f}" if not np.isnan(row["ci_hi"]) else "",
                "P value": fmt_p(row["p"]),
                "Significant (p<0.05)": row["significant"],
                "CI excludes 0": row["ci_excludes_zero"],
            }
        )
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Tipping-point sensitivity analysis for the weight-loss and glycemic cohorts")
    parser.add_argument("--weight_human", type=str, default="data/example/weight_loss/human_arm.xlsx")
    parser.add_argument("--weight_eps", type=str, default="data/example/weight_loss/eps_arm.xlsx")
    parser.add_argument("--weight_human_missing", type=str, default=None)
    parser.add_argument("--weight_eps_missing", type=str, default=None)
    parser.add_argument("--gly_human", type=str, default="data/example/glycemic/human_arm.xlsx")
    parser.add_argument("--gly_eps", type=str, default="data/example/glycemic/eps_arm.xlsx")
    parser.add_argument("--gly_human_missing", type=str, default=None)
    parser.add_argument("--gly_eps_missing", type=str, default=None)
    parser.add_argument("--n_rand_human_wl", type=int, default=100)
    parser.add_argument("--n_rand_eps_wl", type=int, default=100)
    parser.add_argument("--n_rand_human_gl", type=int, default=50)
    parser.add_argument("--n_rand_eps_gl", type=int, default=50)
    parser.add_argument("--m_imputations", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_dir", type=str, default="outputs/sensitivity_analysis")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)
    out_xlsx = out_dir / "tipping_point_results.xlsx"

    print("=" * 60)
    print("TIPPING-POINT SENSITIVITY ANALYSIS")
    print("=" * 60)

    comp_h_wl = load_weight_arm(args.weight_human, "Human")
    comp_e_wl = load_weight_arm(args.weight_eps, "EPS")
    rng_wl = np.random.default_rng(args.seed)

    n_miss_h_wl_expected = max(0, args.n_rand_human_wl - len(comp_h_wl))
    n_miss_e_wl_expected = max(0, args.n_rand_eps_wl - len(comp_e_wl))
    miss_h_wl, source_h_wl, note_h_wl = resolve_weight_missing(
        comp_h_wl,
        "Human",
        n_miss_h_wl_expected,
        rng_wl,
        args.weight_human_missing,
    )
    miss_e_wl, source_e_wl, note_e_wl = resolve_weight_missing(
        comp_e_wl,
        "EPS",
        n_miss_e_wl_expected,
        rng_wl,
        args.weight_eps_missing,
    )
    if args.weight_human_missing and len(miss_h_wl) != n_miss_h_wl_expected:
        print(
            f"Warning: Human weight missing file has {len(miss_h_wl)} rows, "
            f"but randomized count implies {n_miss_h_wl_expected} missing participants."
        )
    if args.weight_eps_missing and len(miss_e_wl) != n_miss_e_wl_expected:
        print(
            f"Warning: EPS weight missing file has {len(miss_e_wl)} rows, "
            f"but randomized count implies {n_miss_e_wl_expected} missing participants."
        )

    for arm, n_comp, n_miss, n_rand in [
        ("Human", len(comp_h_wl), len(miss_h_wl), args.n_rand_human_wl),
        ("EPS", len(comp_e_wl), len(miss_e_wl), args.n_rand_eps_wl),
    ]:
        total = n_comp + n_miss
        mark = "OK" if total == n_rand else "WARNING"
        print(f"WL {mark}: {arm} {n_comp}+{n_miss}={total} (target={n_rand})")

    df_itt_wl = pd.concat([comp_h_wl, comp_e_wl, miss_h_wl, miss_e_wl], ignore_index=True)
    df_itt_wl["group_num"] = (df_itt_wl["group"] == "EPS").astype(float)
    print(f"WL missing sources: Human={describe_source(source_h_wl)} | EPS={describe_source(source_e_wl)}")
    print(f"WL ITT: N={len(df_itt_wl)} (Human={sum(df_itt_wl.group == 'Human')}, EPS={sum(df_itt_wl.group == 'EPS')})")

    comp_h_gl = load_glycemic_arm(args.gly_human, "Human")
    comp_e_gl = load_glycemic_arm(args.gly_eps, "EPS")
    rng_gl = np.random.default_rng(args.seed)

    n_miss_h_gl_expected = max(0, args.n_rand_human_gl - len(comp_h_gl))
    n_miss_e_gl_expected = max(0, args.n_rand_eps_gl - len(comp_e_gl))
    miss_h_gl, source_h_gl, note_h_gl = resolve_glycemic_missing(
        comp_h_gl,
        "Human",
        n_miss_h_gl_expected,
        rng_gl,
        args.gly_human_missing,
    )
    miss_e_gl, source_e_gl, note_e_gl = resolve_glycemic_missing(
        comp_e_gl,
        "EPS",
        n_miss_e_gl_expected,
        rng_gl,
        args.gly_eps_missing,
    )
    if args.gly_human_missing and len(miss_h_gl) != n_miss_h_gl_expected:
        print(
            f"Warning: Human glycemic missing file has {len(miss_h_gl)} rows, "
            f"but randomized count implies {n_miss_h_gl_expected} missing participants."
        )
    if args.gly_eps_missing and len(miss_e_gl) != n_miss_e_gl_expected:
        print(
            f"Warning: EPS glycemic missing file has {len(miss_e_gl)} rows, "
            f"but randomized count implies {n_miss_e_gl_expected} missing participants."
        )

    for arm, n_comp, n_miss, n_rand in [
        ("Human", len(comp_h_gl), len(miss_h_gl), args.n_rand_human_gl),
        ("EPS", len(comp_e_gl), len(miss_e_gl), args.n_rand_eps_gl),
    ]:
        total = n_comp + n_miss
        mark = "OK" if total == n_rand else "WARNING"
        print(f"GL {mark}: {arm} {n_comp}+{n_miss}={total} (target={n_rand})")

    df_itt_gl = pd.concat([comp_h_gl, comp_e_gl, miss_h_gl, miss_e_gl], ignore_index=True)
    df_itt_gl["group_num"] = (df_itt_gl["group"] == "EPS").astype(float)
    print(f"GL missing sources: Human={describe_source(source_h_gl)} | EPS={describe_source(source_e_gl)}")
    if note_h_gl:
        print(f"Human glycemic note: {note_h_gl}")
    if note_e_gl:
        print(f"EPS glycemic note: {note_e_gl}")
    print(f"GL ITT: N={len(df_itt_gl)} (Human={sum(df_itt_gl.group == 'Human')}, EPS={sum(df_itt_gl.group == 'EPS')})")

    wl_covars = ["baseline_wt", "age", "sex", "bmi"]
    gl_covars = ["fpg0", "age", "sex", "bmi"]

    print("\n--- Weight Loss Tipping Points ---")
    wl_delta_range = np.arange(0, -10.1, -0.5)

    print("  A) EPS dropouts only -> wl_kg outcome...")
    tp_wl_eps_kg = tipping_search(
        df_itt_wl,
        mice_wl,
        "wl_kg",
        wl_delta_range,
        wl_covars,
        apply_to="eps_only",
        delta_target="wl_kg",
        m=args.m_imputations,
        seed=args.seed,
    )

    print("  B) EPS dropouts only -> wl_pct outcome...")
    tp_wl_eps_pct = tipping_search(
        df_itt_wl,
        mice_wl,
        "wl_pct",
        wl_delta_range,
        wl_covars,
        apply_to="eps_only",
        delta_target="wl_kg",
        m=args.m_imputations,
        seed=args.seed,
    )

    print("  C) Differential: EPS worse, Human better (+/-delta)...")
    tp_wl_diff = []
    base_wl = mice_wl(df_itt_wl, m=args.m_imputations, seed=args.seed)
    for delta in wl_delta_range:
        coefs, variances = [], []
        for dataset in base_wl:
            shifted = dataset.copy()
            missing_mask = df_itt_wl["completer"] == 0
            eps_missing = missing_mask & (df_itt_wl["group"] == "EPS")
            human_missing = missing_mask & (df_itt_wl["group"] == "Human")

            shifted.loc[eps_missing, "wl_kg"] += delta
            shifted.loc[human_missing, "wl_kg"] -= delta
            baseline_wt = shifted["baseline_wt"]
            shifted["wl_pct"] = np.where(baseline_wt > 0, shifted["wl_kg"] / baseline_wt * 100, np.nan)

            coef, se, _ = ancova_effect(shifted, "wl_kg", wl_covars)
            coefs.append(coef)
            variances.append(se**2 if not np.isnan(se) else np.nan)

        q_bar, lo, hi, p_value, _ = rubins_rules(coefs, variances)
        tp_wl_diff.append(
            {
                "delta": delta,
                "estimate": q_bar,
                "ci_lo": lo,
                "ci_hi": hi,
                "p": p_value,
                "significant": p_value < 0.05 if not np.isnan(p_value) else None,
                "ci_excludes_zero": (lo > 0 or hi < 0) if not (np.isnan(lo) or np.isnan(hi)) else None,
            }
        )

    print("\n--- Glycemic Tipping Points ---")
    gl_delta_range = np.arange(0, 5.1, 0.2)

    print("  D) EPS dropouts only -> fpg_change outcome...")
    tp_gl_eps = tipping_search(
        df_itt_gl,
        mice_gl,
        "fpg_change",
        gl_delta_range,
        gl_covars,
        apply_to="eps_only",
        delta_target="fpg1",
        m=args.m_imputations,
        seed=args.seed,
    )

    print("\n" + "=" * 60)
    print("TIPPING POINT SUMMARY")
    print("=" * 60)

    analyses = [
        ("WL (kg) - EPS dropouts worse", tp_wl_eps_kg, "kg"),
        ("WL (%) - EPS dropouts worse", tp_wl_eps_pct, "kg (applied to wl_kg)"),
        ("WL (kg) - Differential (EPS worse, Human better)", tp_wl_diff, "kg each direction"),
        ("FPG change - EPS dropouts worse", tp_gl_eps, "mmol/L on endline FPG"),
    ]

    summary_rows = []
    for name, results, unit in analyses:
        delta, p_value, lo, hi, already_nonsig = find_tipping(results)
        if already_nonsig:
            print(f"  {name}: already non-significant at delta=0, p={p_value:.4f}, CI=({lo:.3f}, {hi:.3f})")
            summary_rows.append(
                {
                    "Analysis": name,
                    "Tipping delta": "N/A (baseline non-sig)",
                    "Unit": unit,
                    "P at tipping": f"{p_value:.4f}",
                    "CI at tipping": f"({lo:.3f}, {hi:.3f})",
                    "Interpretation": "MAR baseline result is already non-significant; no additional MNAR shift is needed.",
                }
            )
        elif delta is not None:
            print(f"  {name}: tipping at delta={delta:.1f} {unit}, p={p_value:.4f}, CI=({lo:.3f}, {hi:.3f})")
            summary_rows.append(
                {
                    "Analysis": name,
                    "Tipping delta": f"{delta:.1f}",
                    "Unit": unit,
                    "P at tipping": f"{p_value:.4f}",
                    "CI at tipping": f"({lo:.3f}, {hi:.3f})",
                    "Interpretation": f"Result becomes non-significant when EPS dropouts are {abs(delta):.1f} {unit} worse than MAR",
                }
            )
        else:
            print(f"  {name}: robust across all deltas tested (up to {results[-1]['delta']:.1f})")
            summary_rows.append(
                {
                    "Analysis": name,
                    "Tipping delta": f"Beyond {results[-1]['delta']:.1f}",
                    "Unit": unit,
                    "P at tipping": "N/A",
                    "CI at tipping": "N/A",
                    "Interpretation": "Result remains significant across all tested deltas",
                }
            )

    df_summary = pd.DataFrame(summary_rows)
    df_wl_kg = detail_df(tp_wl_eps_kg)
    df_wl_pct = detail_df(tp_wl_eps_pct)
    df_wl_diff = detail_df(tp_wl_diff)
    df_gl = detail_df(tp_gl_eps)

    if "resampled_from_completers" in {source_h_wl, source_e_wl, source_h_gl, source_e_gl}:
        missing_workflow = (
            "Missing-baseline files were not supplied for at least one cohort, so the repository fallback "
            "reconstructed those missing participants by within-arm resampling from completers."
        )
    else:
        missing_workflow = "Supplied missing-baseline files were used as the real baseline records for non-completers."

    notes_rows = [
        {"Item": "Purpose", "Details": "Quantify how far missing data must deviate from MAR to nullify the EPS benefit"},
        {"Item": "Primary scenario", "Details": "Delta applied to missing EPS-arm outcomes only"},
        {
            "Item": "Differential scenario",
            "Details": "For weight loss only: missing EPS outcomes are worsened while missing Human outcomes are improved",
        },
        {
            "Item": "Weight-loss delta unit",
            "Details": "kg applied to weight-loss kg, with weight-loss percent re-derived from baseline weight",
        },
        {
            "Item": "Glycemic delta unit",
            "Details": "mmol/L applied to endline fasting glucose, with change scores re-derived afterward",
        },
        {"Item": "Tipping criterion", "Details": "First delta where p > 0.05 or the 95% CI includes zero"},
        {
            "Item": "Model alignment",
            "Details": "Covariates, naming, and MI settings match ITT_weight_loss.py and ITT_glycemic.py",
        },
        {"Item": "Missing-baseline workflow", "Details": missing_workflow},
        {"Item": "Weight missing source (Human)", "Details": describe_source(source_h_wl)},
        {"Item": "Weight missing source (EPS)", "Details": describe_source(source_e_wl)},
        {"Item": "Glycemic missing source (Human)", "Details": describe_source(source_h_gl)},
        {"Item": "Glycemic missing source (EPS)", "Details": describe_source(source_e_gl)},
        {"Item": "Imputations per delta", "Details": str(args.m_imputations)},
        {"Item": "Seed", "Details": str(args.seed)},
    ]
    if note_h_wl:
        notes_rows.append({"Item": "Weight Human preprocessing", "Details": note_h_wl})
    if note_e_wl:
        notes_rows.append({"Item": "Weight EPS preprocessing", "Details": note_e_wl})
    if note_h_gl:
        notes_rows.append({"Item": "Glycemic Human preprocessing", "Details": note_h_gl})
    if note_e_gl:
        notes_rows.append({"Item": "Glycemic EPS preprocessing", "Details": note_e_gl})
    notes = pd.DataFrame(notes_rows)

    with pd.ExcelWriter(str(out_xlsx), engine="openpyxl") as writer:
        df_summary.to_excel(writer, sheet_name="Summary", index=False)
        df_wl_kg.to_excel(writer, sheet_name="WL kg EPS-only", index=False)
        df_wl_pct.to_excel(writer, sheet_name="WL pct EPS-only", index=False)
        df_wl_diff.to_excel(writer, sheet_name="WL kg Differential", index=False)
        df_gl.to_excel(writer, sheet_name="FPG EPS-only", index=False)
        notes.to_excel(writer, sheet_name="Notes", index=False)

    print(f"\nSaved to: {out_xlsx}")


if __name__ == "__main__":
    main()
