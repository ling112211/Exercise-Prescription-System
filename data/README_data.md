# Data Dictionary

This document describes every dataset included in this repository, the column
conventions expected by each analysis script, and instructions for obtaining
the restricted participant data that cannot be shared publicly.

---

## Directory Layout

```
data/
├── example/                   # Synthetic demonstration data (publicly available)
│   ├── weight_loss/
│   │   ├── human_arm.xlsx     # Synthetic Human-arm weight-loss records (n=80)
│   │   └── eps_arm.xlsx       # Synthetic EPS-arm weight-loss records  (n=80)
│   ├── glycemic/
│   │   ├── human_arm.xlsx     # Synthetic Human-arm glycemic records   (n=40)
│   │   └── eps_arm.xlsx       # Synthetic EPS-arm glycemic records     (n=40)
│   └── questionnaire/
│       ├── human_responses.xlsx  # Synthetic Human-arm questionnaire responses (n=60)
│       └── eps_responses.xlsx    # Synthetic EPS-arm questionnaire responses   (n=60)
├── expert_pilot/              # Anonymised expert pilot scores (publicly available)
│   ├── base_model.csv         # Scores for base LLM outputs (25 raters)
│   ├── eps_without_d2.csv     # Scores after SFT only (25 raters)
│   └── eps.csv                # Scores after SFT + KTO (25 raters)
└── benchmark_results/         # Benchmark accuracy table (publicly available)
    └── benchmark_accuracy.csv # Per-model, per-benchmark accuracy with 95% CI
```

---

## 1. Clinical Trial Data — Weight-Loss Cohort

**Files:** `data/example/weight_loss/human_arm.xlsx`,
`data/example/weight_loss/eps_arm.xlsx`

**Scripts that consume this data:**
- `clinical_trial/baseline_characteristics.py`
- `clinical_trial/weight_loss_analysis.py`
- `Subgroup Forest Plot/weight-loss subgroup forest plot.py`

**Column specification:**

| Column | Type | Unit / Encoding | Notes |
|--------|------|-----------------|-------|
| `age` | integer | years | Participant age |
| `sex` | string | `F` = female, `M` = male | Sex at enrolment |
| `BMI` | float | kg/m² | Body mass index at enrolment |
| `baseline_weight_kg` | float | kg | Body weight at programme entry (baseline) |
| `weight_loss_kg` | float | kg | Absolute weight lost over the programme (positive = loss) |
| `weight_loss_pct` | float | % | Percentage weight lost: `(weight_loss_kg / baseline_weight_kg) × 100` |

**Notes:**

- The `weight_loss_pct` column is optional. When absent, `weight_loss_analysis.py`
  derives it from `weight_loss_kg` and `baseline_weight_kg` automatically.
- The real participant data (n=1,444 per arm) cannot be shared publicly due to
  participant privacy obligations under Chinese law. To request access, see
  Section 6 below.

---

## 2. Clinical Trial Data — Glycemic-Control Cohort

**Files:** `data/example/glycemic/human_arm.xlsx`,
`data/example/glycemic/eps_arm.xlsx`

**Scripts that consume this data:**
- `clinical_trial/baseline_characteristics.py`
- `clinical_trial/glycemic_control_analysis.py`
- `Subgroup Forest Plot/glycemic control subgroup forest plot.py`

**Column specification:**

| Column | Type | Unit / Encoding | Notes |
|--------|------|-----------------|-------|
| `age` | integer | years | Participant age |
| `sex` | string | `F` = female, `M` = male | Same encoding as weight-loss cohort |
| `BMI` | float | kg/m² | Body mass index at enrolment |
| `baseline_fpg_mmol` | float | mmol/L | Fasting plasma glucose at programme entry (baseline) |
| `endpoint_fpg_mmol` | float | mmol/L | Fasting plasma glucose at programme exit (follow-up) |

**Notes:**

- The scripts derive the change score `Δ FPG = baseline_fpg_mmol − endpoint_fpg_mmol` internally;
  no pre-computed change column is required.
- The real participant data (n=40 per arm) cannot be shared publicly. See
  Section 6 for access instructions.

---

## 3. Participant-Reported Questionnaire Data

**Files:** `data/example/questionnaire/human_responses.xlsx`,
`data/example/questionnaire/eps_responses.xlsx`

**Script that consumes this data:**
- `questionnaire/participant_reported.py`

**Column specification:**

| Column | Type | Encoding | Notes |
|--------|------|----------|-------|
| `id` | integer | sequential ID | Participant identifier; used only for record tracking |
| `Q1` | string | `A. Yes...` / `B. No...` | Screening item: whether the participant used the prescribed exercise plan. Only rows with an `A`-prefixed answer are retained after filtering. |
| `Q2` – `Q15` | integer | 1–7 | 14 experience-rating items on a 7-point Likert scale. Values are integers from 1 (strongly disagree) to 7 (strongly agree). |

**Column naming convention:** The script `participant_reported.py` auto-detects
question columns by matching headers against `Q{n}` (e.g. `Q1`, `Q2`, …, `Q15`).
Legacy `{n}.` format headers (e.g. `1.`, `2.`, …) are also accepted.

**Notes:**

- The real questionnaire responses cannot be shared publicly. See Section 6.

---

## 4. Expert Pilot Evaluation Data

**Files:** `data/expert_pilot/base_model.csv`,
`data/expert_pilot/eps_without_d2.csv`,
`data/expert_pilot/eps.csv`

**Script that consumes this data:**
- `expert_pilot/plot_expert_evaluation.py`

**Column specification:**

| Column | Type | Encoding | Notes |
|--------|------|----------|-------|
| `rater_id` | integer | 1–25 | Anonymous rater identifier; consistent across all three files so paired tests can be run |
| `Q1` | string | `A` / `B` / `C` / `D` | Safety and contraindication awareness |
| `Q2` | string | `A` / `B` / `C` / `D` | Appropriateness of exercise type |
| `Q3` | string | `A` / `B` / `C` / `D` | Appropriateness of exercise intensity |
| `Q4` | string | `A` / `B` / `C` / `D` | Appropriateness of exercise frequency |
| `Q5` | string | `A` / `B` / `C` / `D` | Appropriateness of exercise duration |
| `Q6` | string | `A` / `B` / `C` / `D` | Clarity and practicality of instructions |
| `Q7` | string | `A` / `B` / `C` / `D` | Overall prescription quality |

**Grade encoding:** `A` = 0 (best), `B` = 1, `C` = 2, `D` = 3 (worst). Lower
scores indicate higher quality. The script converts grades to numeric scores
internally.

**Anonymisation procedure:** All personal identifiers present in the original
WeChat survey export (WeChat ID, display name, submission timestamp, IP address,
user agent string, geographic location) were removed. Only the bare letter grade
for each dimension was retained and a sequential `rater_id` was assigned.
The 25 raters are the same individuals across all three files; `rater_id`
alignment enables paired significance testing.

**Three model conditions:**

| File | Model condition |
|------|----------------|
| `base_model.csv` | Base LLM (DeepSeek-R1-14B), no fine-tuning |
| `eps_without_d2.csv` | Base LLM + SFT on domain data (D1 only) |
| `eps.csv` | Base LLM + SFT + KTO alignment (D1 + D2; full EPS) |

---

## 5. Benchmark Accuracy Results

**File:** `data/benchmark_results/benchmark_accuracy.csv`

**Scripts that consume this data:**
- `benchmark/evaluate_benchmark.py` (generates this file as output)
- `benchmark/plot_benchmark.py` (reads this file to produce Fig. 3)

This file contains the pre-computed benchmark evaluation results used to
produce the accuracy comparison figures. It may be used directly without
re-running the benchmark evaluation.

**Column specification:**

| Column | Type | Notes |
|--------|------|-------|
| `group` | string | Model group: `Base model`, `EPS`, or `Mainstream LLMs` |
| `model` | string | Model display name (e.g. `DeepSeek-R1-8B`, `GPT-4o`) |
| `benchmark` | string | Benchmark name: `CMB`, `CMExam`, `MedMCQA`, or `MedQA` |
| `mean_accuracy_pct` | float | Mean accuracy across 10 independent runs, as a percentage |
| `ci_low_pct` | float | Lower bound of the 95% confidence interval (percentage) |
| `ci_high_pct` | float | Upper bound of the 95% confidence interval (percentage) |
| `n_questions` | integer | Number of questions in the benchmark subset |

**Benchmark descriptions:**

| Benchmark | Domain | Language | Questions in subset |
|-----------|--------|----------|---------------------|
| CMB | Chinese clinical medicine | Chinese | 252 |
| CMExam | Chinese medical licensing exam | Chinese | 61 |
| MedMCQA | Medical multiple-choice QA | English | 270 |
| MedQA | USMLE-style medical QA | English | 138 |

---

## 6. Requesting Access to Restricted Participant Data

The real clinical trial and questionnaire data (Sections 1–3) cannot be shared
in this repository due to participant privacy obligations under the *Personal
Information Protection Law of the People's Republic of China* (PIPL, 2021) and
the terms of the ethics approval.

Researchers wishing to access the restricted data for replication or secondary
analysis should contact the corresponding author. Requests will be evaluated
on a case-by-case basis and may require a data-sharing agreement.

**Contact:** Guangxin Jiang — *[corresponding author email]*

---

## 7. Reproducing Results with the Example Data

The synthetic example data are designed to allow every analysis script to run
end-to-end and produce outputs of the correct format. The numerical values will
differ from the paper's reported results because the data are synthetic.

```bash
# Baseline characteristics (Tables 1 and 2)
python clinical_trial/baseline_characteristics.py \
  --weight_human data/example/weight_loss/human_arm.xlsx \
  --weight_eps   data/example/weight_loss/eps_arm.xlsx \
  --gly_human    data/example/glycemic/human_arm.xlsx \
  --gly_eps      data/example/glycemic/eps_arm.xlsx \
  --out_dir      outputs/clinical_trial

# Weight-loss outcome analysis (Fig. 5)
python clinical_trial/weight_loss_analysis.py \
  --weight_human data/example/weight_loss/human_arm.xlsx \
  --weight_eps   data/example/weight_loss/eps_arm.xlsx \
  --out_dir      outputs/clinical_trial

# Glycemic-control outcome analysis (Fig. 6)
python clinical_trial/glycemic_control_analysis.py \
  --gly_human data/example/glycemic/human_arm.xlsx \
  --gly_eps   data/example/glycemic/eps_arm.xlsx \
  --out_dir   outputs/clinical_trial

# Participant-reported outcomes (Fig. 7)
python questionnaire/participant_reported.py \
  --human-xlsx data/example/questionnaire/human_responses.xlsx \
  --eps-xlsx   data/example/questionnaire/eps_responses.xlsx \
  --outdir     outputs/questionnaire

# Expert pilot evaluation (Fig. 3)
python expert_pilot/plot_expert_evaluation.py \
  --input-dir data/expert_pilot \
  --outdir    outputs/expert_pilot
```
