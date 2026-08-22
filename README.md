# Benchmarking Data-Integrity Anomaly Detection in Blood-Bank Transaction Records

## Overview

This repository contains the reproducible experimental code and result artifacts for the research study:

**“Benchmarking Data-Integrity Anomaly Detection in Blood-Bank Transaction Records”**

The study evaluates three complementary anomaly-detection approaches for identifying integrity problems in synthetic blood-bank transaction records:

1. **Rule-Based Validator**
2. **Statistical Anomaly Detector (MAD)**
3. **Unsupervised Machine-Learning Baselines (Isolation Forest and LOF)**

A **Hybrid Detector** combines the Rule-Based Validator and MAD detector.

The purpose of this benchmark is to compare different anomaly-detection approaches under controlled and reproducible conditions. The dataset is synthetic, and the results should not be interpreted as evidence of clinical performance.

---
## Visual Benchmark Results

### 1. Overall Performance Comparison
![Benchmark Comparison](main/benchmark_comparison.png)

*(A) Mean ± SD out-of-sample F1-score across 10 independent seed pairs. (B) Model precision, recall, and false-positive rates.*

### 2. Per-Class Anomaly Detection Recall
![Per-Class Anomaly Detection Recall](main/per_class_recall.png)

*Comparison of anomaly detection recall across all nine injected blood-bank data integrity failure modes.*

---

## Methods

### 1. Rule-Based Validator

The Rule-Based Validator checks deterministic domain and data-integrity constraints, including:

- Required-field completeness
- Blood-group validity
- Valid transaction states
- Unit-ID uniqueness
- Donor-ID referential integrity
- Donor and blood-group consistency
- Issued-state constraints
- Collection, issue, and expiry timestamp relationships

The mandatory-field validation includes `expiry_timestamp`.

### 2. Statistical Anomaly Detector

The statistical detector uses a **Median Absolute Deviation (MAD)** based modified Z-score.

MAD parameters are calibrated exclusively on the training dataset.

A modified Z-score greater than **3.5** is classified as anomalous.

### 3. Machine-Learning Baselines

Two unsupervised machine-learning methods are evaluated:

- **Isolation Forest (IF)**
- **Local Outlier Factor (LOF)**

They operate on a common feature space containing numerical variables and one-hot encoded categorical variables.

The feature representation includes:

- Donor weight
- Donor hemoglobin
- Quantity issued
- Collection-to-issue lead time
- Blood group
- State
- Test status
- Component
- Storage location

A predefined global `OneHotEncoder` vocabulary is used to maintain a consistent feature space between independent training and test datasets.

### 4. Hybrid Detector

The Hybrid Detector combines deterministic and statistical detection:

`Hybrid = Rule-Based Validator OR MAD`

A record is classified as anomalous if either detector flags it.

---

## Synthetic Dataset

Each experiment generates:

- **2,000 transaction records**
- **300 anomalous records**
- **1,700 clean records**

The anomalies are distributed across nine predefined classes:

1. `MISSING_FIELDS`
2. `DUPLICATE_RECORDS`
3. `CONFLICTING_RECORDS`
4. `IMPOSSIBLE_STATES`
5. `EXPIRY_VIOLATIONS`
6. `TRACEABILITY_BREAKS`
7. `FORMAT_INCONSISTENCIES`
8. `TEMPORAL_ANOMALIES`
9. `STATISTICAL_BURSTS`

The anomaly classes are injected in a controlled and mutually exclusive manner so that each anomalous record belongs to one predefined anomaly class.

The `STATISTICAL_BURSTS` class contains deliberately exaggerated numerical deviations. The extreme configuration uses synthetic values such as **188.5 kg donor weight** and **24.8 g/dL hemoglobin** to evaluate the upper detection range of the MAD detector.

---

## Experimental Protocol

The benchmark uses 10 independent train/test seed pairs:

- (42, 100)
- (43, 101)
- (44, 102)
- (45, 103)
- (46, 104)
- (47, 105)
- (48, 106)
- (49, 107)
- (50, 108)
- (51, 109)

For every seed pair:

1. A clean training dataset is generated.
2. Training anomalies are injected for the ML training procedure.
3. MAD parameters are calibrated using the training dataset.
4. Isolation Forest is fitted on the training feature matrix.
5. LOF is fitted on the training feature matrix.
6. An independent test dataset is generated.
7. Test anomalies are injected into the independent test dataset.
8. All detectors are evaluated on the test dataset.

The evaluation is performed out-of-sample for the ML baselines and MAD detector.

---

## Evaluation Metrics

The following metrics are calculated:

- True Positives (TP)
- False Positives (FP)
- False Negatives (FN)
- True Negatives (TN)
- Precision
- Recall
- F1 Score
- False Positive Rate (FPR)

Results are summarized using **Mean ± Standard Deviation across the 10 independent seed pairs**.

Per-class recall is also reported for all nine anomaly classes.

---

## Sensitivity Analyses

### MAD Magnitude Sensitivity

The MAD detector is evaluated under three anomaly magnitudes:

- Extreme
- Moderate
- Subtle

This experiment evaluates how detection performance changes as the magnitude of the numerical deviation decreases.

### Isolation Forest Contamination Sensitivity

Isolation Forest is evaluated using four contamination values:

- 0.05
- 0.10
- 0.15
- 0.20

This evaluates the trade-off between anomaly recall and false-positive rate under different assumed contamination levels.

---

## Latest Results

The current experimental configuration produces the following mean ± SD results across the 10 seed pairs:

| Model | Precision | Recall | F1 | FPR |
|---|---:|---:|---:|---:|
| Rule | 1.000 ± 0.000 | 0.890 ± 0.000 | 0.942 ± 0.000 | 0.000 ± 0.000 |
| MAD | 1.000 ± 0.000 | 0.110 ± 0.000 | 0.198 ± 0.000 | 0.000 ± 0.000 |
| IF | 0.262 ± 0.046 | 0.269 ± 0.046 | 0.265 ± 0.045 | 0.134 ± 0.014 |
| LOF | 0.348 ± 0.037 | 0.390 ± 0.041 | 0.367 ± 0.035 | 0.130 ± 0.018 |
| Hybrid | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.000 ± 0.000 |

The Hybrid detector achieved complete coverage of the predefined synthetic anomalies in this benchmark.

This result represents **complete coverage of the controlled synthetic benchmark only**. It does not imply guaranteed performance on real-world blood-bank data or clinical deployment.

---

## Reproducibility

The experiment is designed to be reproducible using fixed random seeds.

The Python script reports the execution environment, including:

- Python version
- scikit-learn version
- NumPy version
- pandas version

Run the experiment with:

`python python-experiment.py`

The script automatically performs the complete benchmark and exports the result artifacts as CSV files.

---

## Output Files

The experiment generates the following result files:

- `metrics_overall_oos.csv`
- `metrics_multiseed_summary.csv`
- `metrics_sensitivity_oos.csv`
- `metrics_per_class_oos.csv`
- `metrics_mad_sensitivity.csv`

### Output Descriptions

| File | Description |
|---|---|
| `metrics_overall_oos.csv` | Overall primary out-of-sample metrics |
| `metrics_multiseed_summary.csv` | Mean ± SD results across the 10 seed pairs |
| `metrics_sensitivity_oos.csv` | Isolation Forest contamination sensitivity results |
| `metrics_per_class_oos.csv` | Per-class anomaly recall results |
| `metrics_mad_sensitivity.csv` | MAD anomaly-magnitude sensitivity results |

---

## Repository Structure

```text
.
├── README.md
├── python-experiment.py
├── manuscript.pdf
├── metrics_overall_oos.csv
├── metrics_multiseed_summary.csv
├── metrics_sensitivity_oos.csv
├── metrics_per_class_oos.csv
└── metrics_mad_sensitivity.csv

```

## Limitations

This benchmark is based entirely on **synthetically generated data** with controlled anomaly injection.

Therefore:

- The results describe performance on the predefined synthetic benchmark.
- The anomaly taxonomy does not cover every possible blood-bank data-integrity failure.
- The extreme numerical anomalies are intentionally exaggerated.
- Unsupervised ML performance depends on feature representation and contamination settings.
- The benchmark does not establish clinical safety, regulatory compliance, or production readiness.
- Validation on real-world, appropriately governed and de-identified blood-bank data would be required before making operational or clinical claims.

---

## Research Interpretation

The objective is not to establish a universally superior anomaly-detection algorithm.

Instead, the benchmark demonstrates that different anomaly types have different structural characteristics.

**Rule-based validation** is well suited to deterministic integrity violations such as missing fields, invalid states, duplicate identifiers, expiry violations, and traceability failures.

**MAD-based detection** is effective for sufficiently large numerical deviations but becomes insensitive to smaller deviations.

**Unsupervised machine-learning methods** can detect some anomalies, but their performance varies substantially across anomaly classes and parameter settings.

The Hybrid detector demonstrates how deterministic validation and robust statistical detection can complement each other within the controlled benchmark.

---

## Citation

If you use this benchmark, code, or results in academic work, please cite the accompanying manuscript:

> Aryan Jain, “Benchmarking Data-Integrity Anomaly Detection in Blood-Bank Transaction Records.”

---

## Disclaimer

This repository is intended for **research and educational purposes only**.

It is not a medical device, clinical decision-support system, blood-bank management system, or substitute for validated clinical or regulatory software.
