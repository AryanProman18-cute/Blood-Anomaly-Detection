# Benchmarking Data-Integrity Anomaly Detection in Blood-Bank Transaction Records

## Overview

This repository contains the reproducible experimental code and result artifacts for the research study:

**“Benchmarking Data-Integrity Anomaly Detection in Blood-Bank Transaction Records”**

The study evaluates three complementary anomaly-detection approaches for identifying integrity problems in synthetic blood-bank transaction records:

1. **Rule-Based Validator**
2. **Statistical Anomaly Detector (MAD)**
3. **Unsupervised Machine-Learning Baselines (Isolation Forest and LOF)**

A **Hybrid Detector** combines the Rule-Based Validator and MAD detector.

The purpose of this benchmark is to compare different anomaly-detection approaches under controlled and reproducible conditions. The dataset is synthetic and the results should not be interpreted as evidence of clinical performance.

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

The mandatory-field validation also includes `expiry_timestamp`.

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

A predefined global `OneHotEncoder` vocabulary is used to maintain a consistent feature space between independent training and test datasets.

### 4. Hybrid Detector

The Hybrid Detector combines deterministic and statistical detection:

```text
Hybrid = Rule-Based Validator OR MAD
