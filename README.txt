CURRENT EXPERIMENT OUTPUTS
==========================

These CSV/JSON artifacts were generated from the current python-experiment.py implementation.

Files:
- metrics_overall_oos.csv       : all 10 train/test seed-pair results for Rule, MAD, IF_OOS, LOF_OOS, Hybrid
- metrics_multiseed_summary.csv : mean +/- SD summary across the 10 seed pairs
- metrics_per_class_oos.csv     : per-anomaly-class recall mean +/- SD across the 10 seed pairs
- metrics_sensitivity_oos.csv   : Isolation Forest contamination sensitivity (c = 0.05, 0.10, 0.15, 0.20)
- metrics_mad_sensitivity.csv   : MAD magnitude sensitivity (extreme, moderate, subtle)
- false_positive_analysis.json  : Seed 42 -> 100 false-positive analysis for Isolation Forest
