import os
import matplotlib.pyplot as plt
import numpy as np

os.makedirs('assets', exist_ok=True)

models = ['Rule-Based\nValidator', 'Statistical\nMAD', 'Isolation\nForest (ML)', 'Local Outlier\nFactor (ML)', 'Hybrid\n(Rule + MAD)']
f1_means = [0.942, 0.198, 0.265, 0.366, 1.000]
f1_sds = [0.000, 0.000, 0.045, 0.035, 0.000]
colors = ['#2b5c8f', '#e67e22', '#7f8c8d', '#8e44ad', '#27ae60']

# 1. Benchmark Comparison (2-Panel)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.2), gridspec_kw={'width_ratios': [1.1, 1]})

bars = ax1.bar(models, f1_means, yerr=f1_sds, capsize=5, color=colors, width=0.55, edgecolor='black', linewidth=0.8, zorder=3)
ax1.set_ylabel('F1-Score (Out-of-Sample)', fontsize=11, fontweight='bold', labelpad=8)
ax1.set_title('(A) F1-Score Benchmark (Mean ± SD)', fontsize=12, fontweight='bold', pad=12)
ax1.set_ylim(0, 1.15)
ax1.grid(axis='y', linestyle='--', alpha=0.5, zorder=0)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

for bar, val, sd in zip(bars, f1_means, f1_sds):
    yval = bar.get_height()
    label = f"{val:.3f}" if sd == 0 else f"{val:.3f}\n±{sd:.3f}"
    ax1.text(bar.get_x() + bar.get_width()/2.0, yval + (0.05 if sd > 0 else 0.02), label, ha='center', va='bottom', fontsize=9.5, fontweight='bold')

x = np.arange(len(models))
width = 0.25
prec_means = [1.000, 1.000, 0.262, 0.347, 1.000]
rec_means = [0.890, 0.110, 0.269, 0.390, 1.000]
fpr_means = [0.000, 0.000, 0.134, 0.130, 0.000]

ax2.bar(x - width, prec_means, width, label='Precision', color='#2980b9', edgecolor='black', linewidth=0.6, zorder=3)
ax2.bar(x, rec_means, width, label='Recall', color='#27ae60', edgecolor='black', linewidth=0.6, zorder=3)
ax2.bar(x + width, fpr_means, width, label='False Positive Rate', color='#c0392b', edgecolor='black', linewidth=0.6, zorder=3)

ax2.set_ylabel('Score / Rate', fontsize=11, fontweight='bold', labelpad=8)
ax2.set_title('(B) Precision, Recall & False Positive Rate', fontsize=12, fontweight='bold', pad=12)
ax2.set_xticks(x)
ax2.set_xticklabels(['Rule', 'MAD', 'IF', 'LOF', 'Hybrid'], fontsize=10, fontweight='bold')
ax2.set_ylim(0, 1.18)
ax2.grid(axis='y', linestyle='--', alpha=0.5, zorder=0)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.legend(frameon=True, loc='upper left', fontsize=9)

plt.tight_layout()
plt.savefig('assets/benchmark_comparison.png', dpi=300)
plt.close()

# 2. Per-Class Recall Breakdown
classes = [
    'Missing Fields', 'Duplicate Records', 'Conflicting Records', 'Impossible States',
    'Expiry Violations', 'Traceability Breaks', 'Format Inconsistencies', 'Temporal Anomalies', 'Statistical Bursts'
]
rule_rec = [100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 0.0]
mad_rec  = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 100.0]
if_rec   = [19.4, 12.4, 16.8, 77.0, 19.7, 8.8, 14.5, 3.0, 71.2]
lof_rec  = [14.7, 11.5, 10.9, 78.5, 42.7, 15.2, 11.5, 69.7, 98.5]
hyb_rec  = [100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0]

y_pos = np.arange(len(classes))
fig, ax = plt.subplots(figsize=(11.5, 6.2))
h = 0.16
ax.barh(y_pos - 2*h, rule_rec, height=h, label='Rule-Based Validator', color='#2b5c8f', edgecolor='black', linewidth=0.5)
ax.barh(y_pos - h, mad_rec, height=h, label='MAD Statistical Detector', color='#e67e22', edgecolor='black', linewidth=0.5)
ax.barh(y_pos, if_rec, height=h, label='Isolation Forest (ML)', color='#7f8c8d', edgecolor='black', linewidth=0.5)
ax.barh(y_pos + h, lof_rec, height=h, label='Local Outlier Factor (ML)', color='#8e44ad', edgecolor='black', linewidth=0.5)
ax.barh(y_pos + 2*h, hyb_rec, height=h, label='Hybrid (Rule + MAD)', color='#27ae60', edgecolor='black', linewidth=0.5)

ax.set_yticks(y_pos)
ax.set_yticklabels(classes, fontsize=10, fontweight='bold')
ax.invert_yaxis()
ax.set_xlabel('Anomaly Detection Recall (%)', fontsize=11, fontweight='bold', labelpad=8)
ax.set_title('Per-Class Anomaly Detection Recall Across 9 Failure Modes', fontsize=12, fontweight='bold', pad=12)
ax.set_xlim(0, 112)
ax.grid(axis='x', linestyle='--', alpha=0.5)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.legend(loc='lower right', frameon=True, fontsize=9.5)

plt.tight_layout()
plt.savefig('assets/per_class_recall.png', dpi=300)
plt.close()
print("Charts generated successfully.")
