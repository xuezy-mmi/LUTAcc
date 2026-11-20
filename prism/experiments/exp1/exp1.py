import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D
from pathlib import Path

# Load data
df = pd.read_csv("results/accuracy_samples.csv")
df["primitive_label"] = df["primitive"].astype(str).str.upper()
df["precision"] = df["precision"].astype(str)

precision_order = ["W4A8", "W4A16", "W8A8", "W8A16"]
primitive_order = ["MAC", "LUT", "HYBRID"]

# Let's test a 3x1 vertical layout
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "mathtext.fontset": "dejavusans",
})

palette = {
    "MAC": "#3b6998",
    "LUT": "#469c76",
    "HYBRID": "#d35e60"
}

metric_configs = [
    ("latency_error_pct", "Latency Error (%)", "(a) Latency Error"),
    ("power_error_pct", "Power Error (%)", "(b) Power Error"),
    ("area_error_pct", "Area Error (%)", "(c) Area Error"),
]

fig, axes = plt.subplots(3, 1, figsize=(11, 11), sharex=True)

for i, (ax, (metric_col, ylabel, title)) in enumerate(zip(axes, metric_configs)):
    sns.boxplot(
        data=df,
        x="precision",
        y=metric_col,
        hue="primitive_label",
        order=precision_order,
        hue_order=primitive_order,
        palette=palette,
        ax=ax,
        width=0.62,
        linewidth=1.2,
        boxprops=dict(alpha=0.9, edgecolor="#2b2b2b"),
        whiskerprops=dict(color="#2b2b2b", linewidth=1.2),
        capprops=dict(color="#2b2b2b", linewidth=1.2),
        medianprops=dict(color="#111111", linewidth=1.6),
        showmeans=True,
        meanprops={
            "marker": "D",
            "markerfacecolor": "white",
            "markeredgecolor": "#222222",
            "markersize": 4.5,
            "markeredgewidth": 1.1,
        },
        flierprops={
            "marker": "o",
            "markerfacecolor": "gray",
            "markeredgecolor": "none",
            "markersize": 3.0,
            "alpha": 0.45,
        },
    )
    ax.set_title(title, fontweight="bold", pad=8, loc="left", fontsize=12)
    ax.set_ylabel(ylabel, fontweight="bold", labelpad=8)
    
    # Only label xlabel on the bottom or all?
    if i == len(axes) - 1:
        ax.set_xlabel("Precision", fontweight="bold", labelpad=8)
    else:
        ax.set_xlabel("")
    
    ax.yaxis.grid(True, linestyle="--", linewidth=0.7, alpha=0.5, color="gray")
    ax.xaxis.grid(False)
    ax.set_axisbelow(True)

    y_vals = df[metric_col].to_numpy()
    y_min, y_max = np.min(y_vals), np.max(y_vals)
    y_span = y_max - y_min
    if y_span > 0:
        pad = y_span * 0.08
        ax.set_ylim(y_min - pad, y_max + pad)

    if ax.get_legend():
        ax.get_legend().remove()

# Legend on top
handles, labels = axes[0].get_legend_handles_labels()
mean_proxy = Line2D(
    [0], [0],
    marker="D",
    color="w",
    markeredgecolor="#222222",
    markeredgewidth=1.1,
    markerfacecolor="white",
    markersize=5.5,
    label="Mean",
)
handles.append(mean_proxy)
labels.append("Mean")

fig.legend(
    handles,
    labels,
    title="Primitive",
    title_fontsize=11.5,
    ncol=len(labels),
    loc="upper center",
    bbox_to_anchor=(0.5, 0.99),
    frameon=True,
    fancybox=True,
    facecolor="#fafafa",
    edgecolor="#cccccc",
)

plt.tight_layout()
plt.subplots_adjust(top=0.93)
plt.savefig("test_3x1_boxplots.pdf", dpi=200, bbox_inches="tight")
# print("Saved test_3x1_boxplots.png")