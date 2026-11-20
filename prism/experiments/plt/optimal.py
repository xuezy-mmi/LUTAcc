#!/usr/bin/env python3
"""
PRISM Page-1 Motivating 3D Waterfall Figure Generator
-----------------------------------------------------
Visualizes the joint trade-off across:
  - X-axis: Precision & Format (W2A4, W4A4, W4A8, W4A16, W8A8, W8A16)
  - Y-axis: Spatial Array Scale (4x4, 8x8, 16x16, 32x32, 64x64, 128x128)
  - Z-axis: Normalized Hardware Efficiency (1 / Normalized EDP, in [0, 1])
  - 4 Distinct Curves per Slice:
      1. LUT Array Full Curve (Teal dashed + triangle markers)
      2. Hybrid MAC-LUT Full Curve (Amber dash-dot + diamond markers)
      3. MAC Array Full Curve (Steel-Blue dotted + square markers)
      4. PRISM Optimal Envelope (Bold Crimson solid curve tracking max(LUT, Hybrid, MAC))
"""

import pathlib
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.lines import Line2D

# Set publication-quality typography
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9.5,
    "axes.labelsize": 10.5,
    "axes.titlesize": 11.2,
    "xtick.labelsize": 8.6,
    "ytick.labelsize": 8.6,
    "figure.dpi": 300,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

PREC_LABELS = ["W2A4", "W4A4", "W4A8", "W4A16", "W8A8", "W8A16"]
SCALE_LABELS = ["4×4", "8×8", "16×16", "32×32", "64×64", "128×128"]

# Four distinct colors: 3 individual architecture curves + 1 common optimal envelope
COLOR_LUT = "#009E73"      # Teal-Green: LUT Array Full Curve
COLOR_HYBRID = "#E68A00"   # Amber-Orange: Hybrid MAC-LUT Full Curve
COLOR_MAC = "#1F65A6"      # Steel-Blue: MAC Array Full Curve
COLOR_ENV = "#D62828"      # Bold Crimson: Common Optimal Envelope (Max Curve)


def evaluate_arch_efficiency(x_grid, scale_idx):
    """
    Analytical efficiency curves Z_lut(x), Z_hybrid(x), Z_mac(x) in [0, 1]
    across precision index x in [0, 5] and array scale index scale_idx in {0..5}.
    """
    s = float(scale_idx)

    # 1. LUT Array: peaks at ultra-low bitwidth (W2A4/W4A4), drops rapidly as
    #    activation slicing D = ceil(p_a/d) and table size 2^(Gd) grow.
    lut_base = 0.35 + 0.116 * s
    z_lut = lut_base * np.exp(-0.36 * (x_grid ** 1.32))
    z_lut = np.clip(z_lut, 0.06, 0.95)

    # 2. MAC Array: steady across precision, wins at high bitwidth (W8A8/W8A16).
    mac_base = 0.24 + 0.088 * s
    z_mac = mac_base * (1.0 - 0.095 * x_grid) + 0.03 * np.exp(-((x_grid - 4.2) ** 2) / 2.0)
    z_mac = np.clip(z_mac, 0.10, 0.85)

    # 3. Hybrid MAC-LUT Array: bridges the W4A8-W4A16 transition zone via
    #    heterogeneous column partitioning (0 < f < 1).
    z_lut_125 = lut_base * np.exp(-0.36 * (1.25 ** 1.32))
    z_mac_325 = mac_base * (1.0 - 0.095 * 3.25) + 0.03 * np.exp(-((3.25 - 4.2) ** 2) / 2.0)
    hyb_center = 2.15
    hyb_peak = 0.56 * z_lut_125 + 0.44 * z_mac_325 + 0.065 + 0.012 * s
    z_hybrid = (
        hyb_peak * np.exp(-((x_grid - hyb_center) ** 2) / 2.45)
        + 0.18 * z_mac
    )
    # Calibrate so Hybrid crosses LUT around x ~ 1.25 and MAC around x ~ 3.25
    z_hybrid = 0.82 * z_hybrid + 0.04
    z_hybrid = np.clip(z_hybrid, 0.08, 0.92)

    # Common Optimal Envelope: point-wise maximum across all three architectures
    stack = np.vstack([z_lut, z_hybrid, z_mac])
    winner = np.argmax(stack, axis=0)
    z_opt = np.max(stack, axis=0)
    return z_lut, z_hybrid, z_mac, z_opt, winner


def generate_3d_tradeoff_figure(out_png, out_pdf):
    fig = plt.figure(figsize=(8.8, 6.8))
    ax = fig.add_subplot(111, projection="3d")
    fig.subplots_adjust(left=0.02, right=0.92, bottom=0.06, top=0.88)

    # Clean white 3D panes
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor("#D0D7DE")
    ax.yaxis.pane.set_edgecolor("#D0D7DE")
    ax.zaxis.pane.set_edgecolor("#D0D7DE")
    ax.grid(True, linestyle=":", linewidth=0.55, color="#B0B8C0", alpha=0.65)

    x_fine = np.linspace(0, 5, 201)
    x_markers = np.arange(len(PREC_LABELS), dtype=float)
    n_scales = len(SCALE_LABELS)

    cross_lh_x, cross_lh_y, cross_lh_z = [], [], []
    cross_hm_x, cross_hm_y, cross_hm_z = [], [], []
    peak_x, peak_y, peak_z = [], [], []

    # Draw slices from back (128x128) to front (4x4) for proper 3D painter's order
    for s_idx in range(n_scales - 1, -1, -1):
        z_lut, z_hyb, z_mac, z_opt, _ = evaluate_arch_efficiency(x_fine, s_idx)
        z_lut_m, z_hyb_m, z_mac_m, _, _ = evaluate_arch_efficiency(x_markers, s_idx)
        y_vec = np.full_like(x_fine, float(s_idx))
        y_m = np.full_like(x_markers, float(s_idx))

        # Find exact crossover indices on x_fine
        idx_lh = int(np.argmin(np.abs(z_lut[:100] - z_hyb[:100])))
        idx_hm = 90 + int(np.argmin(np.abs(z_hyb[90:] - z_mac[90:])))
        x_lh = x_fine[idx_lh]
        x_hm = x_fine[idx_hm]

        # Light translucent curtain under the winning regions for 3D depth
        regions = [
            (0.0, x_lh, COLOR_LUT),
            (x_lh, x_hm, COLOR_HYBRID),
            (x_hm, 5.0, COLOR_MAC),
        ]
        for (x_start, x_end, f_col) in regions:
            mask = (x_fine >= x_start - 1e-6) & (x_fine <= x_end + 1e-6)
            xs = x_fine[mask]
            zs = z_opt[mask]
            if len(xs) < 2:
                continue
            verts_x = np.concatenate([[xs[0]], xs, [xs[-1]]])
            verts_y = np.full_like(verts_x, float(s_idx))
            verts_z = np.concatenate([[0.0], zs, [0.0]])
            poly = Poly3DCollection(
                [list(zip(verts_x, verts_y, verts_z))],
                facecolors=f_col,
                edgecolors="none",
                alpha=0.18,
                zsort="average",
            )
            ax.add_collection3d(poly)

        # Vertical drop lines at crossover points
        ax.plot([x_lh, x_lh], [s_idx, s_idx], [0.0, z_opt[idx_lh]],
                color="#555555", linestyle=":", linewidth=0.85, alpha=0.7)
        ax.plot([x_hm, x_hm], [s_idx, s_idx], [0.0, z_opt[idx_hm]],
                color="#555555", linestyle=":", linewidth=0.85, alpha=0.7)

        # 1. Full LUT Array curve (Teal dashed + triangles)
        ax.plot(x_fine, y_vec, z_lut, color=COLOR_LUT, linestyle="--",
                linewidth=1.55, alpha=0.92, zorder=12)
        ax.scatter(x_markers, y_m, z_lut_m, color=COLOR_LUT, marker="^",
                   s=20, depthshade=False, zorder=13)

        # 2. Full Hybrid MAC-LUT curve (Amber dash-dot + diamonds)
        ax.plot(x_fine, y_vec, z_hyb, color=COLOR_HYBRID, linestyle="-.",
                linewidth=1.55, alpha=0.92, zorder=14)
        ax.scatter(x_markers, y_m, z_hyb_m, color=COLOR_HYBRID, marker="D",
                   s=18, depthshade=False, zorder=15)

        # 3. Full MAC Array curve (Steel-Blue dotted + squares)
        ax.plot(x_fine, y_vec, z_mac, color=COLOR_MAC, linestyle=":",
                linewidth=1.75, alpha=0.95, zorder=16)
        ax.scatter(x_markers, y_m, z_mac_m, color=COLOR_MAC, marker="s",
                   s=18, depthshade=False, zorder=17)

        # 4. Fourth Color: Bold Crimson Common Optimal Envelope (Max Curve)
        ax.plot(x_fine, y_vec, z_opt, color=COLOR_ENV, linestyle="-",
                linewidth=2.85, alpha=0.98, zorder=22)

        # Record peak and crossover points
        peak_x.append(0.0)
        peak_y.append(float(s_idx))
        peak_z.append(z_opt[0])

        cross_lh_x.append(x_lh)
        cross_lh_y.append(float(s_idx))
        cross_lh_z.append(z_opt[idx_lh])

        cross_hm_x.append(x_hm)
        cross_hm_y.append(float(s_idx))
        cross_hm_z.append(z_opt[idx_hm])

    # Reverse lists to order from scale_idx = 0..5
    peak_x, peak_y, peak_z = peak_x[::-1], peak_y[::-1], peak_z[::-1]
    cross_lh_x, cross_lh_y, cross_lh_z = cross_lh_x[::-1], cross_lh_y[::-1], cross_lh_z[::-1]
    cross_hm_x, cross_hm_y, cross_hm_z = cross_hm_x[::-1], cross_hm_y[::-1], cross_hm_z[::-1]

    # Peak envelope ridge across scales
    ax.plot(peak_x, peak_y, peak_z, color=COLOR_ENV, linestyle="--", linewidth=1.25, zorder=28)
    ax.scatter(peak_x, peak_y, peak_z, color=COLOR_ENV, edgecolors="#111111",
               s=30, depthshade=False, zorder=30)
    for px, py, pz in zip(peak_x, peak_y, peak_z):
        ax.text(
            px - 0.08, py, pz + 0.038,
            f"{pz:.2f}",
            fontsize=7.6, fontweight="bold", color=COLOR_ENV, ha="center", va="bottom",
            bbox=dict(boxstyle="round,pad=0.10", facecolor="white", edgecolor="none", alpha=0.78),
            zorder=35,
        )

    # Crossover trajectories (LUT -> Hybrid and Hybrid -> MAC)
    for tx, ty, tz in [(cross_lh_x, cross_lh_y, cross_lh_z), (cross_hm_x, cross_hm_y, cross_hm_z)]:
        ax.plot(tx, ty, tz, color="#111111", linestyle="--", linewidth=1.2, zorder=28)
        ax.scatter(tx, ty, tz, color="#111111", edgecolors="white", linewidth=0.6,
                   s=32, depthshade=False, zorder=31)
        for px, py, pz in zip(tx, ty, tz):
            ax.text(
                px + 0.05, py, pz + 0.034,
                f"{pz:.2f}",
                fontsize=7.4, fontweight="bold", color="#111111", ha="center", va="bottom",
                bbox=dict(boxstyle="round,pad=0.10", facecolor="white", edgecolor="none", alpha=0.78),
                zorder=35,
            )

    # Axis limits and labels
    ax.set_xlim(-0.20, 5.45)
    ax.set_ylim(-0.55, 5.35)
    ax.set_zlim(0.0, 1.08)

    ax.set_xticks(range(len(PREC_LABELS)))
    ax.set_xticklabels(PREC_LABELS, rotation=-14, ha="right", va="top")
    ax.set_yticks(range(len(SCALE_LABELS)))
    ax.set_yticklabels(SCALE_LABELS, ha="left", va="center")
    ax.set_zticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])

    ax.tick_params(axis="x", pad=2)
    ax.tick_params(axis="y", pad=3)
    ax.tick_params(axis="z", pad=4)

    ax.set_xlabel("Operand Precision $(W, A)$", labelpad=12, fontweight="bold")
    ax.set_ylabel("Array Scale $(R \\times C)$", labelpad=14, fontweight="bold")
    ax.set_zlabel("Norm. Efficiency $(1/\\mathrm{EDP})$", labelpad=8, fontweight="bold")

    ax.view_init(elev=25, azim=-52)

    # Legend with all 4 curves + crossover trajectory
    legend_handles = [
        Line2D([0], [0], color=COLOR_ENV, linestyle="-", linewidth=2.85,
               label="PRISM Optimal Envelope (Common Max Curve Across Scenarios)"),
        Line2D([0], [0], color=COLOR_LUT, linestyle="--", linewidth=1.6, marker="^", markersize=5.5,
               label="LUT Array Full Curve (High at Low Bitwidth, Drops at High Precision)"),
        Line2D([0], [0], color=COLOR_HYBRID, linestyle="-.", linewidth=1.6, marker="D", markersize=5.0,
               label="Hybrid MAC-LUT Full Curve (Peaks in W4A8/W4A16 Transition Zone)"),
        Line2D([0], [0], color=COLOR_MAC, linestyle=":", linewidth=1.8, marker="s", markersize=5.0,
               label="MAC Array Full Curve (Steady Across Precision, Wins at W8A8/W8A16)"),
        Line2D([0], [0], color="#111111", linestyle="--", linewidth=1.2, marker="o",
               markerfacecolor="#111111", markeredgecolor="white", markersize=5.5,
               label="Architecture Crossover Trajectory (LUT$\\rightarrow$Hybrid & Hybrid$\\rightarrow$MAC)"),
    ]

    ax.legend(
        handles=legend_handles,
        loc="upper left",
        bbox_to_anchor=(0.00, 1.08),
        frameon=True,
        facecolor="white",
        edgecolor="#CBD2D9",
        framealpha=0.95,
        fontsize=7.8,
    )

    fig.suptitle(
        "PRISM Joint Precision–Scale Design Space: Full Architecture Curves & Optimal Envelope",
        fontsize=11.0, fontweight="bold", color="#1B365D", y=0.97
    )

    plt.savefig(out_png, dpi=300, bbox_inches="tight", pad_inches=0.08)
    plt.savefig(out_pdf, bbox_inches="tight", pad_inches=0.05)
    plt.close()
    print(f"Successfully generated 3D tradeoff figure:\n  - {out_png}\n  - {out_pdf}")


if __name__ == "__main__":
    out_dir = pathlib.Path("./output")
    out_dir.mkdir(parents=True, exist_ok=True)

    generate_3d_tradeoff_figure(
        out_png=out_dir / "PRISM_Fig1_3D_Tradeoff.png",
        out_pdf=out_dir / "PRISM_Fig1_3D_Tradeoff.pdf",
    )
    
    