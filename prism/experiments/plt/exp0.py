import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

# 1. 设置画布与全局排版风格（适配学术论文）
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False
fig = plt.figure(figsize=(11, 8), dpi=300)
ax = fig.add_subplot(111, projection='3d')

# 2. 生成设计空间网格
# X: 数据位宽 (Activation / Weight Precision, bits)
# Y: 阵列规模 (Total Processing Elements, PE Count)
P = np.linspace(2, 16, 60)         # 2-bit 到 16-bit
S = np.linspace(16, 1024, 60)      # 16 到 1024 个 PE
P_grid, S_grid = np.meshgrid(P, S)

# 3. 基于物理建模构建 EDP 函数 (归一化模型)
# GEMM 计算总量恒定为 Workload = 1000
workload = 1000.0

# --- MAC 阵列模型 ---
# 延迟随 PE 并行度下降，存在流水线及访存排空底噪
lat_mac = (workload / S_grid) + 3.0
# 能耗：乘法器随精度呈超线性增长，走线开销随阵列尺寸开根号扩展
energy_mac = (0.45 * (P_grid ** 1.65) + 0.18 * np.sqrt(S_grid)) * 8.0
edp_mac = (lat_mac * energy_mac) / 1000.0

# --- LUT 阵列模型 ---
# 延迟：在低位宽下单周期完成，高位宽下因分位分解 (Digit Decomposition) 增加多周期累加
lat_lut = (workload / S_grid) * (1.0 + 0.07 * np.maximum(0, P_grid - 4)) + 3.0
# 能耗：极低位宽下无乘法逻辑极省电；但随位宽增加查表指数扩大，且大规模阵列面临副本复制与广播走线惩罚
energy_lut = (1.2 * (1.33 ** P_grid) + 0.004 * (S_grid ** 1.28) + 0.7 * P_grid) * 5.0
edp_lut = (lat_lut * energy_lut) / 1000.0

# 4. 绘制两组三维曲面
surf_mac = ax.plot_surface(
    P_grid, S_grid, edp_mac, 
    cmap=cm.Blues, alpha=0.55, edgecolor='none', 
    antialiased=True, label='MAC Array'
)
surf_lut = ax.plot_surface(
    P_grid, S_grid, edp_lut, 
    cmap=cm.Oranges, alpha=0.55, edgecolor='none', 
    antialiased=True, label='LUT Array'
)

# 5. 提取并绘制 Cost Crossover 交线
diff = edp_lut - edp_mac
# 在底面投影交线（决策相图分界线）
z_floor = 0.0
contour_proj = ax.contour(
    P_grid, S_grid, diff, levels=[0], 
    colors='red', linewidths=2.5, linestyles='solid', offset=z_floor
)

# 提取交线的三维空间坐标并绘制
import matplotlib.patheffects as pe
try:
    paths = contour_proj.collections[0].get_paths()
    for path in paths:
        v = path.vertices
        x_cross = v[:, 0]
        y_cross = v[:, 1]
        # 插值获取交线对应的 Z (EDP) 值
        z_cross = (workload / y_cross + 3.0) * (0.45 * (x_cross ** 1.65) + 0.18 * np.sqrt(y_cross)) * 8.0 / 1000.0
        ax.plot(
            x_cross, y_cross, z_cross, 
            color='darkred', linewidth=3.0, linestyle='--',
            path_effects=[pe.Stroke(linewidth=4, foreground='white'), pe.Normal()],
            label='Cost Crossover'
        )
except Exception:
    pass

# 6. 图表标注与视觉修饰
ax.set_xlabel('Precision (Bitwidth, bits)', fontsize=12, labelpad=12, fontweight='bold')
ax.set_ylabel('Array Scale (PE Count)', fontsize=12, labelpad=12, fontweight='bold')
ax.set_zlabel('Normalized EDP ($pJ \cdot s$)', fontsize=12, labelpad=10, fontweight='bold')

ax.set_zlim(z_floor, 35)
ax.view_init(elev=24, azim=-125)  # 调整视角以最佳展示交叉曲面

# 区域文字注解
ax.text(3, 100, 3, "LUT-Optimal Region\n(Low Precision, Moderate Scale)", 
        color='saddlebrown', fontsize=10, fontweight='bold')
ax.text(12, 600, 20, "MAC-Optimal Region\n(High Precision, Large Scale)", 
        color='navy', fontsize=10, fontweight='bold')
ax.text(7, 300, z_floor + 0.5, "Projected Pareto Boundary", 
        color='red', fontsize=9, fontweight='bold')

# 伪造图例项以支持 plot_surface
mac_proxy = plt.Rectangle((0, 0), 1, 1, fc=cm.Blues(0.6), alpha=0.7)
lut_proxy = plt.Rectangle((0, 0), 1, 1, fc=cm.Oranges(0.6), alpha=0.7)
line_proxy = plt.Line2D([0], [0], color='darkred', lw=2.5, linestyle='--')
bound_proxy = plt.Line2D([0], [0], color='red', lw=2.0, linestyle='-')

ax.legend(
    [mac_proxy, lut_proxy, line_proxy, bound_proxy],
    ['MAC Array Surface', 'LUT Array Surface', '3D Crossover Trajectory', 'Projected Pareto Boundary'],
    loc='upper left', bbox_to_anchor=(0.02, 0.95), frameon=True, framealpha=0.9
)

plt.tight_layout()
plt.savefig("prism_fig1_3d_crossover.pdf", dpi=300, bbox_inches='tight')
# plt.show()