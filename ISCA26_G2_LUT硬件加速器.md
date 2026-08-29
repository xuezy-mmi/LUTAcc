# G2：OASIS / Omni-LUT 软硬件协同加速器复现

## 1. 目标与复现边界

本工程以源码级方式重建两套 ISCA 2026 LUT 加速器：

- **OASIS**：非均匀 W4A4/W4A3 双侧量化、Cartesian Product LUT-GEMM、
  look-ahead 主分支、Orizuru 动态 outlier 检测和残差误差补偿。
- **Omni-LUT**：AW-GEMM 与 AA-GEMM 共用 LUT 执行、Key/Value KV-cache
  量化、AS-Bit、BQU、带量化补偿的 LGU、32×4 LUT-PE 阵列，以及
  Prefill-WS / Decode-OS-V 相位自适应数据流。

最终验收目标是：核心机制具有软件数值参考、编译 IR、周期级事件模拟和可综合
RTL；论文公开数值逐点相对误差不超过 10%。闭源模拟器与商业 EDA 结果只能用
公开实现重建，不能声称取得作者源代码或复现绝对流片 PPA。

所有数字强制标记为 `reported`、`digitized`、`measured`、`calibrated`、
`inferred` 或 `predicted`。论文 target 只位于 `artifacts/targets/`，模拟器不读取
结果柱高作为运行时输入。

## 2. 证据来源

- OASIS：arXiv:2507.23035v4（CC BY 4.0）及 DOI
  `10.1109/ISCA66397.2026.00036`。公开 TeX 恢复了公式、硬件表、周期图、
  PPL/zero-shot 表和性能图。
- Omni-LUT：IEEE document `11617798`、DOI
  `10.1109/ISCA66397.2026.00037` 的公开摘要/预览。公开证据支持 AW+AA
  LUT-GEMM、offline+online KV 量化、LUT 生成补偿、混合 stationary 数据流和
  1.25×–1.91× 能效范围；未公开参数继续标为推测。
- FIGLUT：arXiv:2503.06862，提供 group-size=4、bit-serial、RAC/LUT 结构和
  0.14 TOPS / 0.29 W 的 28 nm 原始引擎证据。
- NVIDIA A100 80GB SXM datasheet 与 QuaRot arXiv:2404.00456：提供 GPU
  roofline 上界和 W4A4 机制；有效带宽/活跃功耗仍显式标记为 headline 校准值，
  不是本机 4090 测量或作者 A100 run。
- `/root/chipyard` commit `b5d0131...`：RoCC、BlackBox 和 Verilator SoC
  接口参考。
- `cspool/MLX_dev` commit `21b359d...`：仅借鉴证据分层、事件 DAG 和完成审计
  组织方式；未复制 MLX 论文专用代码或常数。

详细登记见 `artifacts/targets/source_manifest.json` 和
`docs/evidence-policy.md`。

## 3. 共享软件栈

### 3.1 编译器 IR

`src/lutacc/compiler/` 定义可序列化、有类型、可验证 DAG：

- Tensor：shape、dtype、DRAM/SRAM/register/stream 存储与 layout；
- Operation：输入输出、依赖、资源和机制属性；
- OASIS lowering：DMA → activation clustering / Orizuru 并行分支 →
  concat → histogram → LUT reduction / residual GEMM → merge；
- Omni lowering：AS-Bit → BQU → QK-LGU → QK LUT array → softmax →
  PV-LGU → PV LUT array；
- `simulate-ir` 会反序列化并校验完整机制节点集合，再由 IR 中的 shape、precision、
  phase 直接驱动周期模型；篡改或过期 IR 不会只凭 shape 被静默接受；
- 编译期校验唯一 producer、未知 tensor、未知依赖和 DAG 环。

### 3.2 数值参考核

`src/lutacc/quantization.py` 与 `src/lutacc/functional/` 实现：

- 确定性 weighted scalar K-Means、排序 centroid 与 midpoint boundary；
- OASIS 权重全局 normalized codebook + per-output scale；
- activation offline codebook + per-token scale；
- Orizuru top/bottom-k、相同 FP16 值固定选择最低 channel index；
- 真实 concatenated-index histogram 和 128/256-entry Cartesian LUT 加权和；
- `lookahead + residual@Wq` 与传统动态 outlier 路径数值同一性；
- AS-Bit 以二阶矩敏感度和边际误差收益分配精确 2/3/4-bit 总预算；
- K per-channel、V per-token BQU；每个 code 的均值 residual 被 LGU 折入
  centroid product；
- QK 与 PV 都执行 index lookup accumulation，不用“反量化后 GEMM”冒充 LUT。

### 3.3 周期、流量与能量

`src/lutacc/sim/` 包含整数周期资源 DAG scheduler、OASIS、Omni-LUT、FIGLUT
基线和 decoder-layer 系统聚合：

- 显式资源互斥、依赖、开始/结束 cycle、busy cycle 和 trace；
- DRAM/SRAM read/write bytes、常规 FLOP/MAC 口径、energy、TOPS 与 TOPS/W；
- OASIS 主/outlier 分支流式重叠，merge 必须等待两个 completion token；
- Omni Prefill 使用 WS，Decode 使用 OS-V；未知 issue efficiency 放在带
  `inferred` 标记的配置文件中；
- FIGLUT 由 group=4、4 个 bit plane、半 LUT generation fill 和 RAC query
  公式生成，不读取 OASIS 目标速度。

## 4. OASIS 实现规格

### 4.1 论文报告硬件

28 nm、500 MHz、15.31 mm²、9.66 W；16 条 PE Line。每线包含 4096 个
Concat Unit、2 KiB weight-index buffer、32 个 16-input Index Counter、一个
dequantizer、一个 32-input FP16 MAC Tree 和 8 个 FP16 compensation MAC。
芯片级包含 64 KiB output buffer、16 KiB activation-index buffer、2 KiB LUT、
4 个 clustering unit 和 273 个 16-input Orizuru unit。

### 4.2 1×4096×4096 W4A4 / 1% 周期锚点

| 子过程 | 论文 cycle | 当前 cycle | 误差 |
|---|---:|---:|---:|
| activation quantization | 2048 | 2048 | 0% |
| concat | 256 | 256 | 0% |
| index distribution | 1024 | 1024 | 0% |
| reduction | 2304 | 2304 | 0% |
| Orizuru | 1344 | 1344 | 0% |
| error compensation | 1536 | 1536 | 0% |
| merge | 256 | 256 | 0% |
| critical total | 2560 | 2560 | 0% |

公式来自结构参数：output wave、4096 Concat、每周期 histogram issue、
32-input reduction、每线 8 个 compensation MAC 和双分支 `max()`，没有直接
返回常数 2560。

### 4.3 流量与 headline 验证

1×4096×4096 的 SRAM traffic 当前为：weight-index 76.13%、LUT 19.03%、
activation-index 4.76%、output 0.07%；论文分别为 76.0%、19.2%、4.8%、
0.1%，最大绝对差 0.17 个百分点。

同一锚点的七项 stage energy 均写入 trace：activation quantization、index
distribution、compensation、concat、reduction、Orizuru、merge 的最大绝对误差为
0.04 个百分点。该项属于论文比例校准后的 stage model，而非独立 post-layout
power sign-off。

LLaMA-2-7B/13B、LLaMA-30B、LLaMA-2-70B 的统一 decoder 线性层模型得到：

| 指标 | 论文 | 当前 | 相对误差 |
|---|---:|---:|---:|
| A4 speedup vs FIGLUT | 3.00× | 3.130× | 4.35% |
| A3 speedup vs FIGLUT | 3.15× | 3.130× | 0.62% |
| A4 energy efficiency vs FIGLUT | 1.44× | 1.439× | 0.08% |
| A3 energy efficiency vs FIGLUT | 1.51× | 1.439× | 4.72% |
| A4 speedup vs A100 | 5.41× | 5.561× | 2.79% |
| A3 speedup vs A100 | 5.67× | 5.561× | 1.92% |
| A4 energy efficiency vs A100 | 198.10× | 202.640× | 2.29% |
| A3 energy efficiency vs A100 | 206.53× | 202.640× | 1.88% |
| A4 speedup vs QuaRot | 3.12× | 3.200× | 2.56% |
| A3 speedup vs QuaRot | 3.27× | 3.200× | 2.14% |
| A4 energy efficiency vs QuaRot | 108.80× | 111.301× | 2.30% |
| A3 energy efficiency vs QuaRot | 113.56× | 111.301× | 1.99% |

FIGLUT 行是 `predicted_vs_inferred`；A100/QuaRot 行是
`predicted_vs_calibrated_roofline_not_native`。全部达到 10% 数值门槛，但后者的
有效 HBM 带宽与功耗用公开 headline 校准，不能替代作者闭源模拟器或原生 A100
测量。

同一事件模型关闭 look-ahead 后，Orizuru 必须先完成，主分支与补偿分支才能启动。
LLaMA-2-7B 的 A4/A3 throughput gain 分别为 1.185×/1.109×，相对论文
1.16×/1.18× 的误差为 2.12%/6.04%。

## 5. Omni-LUT 实现规格

- `quantize_omni_kv` 同时输出 K/V variable-bit index、scale、bit allocation 和
  compensation table；
- BQU/LGU 的 NumPy/PyTorch 数值模型与 SystemVerilog primitive 均已实现；
- `omni_array.sv` 默认严格为 32×4=128 LUT PE；
- AW-GEMM 由 activation×W4-centroid 的 LGU 表、真实 weight-index gather 与
  per-output scale 实现，软件参考、编译 IR 和周期路径均已闭合；
- controller 在 prefill 拉高 `weight_stationary`，decode 拉高
  `value_output_stationary`；
- context=128/512/2048/8192/16384 的 prefill/decode 周期、KV bytes、流量和
  energy 可由 `scripts/reproduce.py` 生成；
- `omni_system_sweep.json` 聚合每层 QKV/O/MLP 四个 AW-GEMM 与 QK/softmax/PV；
  packed KV 超过 2 MiB SRAM 后，decode 显式切换为 DRAM tile stream；
- 同一 calibration tensor 上，LUT compensation 的量化/attention RMSE 不劣于
  未补偿版本；当前小型真实计算的 KV capacity reduction 为 2.83×（包含 scale
  和 correction metadata）。

基于相同 W4 weight traffic、FIGLUT FP16 AA-GEMM/FP16 KV 与 Omni LUT
AA-GEMM/3-bit KV 的独立 28 nm 能量系数，context=2048–8192 得到
1.276×–1.912×，相对论文摘要 1.25×–1.91× 的端点误差为 2.09%/0.09%。该
结果是 `predicted_from_inferred_28nm_coefficients`；逐 workload 论文柱高和完整
硬件表尚未公开，不能把这个范围拟合称为作者模拟器复现。

## 6. RTL 与系统集成

### 6.1 SystemVerilog

`rtl/oasis/`：Concat、并行 one-hot histogram、midpoint clustering、确定性
Orizuru tree、32-input fixed-point MAC topology、8-lane compensation MAC、
4096-Concat PE Line 和 branch-join controller。

`rtl/omni/`：variable 2/3/4-bit BQU、compensated LGU、可加载 LUT PE、
32×4 array 和 phase controller。FP16 算术位置以可综合定点代理实现；ASIC PPA
时需替换为目标工艺 FP16 macro，不能把代理面积称为论文 PPA。

Verilator transaction tests 已验证聚类、histogram、Orizuru tie、BQU clamp、
LGU compensation 和 LUT accumulation。`./scripts/lint_rtl.sh` 与
`./scripts/run_rtl_smoke.sh` 均通过。

HPE CACTI 7 以公开可用的最近工艺节点 32 nm 对多端口 SRAM 建模：OASIS 四类
buffer 合计 3.479 mm²，论文 28 nm memory-compiler 合计 3.858 mm²，聚合误差
9.82%。单项差异仍保留在 `cacti_memory.json`，不会用聚合通过掩盖工具/端口模型
差异。DRAMPower 6.0.0 的 bundled LPDDR5 x8 流式代理得到 read=54.35、
write=48.28 pJ/B；该值用于交叉检查，不冒充论文 HBM/LPDDR5 的未公开配置。

### 6.2 Chipyard

`integration/chipyard/` 提供与本地 Chipyard 版本兼容的 `LazyRoCC custom3`
overlay、软件 intrinsic、Rocket config 和 transaction timing BlackBox。BlackBox
用于完整 Rocket/Verilator 的可承受时间仿真；真实 datapath 由 `rtl/` 单独验证。
overlay 已通过本地 Chipyard commit `b5d0131...` 的原生 SBT/RocketChip 编译，
custom3 软件通过 RISC-V GCC，timing bridge 通过 Verilator。完整 SoC
config elaboration/boot 尚未执行，仍是验收缺口。

## 7. 真实模型质量路径

`scripts/bootstrap_ml.sh` 建立隔离 CUDA 环境（PyTorch 2.4.1+cu121、
Transformers 4.44.2、Datasets 2.21.0）。`scripts/evaluate_ppl.py` 支持任意公开
HF causal LM：

1. 每个 Linear 分块执行 W4 K-Means 与原位 dequantized numerical replay；
2. C4 或 WikiText calibration 捕获每层输入并学习 A3/A4 codebook；
3. 推理 hook 每 token 动态选择 top/bottom 0.5%，仅 inlier 量化；
4. 在同一 WikiText-2 token stream 上计算 PPL。

已完成的非论文 smoke：OPT-125M、sequence=512、1022 tokens，FP16 PPL 35.74，
OASIS-A4 PPL 49.18；该点只证明端到端链条，明确排除在论文验收之外。

OPT-6.7B 已在 RTX 4090 上完成全部 WikiText-2（139 chunks、284,533
predicted tokens、sequence=2048）：

| 模式 | 论文 PPL | 当前 PPL | 相对误差 |
|---|---:|---:|---:|
| FP16 | 10.86 | 10.757 | 0.94% |
| OASIS-A4 | 11.62 | 12.713 | 9.41% |
| OASIS-A3 | 14.12 | 13.282 | 5.93% |

三点均通过 10% 门槛；A4/A3 实际量化 192 个 Linear、学习 192 个 activation
codebook，并使用 16 条 C4 样本与 1% 动态 outlier。

Mistral-7B-v0.1 也已完成全部 WikiText-2（161 chunks、329,567 tokens）：

| 模式 | 论文 PPL | 当前 PPL | 相对误差 |
|---|---:|---:|---:|
| FP16 | 5.25 | 5.305 | 1.06% |
| OASIS-A4 | 5.75 | 6.115 | 6.34% |
| OASIS-A3 | 7.27 | 6.951 | 4.39% |

Mistral 消融发现共享偶数 K-Means codebook 需要保留一个精确零 centroid：不保留
时 W4-only PPL=8.98，保留后为 5.88。该约束不改变 16-entry LUT、index bit-width
或 per-output scale；它只在 Mistral 实验配置中显式启用，OPT 保持纯 K-Means，
以免模型特定选择被伪装成统一默认机制。

LLaMA-2-7B 使用固定 revision 的无门槛 base-weight mirror，完成全部 WikiText-2
（165 chunks、337,755 tokens）：

| 模式 | 论文 PPL | 当前 PPL | 相对误差 |
|---|---:|---:|---:|
| FP16 | 5.47 | 5.495 | 0.45% |
| OASIS-A4 | 5.90 | 6.339 | 7.45% |
| OASIS-A3 | 7.49 | 7.310 | 2.40% |

A4 对固定零 centroid 额外执行 8 次受约束 Lloyd refit 与一次 per-output scale
最小二乘更新；单元测试保证 refit 维持精确零且不增加 weight RMSE。没有零约束的
完整 A4 run 出现 NaN，因此不会被选为正式结果。

LLaMA-3-8B 使用固定 revision 的无门槛 base-weight mirror，完成全部 WikiText-2
（141 chunks、288,627 tokens）：

| 模式 | 论文 PPL | 当前 PPL | 相对误差 |
|---|---:|---:|---:|
| FP16 | 6.14 | 6.082 | 0.94% |
| OASIS-A4 | 7.11 | 7.535 | 5.97% |
| OASIS-A3 | 8.18 | 8.372 | 2.34% |

因此四个论文选定模型的 FP16/A4/A3 共 12 个完整 PPL 点全部通过，最大误差仍为
OPT-6.7B A4 的 9.41%。

Mistral 的六项 zero-shot 也已在 lm-evaluation-harness 0.4.5 上完成全数据集：

| 模式 | 论文平均准确率 | 当前平均准确率 | 相对误差 | 最大单项误差 |
|---|---:|---:|---:|---:|
| FP16 | 74.99% | 75.70% | 0.95% | 8.15% |
| OASIS-A4 | 73.80% | 73.17% | 0.86% | 5.93% |
| OASIS-A3 | 71.28% | 71.07% | 0.29% | 7.03% |

三种模式的 PIQA、ARC-E、ARC-C、BoolQ、HellaSwag、WinoGrande 及平均值共
21 个检查均低于 10%；量化模式均实际处理 224 个 Linear、学习 224 个
activation codebook，并使用同一份带 SHA256 的 16-sample C4 calibration 文件。

## 8. 命令

```bash
./scripts/bootstrap.sh
.venv/bin/lutacc compile --design oasis --m 1 --k 4096 --n 4096 \
  --output artifacts/generated/oasis_ir.json
.venv/bin/lutacc simulate --design oasis --m 1 --k 4096 --n 4096 \
  --summary --trace artifacts/generated/oasis_trace.json
make test
make reproduce
make audit

./scripts/bootstrap_ml.sh
CUDA_VISIBLE_DEVICES=0 .venv-ml/bin/python scripts/evaluate_ppl.py \
  --model facebook/opt-6.7b --mode oasis-a4 --sequence-length 2048 \
  --output artifacts/results/quality/opt6.7b_oasis_a4.json
```

## 9. 当前验收状态

快速测试：45/45 通过，独立 ML 测试 5/5 通过（包含两套 RTL transaction
simulation、SRAM lifetime/KV tiling 和 Fisher 回归）。
`scripts/audit_reproduction.py` 的所有**已有证据**检查通过，但 strict completion
仍为 false。未闭合项：

- A100/QuaRot 原生性能基线；
- Omni-LUT 1.25×–1.91× 的逐 workload 原论文柱高（当前仅摘要端点通过）；
- 标准单元开源 P&R（Yosys generic-gate、CACTI 32 nm 与 DRAMPower 交叉验证
  均已完成）；
- Chipyard 全 SoC elaboration、bare-metal boot 和 SoC 内 RoCC 返回值验证
  （独立 timing bridge 与原生 SBT 编译已通过）。

只有上述项目全部有权威证据且 strict audit 通过后，才可宣布本目标完成。
