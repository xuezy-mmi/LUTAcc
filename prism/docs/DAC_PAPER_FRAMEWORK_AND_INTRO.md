# PRISM: Precision- and Reuse-Aware Joint Design of MAC-LUT Arrays for Mixed-Precision LLM Inference

## 1. Paper positioning

This paper should be positioned as a design automation paper for ASIC inference accelerators. Its central claim is not that LUTs universally replace MACs. The claim is that arithmetic primitive, array scale, table organization, and dataflow must be selected jointly because their relative costs change with precision, GEMM shape, reuse, and resource constraints.

The DAC audience should see PRISM as a reusable compiler-to-hardware flow that turns workload metadata into a constrained, synthesizable accelerator specialization. The paper should present new design-space structure and a practical generator, rather than only a collection of hand-designed MAC and LUT blocks.

Recommended title:

> **PRISM: Precision- and Reuse-Aware Joint Design of MAC-LUT Arrays for Mixed-Precision LLM Inference**

Recommended one-sentence thesis:

> PRISM shows that the optimal arithmetic primitive for a mixed-precision GEMM emerges from the interaction among precision, weight reuse, array scale, table sharing, and dataflow, and automates this choice from workload specification to synthesizable RTL.

## 2. Six-page organization excluding references

The page budget assumes a two-column DAC format and includes figures and tables.

| Section | Target space | Content | Main artifact |
|---|---:|---|---|
| Abstract | 0.20 p | Problem, method, result, impact | One compact paragraph |
| 1. Introduction | 0.85 p | Why low precision is insufficient; problem; insight; contributions | Figure 1: motivating cost crossover |
| 2. Background and Motivation | 0.65 p | Mixed precision, static-weight LUT computation, array/dataflow effects | Figure 2: MAC/LUT organization |
| 3. PRISM Overview | 0.45 p | End-to-end flow and input/output contract | Figure 3: framework overview |
| 4. Precision- and Reuse-Aware Cost Model | 1.05 p | Equations, LUT decomposition, memory/interconnect, calibration | Table 1: model terms; Figure 4: phase diagram |
| 5. Joint Design-Space Exploration | 0.90 p | Search variables, constraints, Pareto selection, hybrid tiles | Algorithm 1; Table 2: search space |
| 6. RTL Generation and Implementation | 0.55 p | Generator, ROM/table handling, mapping/dataflow metadata, verification | Figure 5: generated hardware |
| 7. Evaluation | 1.15 p | Setup, accuracy of model, comparisons, ablations, case studies | Figures 6–8; Tables 3–4 |
| 8. Related Work | 0.35 p | LUT accelerators, mixed-precision accelerators, DSE/generation | Compact taxonomy table |
| 9. Conclusion | 0.20 p | Findings and scope | One paragraph |
| **Total** | **6.35 p** | Reduce Introduction/Background by 0.35 p during typesetting |  |

The practical six-page target is achieved by treating the abstract as part of the first page, merging Related Work into the end of Background if necessary, and moving implementation details such as complete RTL interfaces to an artifact appendix or supplementary material.

## 3. Proposed paper outline

### 1. Introduction

Introduce the mismatch between bitwidth-centric accelerator design and mixed-precision LLM inference. State that W4A8 and W4A16 change multiplication, indexing, accumulation, conversion, memory, and reuse costs together. Motivate LUTs for static weights, then explain why LUT storage, replication, access, and routing can erase their arithmetic benefit. Establish array size and dataflow as first-class design variables.

End the section with the research question:

> Given a GEMM shape, operand precisions, static-weight assumptions, and hardware budgets, which arithmetic primitive, array organization, and dataflow minimize the target objective while remaining synthesizable?

### 2. Background and Motivation

Define the GEMM and precision notation. Explain digit-decomposed LUT computation:

\[
L_g[x_g] = \sum_{i=0}^{G-1} a_{g,i} w_{g,i}, \qquad
y_j = \sum_g L_{j,g}[x_g].
\]

Use one example to show why W4A16 needs activation digit decomposition instead of a single \(2^{16}\)-entry table. Explain three table-sharing policies: global, per-tile, and per-PE. Show that increasing array size reduces compute cycles but increases replication, bandwidth, reduction, and routing pressure.

### 3. PRISM Overview

Present the flow:

```text
GEMM shape + precision + budgets
              |
              v
  precision/reuse feature extraction
              |
              v
  calibrated MAC/LUT cost model
              |
              v
  constrained Pareto DSE
              |
              v
  table, memory, and RTL generation
              |
              v
  simulation and synthesis feedback
```

Define the input contract and output contract. The input includes \((M,N,K)\), weight/input/output/accumulator formats, static-weight policy, target budgets, and candidate architectural options. The output includes primitive type, array dimensions, `tile_k`, LUT group and digit parameters, sharing policy, dataflow, generated ROM metadata, RTL, and a cost report.

### 4. Precision- and Reuse-Aware Cost Model

Separate the model into compute, storage, movement, and execution terms:

\[
A = A_{compute}+A_{table}+A_{buffer}+A_{network},
\]

\[
E = E_{arithmetic}+E_{lookup}+E_{storage}+E_{movement},
\]

\[
T = T_{tiles}\cdot T_K\cdot T_{primitive}+T_{reduction}.
\]

For LUT mode, expose:

\[
D=\lceil p_a/d\rceil,\quad
Q=\lceil K/G\rceil D,\quad
S=2^{Gd},
\]

where \(p_a\) is activation precision, \(d\) is digit width, \(G\) is group size, and \(S\) is entries per table. Include table output width, replication factor, and table-sharing policy in area and energy. Explain that technology coefficients are fitted from synthesized microblocks and memory macros.

The central analysis should be a phase diagram over activation precision, reuse, and array size. Highlight where LUTs win, where MACs win, and where a hybrid array is Pareto-optimal.

### 5. Joint Design-Space Exploration

Define the candidate tuple:

\[
c=(p, R, C, K_t, G, d, s, f, q),
\]

where \(p\) is primitive, \(R\times C\) is array size, \(K_t\) is reduction tile, \(G,d,s\) describe LUT organization, \(f\) is LUT fraction, and \(q\) is dataflow.

The search applies hard constraints for area, energy, latency, LUT capacity, and utilization. It returns a Pareto set over area, energy, and latency and selects the lowest EDP point under the user objective. The initial implementation is deterministic exhaustive enumeration; the paper can describe pruning and leave Bayesian/NSGA-II search as a scalable backend.

The hybrid design should assign MAC and LUT work at tile or output-group granularity. If the current implementation only supports a global fraction, present heterogeneous assignment as the planned generator extension and label the implemented fraction model precisely.

### 6. RTL Generation and Implementation

Describe the generated parameterized compute tile, table initialization, reduction path, dataflow metadata, and manifest. Explain that static quantized weights are required to populate LUT contents. Report functional equivalence against a Python fixed-point reference and synthesis with Yosys plus the selected target technology flow. Keep the generated zero-filled ROM template clearly separate from a functional model-specific ROM.

### 7. Evaluation

Use projection GEMMs from Llama 3 8B/70B, Qwen 2.5 7B, and Mixtral 8x7B. Include decode \(M=1\) and prefill \(M=2048\), with QKV, attention-output, FFN gate/up, and FFN-down shapes. Evaluate W4A8, W4A16, W8A8, and W8A16, using explicit accumulator precision.

Answer four questions:

1. **Model accuracy:** How closely do analytical area, energy, and latency estimates match RTL synthesis and simulation?
2. **Primitive choice:** When do MAC, LUT, and hybrid designs win, and how does the boundary move with array scale and reuse?
3. **Joint optimization:** How much is lost when primitive, array size, or dataflow is optimized independently?
4. **Automation value:** Can PRISM generate feasible designs with better Pareto points than fixed MAC or fixed LUT baselines?

Required baselines are full-MAC, full-LUT with each sharing policy, fixed-array primitive selection, fixed-primitive array search, and PRISM joint search. Report area, frequency, latency, energy/GEMM, EDP, utilization, table capacity, and prediction error.

### 8. Related Work

Organize related work by design decision rather than publication chronology:

1. LUT and table-based neural-network accelerators;
2. mixed-precision and quantized LLM accelerators;
3. array/dataflow accelerators;
4. hardware generators and accelerator DSE systems.

The distinction to emphasize is that PRISM jointly explores arithmetic primitive, array scale, table organization, and dataflow under one cost model and produces synthesizable artifacts.

### 9. Conclusion

Restate the design lesson: precision is a necessary input but not a sufficient decision rule. The best implementation depends on reuse and organization. Close with the scope limitation that the first version targets static-weight ASIC inference and can be extended to dynamic weights and compiler scheduling.

## 4. Recommended figures and tables

| ID | Figure/table | Purpose |
|---|---|---|
| Fig. 1 | MAC/LUT cost crossover | Establish the problem with W4A8/W4A16 and array-size curves |
| Fig. 2 | LUT group/digit decomposition | Explain why high activation precision changes LUT cost |
| Fig. 3 | PRISM pipeline | Show metadata-to-RTL automation |
| Fig. 4 | Phase diagram | Show primitive regions over reuse and array scale |
| Fig. 5 | Generated heterogeneous tile | Connect DSE parameters to hardware organization |
| Fig. 6 | Model prediction error | Establish cost-model credibility |
| Fig. 7 | Pareto frontiers | Compare fixed and joint optimization |
| Fig. 8 | LLM workload heatmap | Show choices across decode/prefill operators |
| Table 1 | Cost-model terms | Make assumptions auditable |
| Table 2 | Search-space dimensions | Establish reproducibility |
| Table 3 | Main PPA comparison | State the primary result |
| Table 4 | Ablation study | Isolate reuse, array, dataflow, and hybrid effects |

## 5. Introduction draft

### 1. Introduction

Large language model (LLM) inference is increasingly dominated by matrix multiplication, while the demand for higher throughput and lower energy continues to grow. Quantization has become a central technique for meeting this demand: weight-only and mixed-precision formats such as W4A8 and W4A16 substantially reduce storage and arithmetic bandwidth compared with conventional FP16 execution. This trend has encouraged accelerator designs that specialize aggressively to operand bitwidths.

However, bitwidth alone does not determine the cost of a specialized GEMM engine. A low-precision multiplication can be implemented with a conventional multiply-accumulate (MAC), a precomputed lookup table (LUT), or a combination of both. The relative cost of these choices depends on the reduction dimension, weight reuse, activation decomposition, accumulation precision, table organization, and the size of the compute array. For example, a static W4 weight group can replace repeated multiplications with a table of precomputed partial sums. The same transformation becomes less attractive when activation precision requires multiple digits, when tables must be replicated to provide sufficient bandwidth, or when a large array increases routing and storage overhead. Consequently, a design that is optimal for a small decode GEMM can be suboptimal for a large prefill GEMM even when both use the same quantization format.

This interaction creates a gap in current accelerator design practice. Existing designs commonly commit to a fixed arithmetic primitive, such as a MAC array or a LUT-based datapath, and then optimize its microarchitecture. Other approaches explore array dimensions or dataflow while treating the compute primitive as fixed. These decompositions simplify implementation, but they can miss cross-dimensional tradeoffs. In particular, they do not expose when a larger array reduces execution time enough to justify additional LUT replication, or when a shared LUT reduces area while increasing access and movement energy. Mixed-precision inference makes the problem more difficult because W4A8 and W4A16 have different lookup decomposition, accumulator, conversion, and bandwidth requirements.

We present **PRISM**, a precision- and reuse-aware framework for the joint design of MAC-LUT arrays for mixed-precision LLM inference. PRISM takes a GEMM description, operand and accumulator precisions, and hardware resource constraints as input. It builds a calibrated cost model that accounts for arithmetic, table storage, table replication, data movement, array utilization, and reduction latency. A constrained design-space explorer then searches arithmetic primitive, array dimensions, reduction tile size, LUT group and digit parameters, table sharing, and dataflow together. Finally, PRISM emits a parameterized hardware implementation, LUT initialization metadata, and a machine-readable cost report for synthesis and verification.

The key insight behind PRISM is that the choice between MAC and LUT has scale-dependent phase transitions. Precision determines the size and number of elementary operations, but reuse determines how often a precomputed result amortizes table construction and storage. Array scale determines both parallel execution capacity and the replication or movement cost needed to feed that capacity. Dataflow determines which operands remain local and which operands traverse the array. These effects are coupled, so optimizing any one of them in isolation can select a configuration that is dominated after the remaining costs are included.

PRISM makes this interaction explicit and actionable. Its cost model provides a common representation for MAC, LUT, and hybrid tiles; its search procedure returns a Pareto set under area, energy, latency, storage, and utilization constraints; and its generator turns a selected point into synthesizable RTL and model-specific LUT artifacts. This separation also supports calibration: analytical coefficients can be fitted using technology-specific MAC, ROM/SRAM, and interconnect measurements without changing the DSE interface.

We make the following contributions:

* **A unified precision- and reuse-aware cost model** for MAC, LUT, and hybrid GEMM tiles. The model captures digit-decomposed lookup, accumulation width, table replication, memory movement, array utilization, and dataflow-dependent costs, and exposes the phase boundaries between arithmetic organizations.
* **A joint arithmetic-array design-space exploration method** that selects primitive type, array scale, LUT organization, reduction granularity, and dataflow under hard hardware constraints. It returns Pareto-optimal configurations and supports heterogeneous MAC-LUT specialization through a configurable tile assignment model.
* **An end-to-end hardware generation flow** that converts the selected configuration into parameterized RTL, LUT/ROM initialization artifacts, and reproducible cost reports. The flow is evaluated on representative decode and prefill GEMMs from modern LLMs across W4A8, W4A16, W8A8, and W8A16 settings.

Our evaluation is designed to answer whether joint exploration provides a measurable advantage over fixed-MAC, fixed-LUT, primitive-only, and array-only baselines. We also quantify cost-model error against synthesized implementations and use ablations to isolate the effects of precision decomposition, table sharing, array scale, and dataflow. The results establish when LUT specialization is beneficial, when conventional MACs remain preferable, and when a hybrid organization provides the best area-energy-latency tradeoff.

## 6. Claims to validate before submission

The Introduction should only retain claims supported by final experiments. In particular, the paper should report:

- synthesis-calibrated model error for area and timing, plus an energy methodology;
- at least one workload where MAC wins, one where LUT wins, and one where hybrid specialization is Pareto-optimal;
- a measurable gain from joint optimization over primitive-only and array-only search;
- generated RTL that passes functional differential tests for MAC and LUT paths;
- exact tool versions, technology assumptions, GEMM dimensions, batch/sequence settings, and table-sharing policy.

If these measurements are unavailable, soften “Pareto-optimal” to “model-selected” and present the work as a framework prototype until synthesis validation is complete.
