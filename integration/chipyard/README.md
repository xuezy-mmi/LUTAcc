# Chipyard/Rocket integration overlay

This overlay targets the locally available Chipyard commit
`b5d013190d637e634113cb5179f8c8885df1945a` and follows its SHA3 `LazyRoCC`
example. It intentionally leaves `/root/chipyard` untouched.

The RoCC interface uses `custom3`:

| funct | Meaning | rs1 | rs2 | response |
|---:|---|---|---|---|
| 0 | configure | `m[31:0], k[63:32]` | `n[31:0], A_bits-2[33:32], outlier_bp[49:34]` | 0 |
| 1 | launch OASIS | ignored | ignored | modeled cycles |
| 2 | launch Omni-LUT AW-GEMM | ignored | ignored | modeled cycles |
| 3 | read status | ignored | ignored | bit 0 busy |

`LUTAccTimingBlackBox.sv` is a transaction-level bridge for full Rocket/Verilator
software simulation. It implements the same integer-cycle equations as the
Python OASIS GEMM and inferred Omni AW-GEMM simulators, including OASIS
Orizuru/compensation and DRAM bounds. It does not replace the datapath RTL under
`rtl/`. This keeps
SoC boot/software tests practical while unit-level Verilator tests exercise the
real BQU/LGU/LUT-PE and OASIS primitives.

To add the overlay to a disposable Chipyard tree:

1. Copy `src/main` to `generators/lutacc/src/main`.
2. Add the `lutacc` project from `build.sbt.fragment` to Chipyard `build.sbt`.
3. Make the `chipyard` project depend on `lutacc`.
4. Copy `config-src/chipyard/config/LUTAccRocketConfig.scala` into
   `generators/chipyard/src/main/scala/config/`.
5. Build from `sims/verilator` with `CONFIG=LUTAccRocketConfig`.

The standalone software smoke is in `software/lutacc_rocc_smoke.c`.
