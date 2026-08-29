# Evidence policy

Every number compared with a paper target carries one of six labels:

| Label | Meaning |
|---|---|
| `reported` | Printed as a number in paper text/table |
| `digitized` | Recovered from vector/raster plot geometry |
| `measured` | Produced by a local run or RTL tool |
| `calibrated` | Fit using declared training anchors |
| `inferred` | Derived from prose, equations, or hardware structure |
| `predicted` | Extrapolated outside the calibration set |

Rules enforced by the audit scripts:

1. Paper targets live only in `artifacts/targets/`; simulator modules do not
   import that directory.
2. Calibrated constants live in `configs/calibration/` and identify their
   training anchors.
3. A headline mean cannot hide a failed point. Acceptance is point-wise and
   uses `abs(simulated-target)/abs(target) <= 0.10` unless the target declares a
   different uncertainty.
4. Functional equivalence, cycle agreement, and PPA/energy agreement are
   separate gates.
5. Missing proprietary EDA or unpublished source is reported as a limitation;
   it is never relabeled as a passing reproduction.

The event-simulator/evidence separation follows the useful organizational idea
in `cspool/MLX_dev` at commit `21b359d06c4196808a5a2dc40a692478dce99a25`.
No MLX paper-specific source or constants are copied. Chipyard commit
`b5d013190d637e634113cb5179f8c8885df1945a` is used as the local Verilator/SoC
harness reference.
