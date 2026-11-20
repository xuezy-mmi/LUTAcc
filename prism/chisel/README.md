# Chisel backend

This is the Chisel entry point for a future PRISM hardware backend. The current
`SmokeTile` is intentionally small and validates that sbt, Scala, Chisel, and
Verilog emission work in the WSL environment. The Python generator remains the
reference backend until its parameters are wired into this module.

From WSL:

```bash
cd /home/xzy/hwgen/prism/chisel
sbt run
```

Generated Verilog is placed under `chisel/generated/` and is ignored by the
research scripts. Pin the Chisel version here when comparing synthesis results.
