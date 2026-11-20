# WSL Hardware Toolchain

## Chisel status

Chisel is configured and verified in `chisel/`:

- Ubuntu 22.04.2 LTS (WSL)
- OpenJDK 11.0.31
- Scala 2.13.14
- sbt runner 1.12.0, project sbt 1.10.2
- Chisel 6.6.0
- Chisel compiler plugin 6.6.0

Verification:

```bash
cd /home/xzy/hwgen/prism/chisel
sbt -batch run
```

This generated `chisel/generated/SmokeTile.sv` successfully. The project
currently has no `chiseltest` dependency; add a Maven Central version after
selecting the simulator version for functional tests.

## Yosys status

Yosys is **not installed**. `command -v yosys` returned no executable. An
attempted Ubuntu package installation was blocked because the configured apt
repository did not respond in the WSL environment; no package was left in a
partially configured state.

When apt network access is available, install and verify with:

```bash
sudo apt-get update
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y yosys
yosys -V
```

Ubuntu 22.04's repository package is Yosys 0.9-2. For publication-quality
experiments, record the exact Yosys version and synthesis script. The generated
SystemVerilog may also require `read_verilog -sv` in Yosys.
