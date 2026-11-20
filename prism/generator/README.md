# PRISM Generator

`HardwareGenerator.generate(request, result, output_dir)` converts a DSE-selected configuration into a parameter-specialized SystemVerilog compute tile and metadata.

Generated files are:

- `<module>.sv`: synthesizable MAC, LUT, or hybrid compute tile;
- `prism_config.svh`: selected array, precision, primitive, and dataflow constants;
- `manifest.json`: input request and DSE result for provenance;
- `cost_report.json`: exact analytical PPA report used for selection;
- `lut_init.mem`: zero-filled ROM template for LUT/hybrid designs.

The generated tile uses unpacked array ports for activations, weights, lookup indices, table selection, and partial sums. An external controller supplies one reduction step per valid cycle and asserts `clear` at output-tile boundaries. Dataflow is carried into `DATAFLOW_ID`; the memory/controller generator is the next extension point.

For LUT and hybrid modes, weights are static but model-specific, so the generator cannot infer table contents from GEMM dimensions alone. Build offline tables from quantized weights, then replace `lut_init.mem`:

```python
from generator import build_lut_tables, write_lut_mem

tables = build_lut_tables([[3], [-2]], digit_bits=4)
write_lut_mem("lut_init.mem", tables, output_bits=8)
```

The default generated ROM file is deliberately all zeros and is suitable only for RTL elaboration, not functional inference.
