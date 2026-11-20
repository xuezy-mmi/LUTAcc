"""Generate synthesizable compute-tile RTL from a PRISM DSE result."""

from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path

from costmodel import Primitive
from dse import DseRequest, DseResult


@dataclass(frozen=True)
class GeneratedArtifacts:
    output_dir: Path
    rtl: Path
    config_header: Path
    manifest: Path
    cost_report: Path
    lut_init: Path | None


class HardwareGenerator:
    """Emit a parameter-specialized SystemVerilog compute tile and metadata."""

    def generate(
        self,
        request: DseRequest,
        result: DseResult,
        output_dir: str | Path,
        module_name: str = "prism_accelerator",
    ) -> GeneratedArtifacts:
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", module_name):
            raise ValueError("module_name must be a valid SystemVerilog identifier")
        target = Path(output_dir)
        target.mkdir(parents=True, exist_ok=True)
        config = result.selected
        lut = config.lut
        table_count = 0 if lut is None else math.ceil(request.shape.k / lut.group_size) * math.ceil(
            request.input_precision.bits / lut.digit_bits
        )
        entries = 0 if lut is None else 1 << (lut.group_size * min(lut.digit_bits, request.input_precision.bits))
        table_output_bits = 0 if lut is None else self._table_output_bits(request, lut.group_size, lut.digit_bits)

        rtl = target / f"{module_name}.sv"
        header = target / "prism_config.svh"
        manifest = target / "manifest.json"
        report = target / "cost_report.json"
        lut_init = target / "lut_init.mem" if lut is not None else None

        rtl.write_text(
            self._render_rtl(module_name, request, result, table_count, entries, table_output_bits),
            encoding="ascii",
        )
        header.write_text(self._render_header(request, result, table_count, entries), encoding="ascii")
        report.write_text(json.dumps(result.report.as_dict(), indent=2) + "\n", encoding="ascii")
        manifest.write_text(
            json.dumps(
                {
                    "request": self._request_dict(request),
                    "dse_result": result.as_dict(),
                    "rtl_module": module_name,
                    "rtl_file": rtl.name,
                    "lut_table_count": table_count,
                    "lut_entries_per_table": entries,
                    "lut_output_bits": table_output_bits,
                    "lut_init_file": None if lut_init is None else lut_init.name,
                },
                indent=2,
            )
            + "\n",
            encoding="ascii",
        )
        if lut_init is not None:
            lut_init.write_text("0\n" * (table_count * entries), encoding="ascii")
        (target / "README.md").write_text(self._render_readme(module_name, lut_init is not None), encoding="ascii")
        return GeneratedArtifacts(target, rtl, header, manifest, report, lut_init)

    @staticmethod
    def _table_output_bits(request: DseRequest, group_size: int, digit_bits: int) -> int:
        return request.weight_precision.bits + min(digit_bits, request.input_precision.bits) + math.ceil(
            math.log2(max(1, group_size))
        )

    def _render_header(self, request: DseRequest, result: DseResult, tables: int, entries: int) -> str:
        config = result.selected
        lut = config.lut
        return "\n".join(
            [
                "`ifndef PRISM_CONFIG_SVH",
                "`define PRISM_CONFIG_SVH",
                f"`define PRISM_ARRAY_ROWS {config.array.rows}",
                f"`define PRISM_ARRAY_COLS {config.array.cols}",
                f"`define PRISM_TILE_K {config.array.tile_k}",
                f"`define PRISM_WEIGHT_BITS {request.weight_precision.bits}",
                f"`define PRISM_ACTIVATION_BITS {request.input_precision.bits}",
                f"`define PRISM_ACCUMULATOR_BITS {config.accumulator_precision.bits}",
                f"`define PRISM_DATAFLOW_ID {self._dataflow_id(config.array.dataflow)}",
                f"`define PRISM_PRIMITIVE_ID {self._primitive_id(config.primitive)}",
                f"`define PRISM_LUT_GROUP_SIZE {0 if lut is None else lut.group_size}",
                f"`define PRISM_LUT_DIGIT_BITS {0 if lut is None else lut.digit_bits}",
                f"`define PRISM_LUT_TABLES {tables}",
                f"`define PRISM_LUT_ENTRIES {entries}",
                "`endif",
                "",
            ]
        )

    def _render_rtl(
        self, module: str, request: DseRequest, result: DseResult, tables: int, entries: int, output_bits: int
    ) -> str:
        config = result.selected
        common = self._parameters(module, request, result, tables, entries, output_bits)
        if config.primitive == Primitive.MAC:
            body = self._mac_body()
            ports = self._mac_ports()
        else:
            assert config.lut is not None
            lut_setup = self._lut_setup(config.lut.sharing, tables, entries, output_bits, request, config.array.rows, config.array.cols)
            body = self._lut_body(lut_setup, hybrid=config.primitive == Primitive.HYBRID, lut_fraction=config.lut_fraction)
            ports = self._lut_ports()
        return f"`include \"prism_config.svh\"\n\nmodule {module} #(\n{common}\n) (\n{ports}\n);\n\n{body}\nendmodule\n"

    @staticmethod
    def _parameters(
        module: str, request: DseRequest, result: DseResult, tables: int, entries: int, output_bits: int
    ) -> str:
        config = result.selected
        parameters = [
                f"  parameter integer ARRAY_ROWS = {config.array.rows}",
                f"  parameter integer ARRAY_COLS = {config.array.cols}",
                f"  parameter integer WEIGHT_BITS = {request.weight_precision.bits}",
                f"  parameter integer ACTIVATION_BITS = {request.input_precision.bits}",
                f"  parameter integer ACCUMULATOR_BITS = {config.accumulator_precision.bits}",
                f"  parameter integer TILE_K = {config.array.tile_k}",
                f"  parameter integer DATAFLOW_ID = {HardwareGenerator._dataflow_id(config.array.dataflow)}",
        ]
        if config.lut is not None:
            parameters.extend(
                [
                    f"  parameter integer LUT_INDEX_BITS = {math.ceil(math.log2(entries))}",
                    f"  parameter integer TABLE_SELECT_BITS = {max(1, math.ceil(math.log2(tables)))}",
                    f"  parameter integer LUT_TABLES = {tables}",
                    f"  parameter integer LUT_ENTRIES = {entries}",
                    f"  parameter integer LUT_OUTPUT_BITS = {output_bits}",
                ]
            )
        return ",\n".join(parameters)

    @staticmethod
    def _mac_ports() -> str:
        return "\n".join(
            [
                "  input logic clk,",
                "  input logic rst,",
                "  input logic clear,",
                "  input logic valid,",
                "  input logic signed [ACTIVATION_BITS-1:0] activation [0:ARRAY_ROWS-1],",
                "  input logic signed [WEIGHT_BITS-1:0] weight [0:ARRAY_ROWS-1][0:ARRAY_COLS-1],",
                "  output logic signed [ACCUMULATOR_BITS-1:0] psum [0:ARRAY_ROWS-1][0:ARRAY_COLS-1]",
            ]
        )

    @staticmethod
    def _lut_ports() -> str:
        return "\n".join(
            [
                "  input logic clk,",
                "  input logic rst,",
                "  input logic clear,",
                "  input logic valid,",
                "  input logic signed [ACTIVATION_BITS-1:0] activation [0:ARRAY_ROWS-1],",
                "  input logic signed [WEIGHT_BITS-1:0] weight [0:ARRAY_ROWS-1][0:ARRAY_COLS-1],",
                "  input logic [LUT_INDEX_BITS-1:0] lookup_index [0:ARRAY_ROWS-1][0:ARRAY_COLS-1],",
                "  input logic [TABLE_SELECT_BITS-1:0] table_select [0:ARRAY_ROWS-1][0:ARRAY_COLS-1],",
                "  output logic signed [ACCUMULATOR_BITS-1:0] psum [0:ARRAY_ROWS-1][0:ARRAY_COLS-1]",
            ]
        )

    @staticmethod
    def _mac_body() -> str:
        return """  integer row;
  integer col;
  always_ff @(posedge clk) begin
    if (rst || clear) begin
      for (row = 0; row < ARRAY_ROWS; row = row + 1)
        for (col = 0; col < ARRAY_COLS; col = col + 1)
          psum[row][col] <= '0;
    end else if (valid) begin
      for (row = 0; row < ARRAY_ROWS; row = row + 1)
        for (col = 0; col < ARRAY_COLS; col = col + 1)
          psum[row][col] <= psum[row][col] + activation[row] * weight[row][col];
    end
  end"""

    @staticmethod
    def _lut_setup(sharing: str, tables: int, entries: int, output_bits: int, request: DseRequest, rows: int, cols: int) -> str:
        copies = 1 if sharing == "global" else (math.ceil(rows / 4) * math.ceil(cols / 4) if sharing == "per_tile" else rows * cols)
        return "\n".join(
            [
                f"  localparam integer ROM_COPIES = {copies};",
                f"  localparam integer SHARING_MODE = {0 if sharing == 'global' else (1 if sharing == 'per_tile' else 2)};",
                "  logic signed [LUT_OUTPUT_BITS-1:0] lut_mem [0:ROM_COPIES-1][0:LUT_TABLES*LUT_ENTRIES-1];",
                "  integer init_copy;",
                "  initial begin",
                "    for (init_copy = 0; init_copy < ROM_COPIES; init_copy = init_copy + 1)",
                "      $readmemh(\"lut_init.mem\", lut_mem[init_copy]);",
                "  end",
                "  function automatic integer rom_index(input integer row, input integer col);",
                "    if (SHARING_MODE == 0) rom_index = 0;",
                "    else if (SHARING_MODE == 1) rom_index = (row / 4) * ((ARRAY_COLS + 3) / 4) + (col / 4);",
                "    else rom_index = row * ARRAY_COLS + col;",
                "  endfunction",
            ]
        )

    @staticmethod
    def _lut_body(setup: str, hybrid: bool, lut_fraction: float) -> str:
        mac_columns = max(1, min(1023, round((1.0 - lut_fraction) * 1024)))
        hybrid_line = ""
        if hybrid:
            hybrid_line = (
                f"  localparam integer MAC_COLUMN_THRESHOLD = (ARRAY_COLS * {mac_columns}) / 1024;\n"
                "          if (col < MAC_COLUMN_THRESHOLD)\n"
                "            psum[row][col] <= psum[row][col] + activation[row] * weight[row][col];\n"
                "          else\n"
            )
        return setup + "\n  integer row;\n  integer col;\n  always_ff @(posedge clk) begin\n" + """    if (rst || clear) begin
      for (row = 0; row < ARRAY_ROWS; row = row + 1)
        for (col = 0; col < ARRAY_COLS; col = col + 1)
          psum[row][col] <= '0;
    end else if (valid) begin
      for (row = 0; row < ARRAY_ROWS; row = row + 1)
        for (col = 0; col < ARRAY_COLS; col = col + 1) begin
""" + hybrid_line + """          psum[row][col] <= psum[row][col] + lut_mem[rom_index(row, col)][table_select[row][col] * LUT_ENTRIES + lookup_index[row][col]];
        end
    end
  end"""

    @staticmethod
    def _dataflow_id(dataflow: str) -> int:
        return {"output_stationary": 0, "weight_stationary": 1, "input_stationary": 2}[dataflow]

    @staticmethod
    def _primitive_id(primitive: Primitive) -> int:
        return {Primitive.MAC: 0, Primitive.LUT: 1, Primitive.HYBRID: 2}[primitive]

    @staticmethod
    def _request_dict(request: DseRequest) -> dict[str, object]:
        return {
            "shape": asdict(request.shape),
            "weight_precision": asdict(request.weight_precision),
            "input_precision": asdict(request.input_precision),
            "output_precision": asdict(request.output_precision),
            "constraints": asdict(request.constraints),
        }

    @staticmethod
    def _render_readme(module_name: str, has_lut: bool) -> str:
        lut_note = (
            "`lut_init.mem` is a zero-filled template. Populate it with `generator.write_lut_mem` using static quantized weights before functional simulation."
            if has_lut
            else "This MAC tile receives weights through its weight input ports."
        )
        return f"""# Generated PRISM Tile

`{module_name}.sv` is a parameter-specialized, synthesizable compute tile. The external controller drives `clear` at the start of an output tile and drives `valid` for each reduction step. The selected dataflow is emitted as `DATAFLOW_ID` metadata for controller and memory generation.

{lut_note}

`manifest.json` records the originating request and selected DSE configuration. `cost_report.json` is the analytical report for exactly this generated configuration.
"""
