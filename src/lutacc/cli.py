"""Command-line interface for compilation and simulation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

from .compiler import (
    compile_omni_attention,
    compile_omni_gemm,
    compile_oasis_gemm,
    compile_transformer_layer,
    load_hf_model_spec,
)
from .compiler.ir import Program
from .schema import AttentionShape, GemmShape
from .sim import (
    simulate_decoder_linear,
    simulate_omni_attention,
    simulate_omni_gemm,
    simulate_omni_transformer,
    simulate_oasis_gemm,
    simulate_program,
)


def _write_json(document: Dict[str, Any], output: Optional[str]) -> None:
    encoded = json.dumps(document, indent=2, sort_keys=True) + "\n"
    if output:
        target = Path(output)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(encoded, encoding="utf-8")
    else:
        print(encoded, end="")


def _gemm_shape(arguments: argparse.Namespace) -> GemmShape:
    return GemmShape(arguments.m, arguments.k, arguments.n, arguments.batch)


def _attention_shape(arguments: argparse.Namespace) -> AttentionShape:
    return AttentionShape(
        arguments.batch,
        arguments.heads,
        arguments.query_length,
        arguments.kv_length,
        arguments.head_dim,
    )


def command_compile(arguments: argparse.Namespace) -> int:
    if arguments.design == "oasis":
        if arguments.workload == "attention":
            raise ValueError("OASIS compile accepts GEMM workloads")
        program = compile_oasis_gemm(
            _gemm_shape(arguments),
            activation_bits=arguments.activation_bits,
            outlier_fraction=arguments.outlier_fraction,
        )
    elif arguments.workload == "gemm":
        program = compile_omni_gemm(_gemm_shape(arguments))
    else:
        program = compile_omni_attention(
            _attention_shape(arguments),
            arguments.phase,
            arguments.key_bits,
            arguments.value_bits,
        )
    _write_json(program.to_dict(), arguments.output)
    return 0


def command_simulate(arguments: argparse.Namespace) -> int:
    if arguments.design == "oasis":
        if arguments.workload == "attention":
            raise ValueError("OASIS simulation accepts GEMM workloads")
        result = simulate_oasis_gemm(
            _gemm_shape(arguments),
            arguments.activation_bits,
            arguments.outlier_fraction,
            resident_weights=not arguments.stream_weights,
        )
    elif arguments.workload == "gemm":
        result = simulate_omni_gemm(
            _gemm_shape(arguments),
            resident_weights=not arguments.stream_weights,
        )
    else:
        result = simulate_omni_attention(
            _attention_shape(arguments),
            arguments.phase,
            arguments.key_bits,
            arguments.value_bits,
        )
    document = result.to_dict()
    _write_json(document, arguments.trace)
    if arguments.summary:
        print(
            "%s: %d cycles, %.3f us, %.3f TOPS, %.6f mJ"
            % (
                result.workload,
                result.cycles,
                result.latency_us,
                result.throughput_tops,
                result.energy_mj,
            )
        )
    return 0


def command_compile_model(arguments: argparse.Namespace) -> int:
    model = load_hf_model_spec(arguments.config, arguments.name)
    programs = compile_transformer_layer(
        model,
        arguments.batch,
        arguments.prompt_length,
        arguments.decode_tokens,
        arguments.design,
    )
    output = Path(arguments.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    files = []
    for index, program in enumerate(programs):
        path = output / ("%03d_%s.json" % (index, program.name.replace("/", "_")))
        program.write_json(str(path))
        files.append(str(path))
    _write_json(
        {
            "model": model.to_dict(),
            "design": arguments.design,
            "program_count": len(programs),
            "programs": files,
        },
        str(output / "manifest.json"),
    )
    return 0


def command_simulate_ir(arguments: argparse.Namespace) -> int:
    program = Program.read_json(arguments.program)
    result = simulate_program(
        program,
        resident_weights=not arguments.stream_weights,
    )
    _write_json(result.to_dict(), arguments.trace)
    if arguments.summary:
        print(
            "%s: %d cycles, %.3f us, %.3f TOPS, %.6f mJ"
            % (
                program.name,
                result.cycles,
                result.latency_us,
                result.throughput_tops,
                result.energy_mj,
            )
        )
    return 0


def command_simulate_model(arguments: argparse.Namespace) -> int:
    model = load_hf_model_spec(arguments.config, arguments.name)
    if arguments.design == "oasis":
        if arguments.phase != "decode" or arguments.query_length != 1:
            raise ValueError("the OASIS system model currently represents one-token decode")
        result = simulate_decoder_linear(
            model,
            "oasis",
            arguments.activation_bits,
            arguments.batch,
        )
    else:
        result = simulate_omni_transformer(
            model,
            arguments.phase,
            arguments.query_length,
            arguments.kv_length,
            arguments.batch,
            arguments.key_bits,
            arguments.value_bits,
        )
    document = result.to_dict()
    _write_json(document, arguments.output)
    if arguments.summary:
        cycles = document.get("cycles", document.get("cycles_per_token"))
        print(
            "%s %s: %d cycles, %.3f ms"
            % (arguments.design, model.name, cycles, document["latency_ms"])
        )
    return 0


def _add_shared(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--design", choices=("oasis", "omni-lut"), required=True)
    parser.add_argument(
        "--workload",
        choices=("gemm", "attention"),
        help="defaults to GEMM for OASIS and attention for Omni-LUT",
    )
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--activation-bits", type=int, choices=(3, 4), default=4)
    parser.add_argument("--outlier-fraction", type=float, default=0.01)
    parser.add_argument("--m", type=int, default=1)
    parser.add_argument("--k", type=int, default=4096)
    parser.add_argument("--n", type=int, default=4096)
    parser.add_argument("--heads", type=int, default=32)
    parser.add_argument("--query-length", type=int, default=1)
    parser.add_argument("--kv-length", type=int, default=2048)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--phase", choices=("prefill", "decode"), default="decode")
    parser.add_argument("--key-bits", type=float, default=3.0)
    parser.add_argument("--value-bits", type=float, default=3.0)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lutacc",
        description="Compile and simulate OASIS/Omni-LUT workloads",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    compile_parser = subparsers.add_parser("compile", help="emit validated LUTAcc IR")
    _add_shared(compile_parser)
    compile_parser.add_argument("--output")
    compile_parser.set_defaults(handler=command_compile)

    simulate_parser = subparsers.add_parser("simulate", help="run cycle-level simulation")
    _add_shared(simulate_parser)
    simulate_parser.add_argument("--trace", help="write full JSON result")
    simulate_parser.add_argument("--summary", action="store_true")
    simulate_parser.add_argument(
        "--stream-weights",
        action="store_true",
        help="charge the GEMM's packed weights to DRAM instead of assuming residency",
    )
    simulate_parser.set_defaults(handler=command_simulate)

    ir_simulate_parser = subparsers.add_parser(
        "simulate-ir", help="validate and simulate a serialized LUTAcc IR program"
    )
    ir_simulate_parser.add_argument("--program", required=True)
    ir_simulate_parser.add_argument("--trace", help="write full JSON result")
    ir_simulate_parser.add_argument("--summary", action="store_true")
    ir_simulate_parser.add_argument("--stream-weights", action="store_true")
    ir_simulate_parser.set_defaults(handler=command_simulate_ir)

    model_parser = subparsers.add_parser(
        "compile-model", help="lower a Hugging Face config.json"
    )
    model_parser.add_argument("--config", required=True)
    model_parser.add_argument("--name", default="")
    model_parser.add_argument("--design", choices=("oasis", "omni-lut"), required=True)
    model_parser.add_argument("--batch", type=int, default=1)
    model_parser.add_argument("--prompt-length", type=int, default=128)
    model_parser.add_argument("--decode-tokens", type=int, default=1)
    model_parser.add_argument("--output-dir", required=True)
    model_parser.set_defaults(handler=command_compile_model)

    model_simulate_parser = subparsers.add_parser(
        "simulate-model",
        help="simulate model-level accelerator paths from HF config.json",
    )
    model_simulate_parser.add_argument("--config", required=True)
    model_simulate_parser.add_argument("--name", default="")
    model_simulate_parser.add_argument(
        "--design", choices=("oasis", "omni-lut"), required=True
    )
    model_simulate_parser.add_argument("--batch", type=int, default=1)
    model_simulate_parser.add_argument(
        "--phase", choices=("prefill", "decode"), default="decode"
    )
    model_simulate_parser.add_argument("--query-length", type=int, default=1)
    model_simulate_parser.add_argument("--kv-length", type=int, default=2048)
    model_simulate_parser.add_argument(
        "--activation-bits", type=int, choices=(3, 4), default=4
    )
    model_simulate_parser.add_argument("--key-bits", type=float, default=3.0)
    model_simulate_parser.add_argument("--value-bits", type=float, default=3.0)
    model_simulate_parser.add_argument("--output")
    model_simulate_parser.add_argument("--summary", action="store_true")
    model_simulate_parser.set_defaults(handler=command_simulate_model)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    arguments = parser.parse_args(argv)
    return int(arguments.handler(arguments))


if __name__ == "__main__":
    raise SystemExit(main())
