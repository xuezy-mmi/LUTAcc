"""Shared schemas for compiler, functional models, and cycle simulation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Mapping, Optional, Tuple


class EvidenceKind(str, Enum):
    """Required provenance labels for every externally compared number."""

    REPORTED = "reported"
    DIGITIZED = "digitized"
    MEASURED = "measured"
    CALIBRATED = "calibrated"
    INFERRED = "inferred"
    PREDICTED = "predicted"


@dataclass(frozen=True)
class EvidenceValue:
    value: float
    kind: EvidenceKind
    source: str
    unit: str = ""
    uncertainty: Optional[float] = None

    def __post_init__(self) -> None:
        if not self.source:
            raise ValueError("evidence values require a source")
        if self.uncertainty is not None and self.uncertainty < 0:
            raise ValueError("uncertainty cannot be negative")

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result["kind"] = self.kind.value
        return result


@dataclass(frozen=True)
class GemmShape:
    """Logical MxK times KxN GEMM shape."""

    m: int
    k: int
    n: int
    batch: int = 1

    def __post_init__(self) -> None:
        if min(self.m, self.k, self.n, self.batch) <= 0:
            raise ValueError("GEMM dimensions and batch must be positive")

    @property
    def macs(self) -> int:
        return self.batch * self.m * self.k * self.n

    @property
    def output_elements(self) -> int:
        return self.batch * self.m * self.n

    def to_dict(self) -> Dict[str, int]:
        return asdict(self)


@dataclass(frozen=True)
class AttentionShape:
    batch: int
    heads: int
    query_length: int
    kv_length: int
    head_dim: int

    def __post_init__(self) -> None:
        if min(
            self.batch,
            self.heads,
            self.query_length,
            self.kv_length,
            self.head_dim,
        ) <= 0:
            raise ValueError("attention dimensions must be positive")

    @property
    def qk_macs(self) -> int:
        return (
            self.batch
            * self.heads
            * self.query_length
            * self.kv_length
            * self.head_dim
        )

    @property
    def pv_macs(self) -> int:
        return self.qk_macs

    def to_dict(self) -> Dict[str, int]:
        return asdict(self)


@dataclass(frozen=True)
class StageResult:
    name: str
    resource: str
    start_cycle: int
    end_cycle: int
    busy_cycles: int
    operations: int = 0
    read_bytes: int = 0
    write_bytes: int = 0
    energy_nj: float = 0.0
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.start_cycle < 0 or self.end_cycle < self.start_cycle:
            raise ValueError("invalid stage interval")
        if self.busy_cycles < 0:
            raise ValueError("busy cycles cannot be negative")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SimulationResult:
    design: str
    workload: str
    clock_hz: float
    cycles: int
    stages: List[StageResult]
    operations: int
    dram_read_bytes: int
    dram_write_bytes: int
    sram_read_bytes: int
    sram_write_bytes: int
    energy_mj: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.design or not self.workload:
            raise ValueError("design and workload names are required")
        if self.clock_hz <= 0 or self.cycles < 0:
            raise ValueError("clock must be positive and cycles non-negative")
        if min(
            self.operations,
            self.dram_read_bytes,
            self.dram_write_bytes,
            self.sram_read_bytes,
            self.sram_write_bytes,
        ) < 0:
            raise ValueError("counts cannot be negative")

    @property
    def latency_s(self) -> float:
        return self.cycles / self.clock_hz

    @property
    def latency_us(self) -> float:
        return self.latency_s * 1e6

    @property
    def throughput_tops(self) -> float:
        if self.cycles == 0:
            return 0.0
        return self.operations / self.latency_s / 1e12

    @property
    def tops_per_watt(self) -> float:
        if self.energy_mj <= 0 or self.latency_s <= 0:
            return 0.0
        average_power_w = self.energy_mj * 1e-3 / self.latency_s
        return self.throughput_tops / average_power_w

    def to_dict(self) -> Dict[str, Any]:
        return {
            "design": self.design,
            "workload": self.workload,
            "clock_hz": self.clock_hz,
            "cycles": self.cycles,
            "latency_us": self.latency_us,
            "operations": self.operations,
            "throughput_tops": self.throughput_tops,
            "dram_read_bytes": self.dram_read_bytes,
            "dram_write_bytes": self.dram_write_bytes,
            "sram_read_bytes": self.sram_read_bytes,
            "sram_write_bytes": self.sram_write_bytes,
            "energy_mj": self.energy_mj,
            "tops_per_watt": self.tops_per_watt,
            "stages": [stage.to_dict() for stage in self.stages],
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class ModelSpec:
    name: str
    layers: int
    hidden_size: int
    intermediate_size: int
    attention_heads: int
    kv_heads: int
    head_dim: int

    def __post_init__(self) -> None:
        if min(
            self.layers,
            self.hidden_size,
            self.intermediate_size,
            self.attention_heads,
            self.kv_heads,
            self.head_dim,
        ) <= 0:
            raise ValueError("model dimensions must be positive")
        if self.attention_heads % self.kv_heads:
            raise ValueError("attention_heads must be divisible by kv_heads")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


MODEL_SPECS: Dict[str, ModelSpec] = {
    "opt-6.7b": ModelSpec("opt-6.7b", 32, 4096, 16384, 32, 32, 128),
    "llama-30b": ModelSpec("llama-30b", 60, 6656, 17920, 52, 52, 128),
    "llama2-7b": ModelSpec("llama2-7b", 32, 4096, 11008, 32, 32, 128),
    "llama2-13b": ModelSpec("llama2-13b", 40, 5120, 13824, 40, 40, 128),
    "llama2-70b": ModelSpec("llama2-70b", 80, 8192, 28672, 64, 8, 128),
    "llama3-8b": ModelSpec("llama3-8b", 32, 4096, 14336, 32, 8, 128),
    "mistral-7b": ModelSpec("mistral-7b", 32, 4096, 14336, 32, 8, 128),
    "qwen2-7b": ModelSpec("qwen2-7b", 28, 3584, 18944, 28, 4, 128),
}


def model_spec(name: str) -> ModelSpec:
    key = name.lower()
    if key not in MODEL_SPECS:
        raise KeyError("unknown model %r; choices: %s" % (name, sorted(MODEL_SPECS)))
    return MODEL_SPECS[key]
