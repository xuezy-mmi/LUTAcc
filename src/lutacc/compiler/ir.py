"""Small serializable IR shared by both reconstructed accelerators."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple


VALID_DTYPES = {
    "fp16",
    "fp32",
    "int2",
    "int3",
    "int4",
    "uint4",
    "uint8",
    "index3",
    "index4",
}


@dataclass(frozen=True)
class TensorSpec:
    name: str
    shape: Tuple[int, ...]
    dtype: str
    storage: str
    layout: str = "row_major"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name or not self.shape or min(self.shape) <= 0:
            raise ValueError("tensor name and positive shape are required")
        if self.dtype not in VALID_DTYPES:
            raise ValueError("unsupported dtype %s" % self.dtype)
        if self.storage not in {"dram", "sram", "register", "stream"}:
            raise ValueError("unsupported storage %s" % self.storage)

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result["shape"] = list(self.shape)
        return result


@dataclass(frozen=True)
class Operation:
    name: str
    kind: str
    inputs: Tuple[str, ...]
    outputs: Tuple[str, ...]
    dependencies: Tuple[str, ...] = ()
    resource: str = "control"
    attributes: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name or not self.kind or not self.resource:
            raise ValueError("operation name, kind, and resource are required")
        if not self.outputs:
            raise ValueError("operation %s has no output" % self.name)

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        for key in ("inputs", "outputs", "dependencies"):
            result[key] = list(result[key])
        return result


@dataclass
class Program:
    name: str
    design: str
    tensors: List[TensorSpec]
    operations: List[Operation]
    metadata: Dict[str, Any] = field(default_factory=dict)
    schema_version: int = 1

    def validate(self) -> None:
        if self.design not in {"oasis", "omni-lut"}:
            raise ValueError("unsupported design %s" % self.design)
        tensor_names = [tensor.name for tensor in self.tensors]
        if len(tensor_names) != len(set(tensor_names)):
            raise ValueError("tensor names must be unique")
        operation_names = [operation.name for operation in self.operations]
        if len(operation_names) != len(set(operation_names)):
            raise ValueError("operation names must be unique")
        tensors = set(tensor_names)
        operations = {operation.name: operation for operation in self.operations}
        producers: Dict[str, str] = {}
        for operation in self.operations:
            unknown = set(operation.inputs + operation.outputs) - tensors
            if unknown:
                raise ValueError(
                    "operation %s references unknown tensors %s"
                    % (operation.name, sorted(unknown))
                )
            for output in operation.outputs:
                if output in producers:
                    raise ValueError("tensor %s has multiple producers" % output)
                producers[output] = operation.name
            for dependency in operation.dependencies:
                if dependency not in operations:
                    raise ValueError("unknown dependency %s" % dependency)
        self._topological_names(operations)

    def _topological_names(self, operations: Mapping[str, Operation]) -> List[str]:
        waiting = {name: set(op.dependencies) for name, op in operations.items()}
        ready = sorted(name for name, deps in waiting.items() if not deps)
        order: List[str] = []
        while ready:
            name = ready.pop(0)
            order.append(name)
            for other in sorted(waiting):
                if name in waiting[other]:
                    waiting[other].remove(name)
                    if not waiting[other] and other not in order and other not in ready:
                        ready.append(other)
            ready.sort()
        if len(order) != len(operations):
            cyclic = sorted(name for name, deps in waiting.items() if deps)
            raise ValueError("operation dependency cycle: %s" % cyclic)
        return order

    def topological_operations(self) -> List[Operation]:
        mapping = {operation.name: operation for operation in self.operations}
        return [mapping[name] for name in self._topological_names(mapping)]

    def to_dict(self) -> Dict[str, Any]:
        self.validate()
        return {
            "schema_version": self.schema_version,
            "name": self.name,
            "design": self.design,
            "tensors": [tensor.to_dict() for tensor in self.tensors],
            "operations": [operation.to_dict() for operation in self.operations],
            "metadata": self.metadata,
        }

    def write_json(self, path: str) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2, sort_keys=True)
            handle.write("\n")

    @classmethod
    def from_dict(cls, document: Mapping[str, Any]) -> "Program":
        version = int(document.get("schema_version", 1))
        if version != 1:
            raise ValueError("unsupported IR schema version %d" % version)
        tensors = [
            TensorSpec(
                name=item["name"],
                shape=tuple(int(value) for value in item["shape"]),
                dtype=item["dtype"],
                storage=item["storage"],
                layout=item.get("layout", "row_major"),
                metadata=dict(item.get("metadata", {})),
            )
            for item in document["tensors"]
        ]
        operations = [
            Operation(
                name=item["name"],
                kind=item["kind"],
                inputs=tuple(item["inputs"]),
                outputs=tuple(item["outputs"]),
                dependencies=tuple(item.get("dependencies", ())),
                resource=item.get("resource", "control"),
                attributes=dict(item.get("attributes", {})),
            )
            for item in document["operations"]
        ]
        program = cls(
            name=document["name"],
            design=document["design"],
            tensors=tensors,
            operations=operations,
            metadata=dict(document.get("metadata", {})),
            schema_version=version,
        )
        program.validate()
        return program

    @classmethod
    def read_json(cls, path: str) -> "Program":
        with Path(path).open("r", encoding="utf-8") as handle:
            document = json.load(handle)
        if not isinstance(document, dict):
            raise ValueError("IR JSON root must be an object")
        return cls.from_dict(document)


def tensor(
    name: str,
    shape: Sequence[int],
    dtype: str,
    storage: str,
    layout: str = "row_major",
    **metadata: Any
) -> TensorSpec:
    return TensorSpec(name, tuple(int(value) for value in shape), dtype, storage, layout, metadata)
