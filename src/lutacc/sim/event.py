"""Deterministic integer-cycle DAG scheduler with resource contention."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

from ..schema import StageResult


@dataclass(frozen=True)
class EventSpec:
    name: str
    resource: str
    cycles: int
    dependencies: Tuple[str, ...] = ()
    operations: int = 0
    read_bytes: int = 0
    write_bytes: int = 0
    energy_nj: float = 0.0
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name or not self.resource or self.cycles < 0:
            raise ValueError("events require a name/resource and non-negative cycles")
        if min(self.operations, self.read_bytes, self.write_bytes) < 0:
            raise ValueError("event counters cannot be negative")


class EventScheduler:
    """ASAP list scheduler.

    Resources default to one lane. A resource with N lanes tracks N independent
    availability times and assigns each ready event to its earliest lane.
    Dependencies and all timing are integer cycles, which makes traces directly
    comparable with Verilator transaction traces.
    """

    def __init__(self, resource_lanes: Mapping[str, int] = None):
        self.resource_lanes = dict(resource_lanes or {})
        if any(value <= 0 for value in self.resource_lanes.values()):
            raise ValueError("resource lane counts must be positive")

    def schedule(self, events: Sequence[EventSpec]) -> List[StageResult]:
        by_name = {event.name: event for event in events}
        if len(by_name) != len(events):
            raise ValueError("event names must be unique")
        for event in events:
            unknown = set(event.dependencies) - set(by_name)
            if unknown:
                raise ValueError("event %s has unknown dependencies %s" % (event.name, unknown))
        waiting = {event.name: set(event.dependencies) for event in events}
        completed: Dict[str, StageResult] = {}
        availability: Dict[str, List[int]] = {}
        order = {event.name: index for index, event in enumerate(events)}
        while len(completed) < len(events):
            ready = [
                by_name[name]
                for name, dependencies in waiting.items()
                if name not in completed and dependencies.issubset(completed)
            ]
            if not ready:
                unresolved = sorted(set(by_name) - set(completed))
                raise RuntimeError("event dependency cycle: %s" % unresolved)
            ready.sort(
                key=lambda event: (
                    max((completed[name].end_cycle for name in event.dependencies), default=0),
                    order[event.name],
                )
            )
            event = ready[0]
            dependency_ready = max(
                (completed[name].end_cycle for name in event.dependencies), default=0
            )
            lanes = availability.setdefault(
                event.resource,
                [0] * self.resource_lanes.get(event.resource, 1),
            )
            lane_index = min(range(len(lanes)), key=lambda index: (lanes[index], index))
            start = max(dependency_ready, lanes[lane_index])
            end = start + event.cycles
            lanes[lane_index] = end
            completed[event.name] = StageResult(
                name=event.name,
                resource=event.resource,
                start_cycle=start,
                end_cycle=end,
                busy_cycles=event.cycles,
                operations=event.operations,
                read_bytes=event.read_bytes,
                write_bytes=event.write_bytes,
                energy_nj=event.energy_nj,
                metadata={**dict(event.metadata), "resource_lane": lane_index},
            )
        return [completed[event.name] for event in events]


def makespan(stages: Iterable[StageResult]) -> int:
    return max((stage.end_cycle for stage in stages), default=0)
