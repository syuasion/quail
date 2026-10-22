"""Generic component roofline calculation."""

from __future__ import annotations

from dataclasses import dataclass

from quail.specs import DeviceSpec, Precision


@dataclass(frozen=True)
class CostComponent:
    """One model component's arithmetic and memory work."""

    name: str
    flops: float
    bytes_moved: float
    precision: Precision
    gemm: bool = False


@dataclass(frozen=True)
class ComponentLatency:
    """The two limits and final time for one component."""

    component: CostComponent
    compute_seconds: float
    memory_seconds: float

    @property
    def name(self) -> str:
        return self.component.name

    @property
    def flops(self) -> float:
        return self.component.flops

    @property
    def bytes_moved(self) -> float:
        return self.component.bytes_moved

    @property
    def precision(self) -> Precision:
        return self.component.precision

    @property
    def seconds(self) -> float:
        return max(self.compute_seconds, self.memory_seconds)

    @property
    def bound_by(self) -> str:
        return ("compute" if self.compute_seconds >= self.memory_seconds
                else "memory")


def component_latency(component: CostComponent,
                      device: DeviceSpec) -> ComponentLatency:
    """Price one component against the device limits."""
    bandwidth = (device.gemm_flops(component.precision) if component.gemm
                 else device.arithmetic_bandwidth(component.precision))
    return ComponentLatency(
        component=component,
        compute_seconds=component.flops / bandwidth,
        memory_seconds=component.bytes_moved / device.hbm_bw,
    )


def component_latencies(
        components, device: DeviceSpec) -> tuple[ComponentLatency, ...]:
    """Price model components in execution order."""
    return tuple(component_latency(component, device)
                 for component in components)
