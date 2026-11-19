"""Collect operator and kernel names for the three matched HF bias-free projections."""

from torch.nn.functional import linear
from torch.profiler import ProfilerActivity, profile, record_function


PROJECTIONS = ("q", "k", "v")
PREFIX = "W4_HF_LAYER0_PREBIAS_"
MATRIX_OPS = {"aten::linear", "aten::matmul", "aten::mm", "aten::addmm", "aten::bmm"}


def children(event):
    yield event
    for child in event.cpu_children:
        yield from children(child)


def summarize(profiler, *, require_device):
    events = profiler.events()
    phases = {}
    for projection in PROJECTIONS:
        label = PREFIX + projection
        matches = [event for event in events if event.name == label and str(event.device_type) == "DeviceType.CPU"]
        if len(matches) != 1:
            raise ValueError(f"expected one profiler region for projection {projection}")
        operations = []
        for event in children(matches[0]):
            kernels = [kernel.name for kernel in event.kernels]
            if event.name in MATRIX_OPS or kernels:
                operations.append({"operator": event.name, "input_shapes": event.input_shapes, "device_kernels": kernels})
        if not operations or (require_device and not any(op["device_kernels"] for op in operations)):
            raise ValueError(f"missing matrix operation or device kernel for projection {projection}")
        phases[projection] = operations
    return phases


def capture(normalized, weight_rows, *, require_device=True):
    activities = [ProfilerActivity.CPU]
    if require_device:
        activities.append(ProfilerActivity.CUDA)
    outputs = []
    with profile(activities=activities, record_shapes=True) as profiler:
        for projection, weight in zip(PROJECTIONS, weight_rows, strict=True):
            with record_function(PREFIX + projection):
                outputs.append(linear(normalized, weight))
    return outputs, summarize(profiler, require_device=require_device)
