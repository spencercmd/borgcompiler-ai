import time
import statistics
import torch
import torch.fx as fx
from typing import Any
from torch._dynamo.backends.common import aot_autograd
from torch._inductor.decomposition import select_decomp_table

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _collect_inputs(args: tuple) -> list[str]:
    # Recursively resolve Node refs from nested args (output nodes wrap in ((node,),))
    result = []
    for arg in args:
        if isinstance(arg, fx.Node):
            result.append(arg.name)
        elif isinstance(arg, (tuple, list)):
            result.extend(_collect_inputs(arg))
    return result


def _graph_module_to_dict(gm: fx.GraphModule) -> dict:
    """Serialize an FX GraphModule to {nodes, edges}."""
    nodes = []
    for node in gm.graph.nodes:
        input_ids = _collect_inputs(node.args)
        target = node.target
        if callable(target) and hasattr(target, "__name__"):
            target_label = f"torch.{target.__name__}"
        else:
            target_label = str(target)

        shape_label = None
        val = node.meta.get("val")  # FakeTensor populated by Dynamo; absent in symbolic_trace
        if val is not None and hasattr(val, "shape"):
            dtype_short = str(val.dtype).replace("torch.", "")
            shape_label = f"{list(val.shape)} {dtype_short}"

        nodes.append({
            "id": node.name,
            "op": node.op,
            "target": target_label,
            "inputs": input_ids,
            "shape": shape_label,
        })

    edges = [
        {"source": src, "target": node["id"]}
        for node in nodes
        for src in node["inputs"]
    ]
    return {"nodes": nodes, "edges": edges}


def trace_to_graph(fn: callable, example_inputs: list[Any]) -> dict:
    """Symbolically trace fn (no real inputs needed) → {nodes, edges}."""
    gm: fx.GraphModule = fx.symbolic_trace(fn)
    return _graph_module_to_dict(gm)


def lower_to_graph(fn: callable, example_inputs: list[Any]) -> dict:
    """AOT Autograd + Inductor decompositions → primitive ATen graph Triton receives."""
    captured: dict = {}

    def fw_compiler(gm: fx.GraphModule, example_inputs):
        captured["gm"] = gm
        return gm

    backend = aot_autograd(fw_compiler=fw_compiler, decompositions=select_decomp_table())
    inputs_on_device = [t.to(DEVICE) if isinstance(t, torch.Tensor) else t for t in example_inputs]

    torch._dynamo.reset()
    compiled = torch.compile(fn, backend=backend, fullgraph=True)
    compiled(*inputs_on_device)

    if "gm" not in captured:
        raise RuntimeError("AOT Autograd did not capture a forward graph.")

    return _graph_module_to_dict(captured["gm"])


def compile_to_graph(fn: callable, example_inputs: list[Any]) -> dict:
    """TorchDynamo export → shape-specialized, canonicalized graph."""
    torch._dynamo.reset()
    inputs_on_device = [t.to(DEVICE) if isinstance(t, torch.Tensor) else t for t in example_inputs]
    export_result = torch._dynamo.export(fn)(*inputs_on_device)
    return _graph_module_to_dict(export_result.graph_module)


def _time_fn(fn: callable, inputs: list, n_warmup: int, n_iter: int) -> list[float]:
    """Warmup then time fn(*inputs); returns per-call latency in ms. Syncs on CUDA."""
    is_cuda = DEVICE == "cuda"
    for _ in range(n_warmup):
        fn(*inputs)
    times = []
    for _ in range(n_iter):
        if is_cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn(*inputs)
        if is_cuda:
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
    return times


def benchmark_fn(fn: callable, example_inputs: list[Any], n_warmup: int = 20, n_iter: int = 100) -> dict:
    """Benchmark eager vs compiled (inductor on CUDA, aot_eager on CPU). Returns medians + samples."""
    inputs = [t.to(DEVICE) if isinstance(t, torch.Tensor) else t for t in example_inputs]

    eager_runs = _time_fn(fn, inputs, n_warmup, n_iter)

    torch._dynamo.reset()
    backend = "inductor" if DEVICE == "cuda" else "aot_eager"
    compiled = torch.compile(fn, backend=backend)

    # First call compiles — measure separately as one-time cost
    try:
        t0 = time.perf_counter()
        compiled(*inputs)
        if DEVICE == "cuda":
            torch.cuda.synchronize()
        compile_ms = (time.perf_counter() - t0) * 1000
    except Exception:
        # Triton unavailable — fall back to aot_eager
        torch._dynamo.reset()
        backend = "aot_eager"
        compiled = torch.compile(fn, backend=backend)
        t0 = time.perf_counter()
        compiled(*inputs)
        compile_ms = (time.perf_counter() - t0) * 1000

    compiled_runs = _time_fn(compiled, inputs, n_warmup, n_iter)

    eager_median = statistics.median(eager_runs)
    compiled_median = statistics.median(compiled_runs)

    return {
        "eager_ms":        round(eager_median, 4),
        "compiled_ms":     round(compiled_median, 4),
        "compile_time_ms": round(compile_ms, 1),
        "speedup":         round(eager_median / compiled_median, 2),
        "backend":         backend,
        "eager_runs":      [round(x, 4) for x in eager_runs[:50]],
        "compiled_runs":   [round(x, 4) for x in compiled_runs[:50]],
    }
