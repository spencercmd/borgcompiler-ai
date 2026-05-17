import torch
import torch.fx as fx
from typing import Any
from torch._dynamo.backends.common import aot_autograd
from torch._inductor.decomposition import select_decomp_table

# Use GPU if available — Inductor generates Triton kernels on CUDA,
# C++ AVX kernels on CPU. Both paths work; GPU gives richer output.
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _collect_inputs(args: tuple) -> list[str]:
    """
    Recursively walk args to find all Node references.
    Needed because output nodes wrap their args in nested tuples: ((node,),)
    while call_function nodes have flat args: (node1, node2).
    """
    result = []
    for arg in args:
        if isinstance(arg, fx.Node):
            result.append(arg.name)
        elif isinstance(arg, (tuple, list)):
            result.extend(_collect_inputs(arg))
    return result


def _graph_module_to_dict(gm: fx.GraphModule) -> dict:
    """Shared helper: serialize an FX GraphModule to {nodes, edges}."""
    nodes = []
    for node in gm.graph.nodes:
        input_ids = _collect_inputs(node.args)
        target = node.target
        if callable(target) and hasattr(target, "__name__"):
            target_label = f"torch.{target.__name__}"
        else:
            target_label = str(target)

        # node.meta["val"] is a FakeTensor set by Dynamo's shape propagation.
        # symbolic_trace doesn't populate it, so we guard with .get().
        shape_label = None
        val = node.meta.get("val")
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
    """
    Run fn through AOT Autograd with Inductor's decomposition table.

    This is Stage 3: the IR that Triton's scheduler actually receives.
    High-level ops like softmax, layer_norm, gelu are broken into the
    primitive aten ops (exp, sum, div, etc.) that Inductor fuses into kernels.
    """
    torch._dynamo.reset()

    captured: dict = {}

    def fw_compiler(gm: fx.GraphModule, example_inputs):
        # This callback fires after AOT Autograd has decomposed the graph.
        # We capture it here, before any Inductor codegen runs.
        captured["gm"] = gm
        return gm  # return as-is — no actual compilation needed

    backend = aot_autograd(
        fw_compiler=fw_compiler,
        decompositions=select_decomp_table(),
    )

    inputs_on_device = [
        t.to(DEVICE) if isinstance(t, torch.Tensor) else t
        for t in example_inputs
    ]

    torch._dynamo.reset()
    compiled = torch.compile(fn, backend=backend, fullgraph=True)
    compiled(*inputs_on_device)

    if "gm" not in captured:
        raise RuntimeError("AOT Autograd did not capture a forward graph.")

    return _graph_module_to_dict(captured["gm"])


def compile_to_graph(fn: callable, example_inputs: list[Any]) -> dict:
    """
    Run fn through TorchDynamo's export to get the post-optimization graph.
    Requires real tensors in example_inputs so Dynamo can specialize shapes.
    """
    torch._dynamo.reset()

    # Move example_inputs to the best available device so Dynamo specializes
    # for the correct hardware and shape propagation reflects the real dtype.
    inputs_on_device = [
        t.to(DEVICE) if isinstance(t, torch.Tensor) else t
        for t in example_inputs
    ]

    export_result = torch._dynamo.export(fn)(*inputs_on_device)
    return _graph_module_to_dict(export_result.graph_module)
