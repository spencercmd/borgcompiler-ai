import torch
import torch.fx as fx
from typing import Any


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


def compile_to_graph(fn: callable, example_inputs: list[Any]) -> dict:
    """
    Run fn through TorchDynamo's export to get the post-optimization graph.
    Requires real tensors in example_inputs so Dynamo can specialize shapes.
    """
    # Reset Dynamo's cache so previous compilations don't interfere.
    torch._dynamo.reset()

    # _dynamo.export() traces fn through Dynamo's full capture pipeline —
    # constant folding, op canonicalization, decomposition — and returns
    # an ExportResult whose .graph_module is the optimized FX graph.
    # Unlike torch.export.export(), this accepts plain functions (not just nn.Module).
    export_result = torch._dynamo.export(fn)(*example_inputs)
    return _graph_module_to_dict(export_result.graph_module)
