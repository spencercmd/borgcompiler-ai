# LLM Compiler Explorer

A Godbolt-style playground for the PyTorch compilation pipeline. Write a function, inspect every IR stage from FX graph to lowered ATen primitives, and benchmark eager vs compiled latency.

## Pipeline stages

| Stage | Tool | What you see |
|---|---|---|
| ① FX Trace | `torch.fx.symbolic_trace` | Raw computation graph, no shapes |
| ② Dynamo Export | `torch._dynamo.export` | Shape-specialized, canonicalized graph |
| ③ Lowered IR | `aot_autograd` + `select_decomp_table` | Primitive ATen ops Triton's scheduler receives |
| ④ Perf | `torch.compile` (inductor / aot_eager) | Eager vs compiled median latency + distribution |

## Stack

- **Backend** — FastAPI, PyTorch 2.7, TorchDynamo, TorchInductor
- **Frontend** — React, TypeScript, React Flow, Monaco Editor, Recharts
- **Runtime** — Docker; CUDA 12.8 + cuDNN 9 base image (Triton JIT requires `devel` variant)

## Running

```bash
docker compose up --build
```

Frontend: http://localhost:5173  
Backend: http://localhost:8000

## Usage

Define `fn` and optionally `example_inputs` in the editor:

```python
def fn(x):
    return torch.layer_norm(x, x.shape)

example_inputs = [torch.randn(512, 512)]
```

- **Compile** — traces all three graph stages
- **Benchmark** — runs 20 warmup + 100 timed iterations, reports median latency and speedup

## Notes

On CPU (no Triton), the benchmark backend falls back to `aot_eager`. Compiled latency will typically be **higher** than eager for small tensors — this is expected: AOT decomposition adds kernel launches without fusion. Speedup requires `inductor` + Triton on CUDA.
