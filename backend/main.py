import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from compiler import trace_to_graph, compile_to_graph, lower_to_graph, benchmark_fn, DEVICE

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["POST"],
    allow_headers=["*"],
)


class CompileRequest(BaseModel):
    code: str


class CompileResponse(BaseModel):
    before: dict
    after: dict | None = None
    lowered: dict | None = None


@app.get("/device")
def get_device():
    return {"device": DEVICE}


@app.post("/compile", response_model=CompileResponse)
def compile_graph(req: CompileRequest):
    namespace = {"torch": torch}
    try:
        exec(req.code, namespace)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Code error: {e}")

    fn = namespace.get("fn")
    if fn is None:
        raise HTTPException(status_code=400, detail="Your code must define a function named 'fn'.")

    try:
        before = trace_to_graph(fn, [])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Trace error: {e}")

    example_inputs = namespace.get("example_inputs")
    after = None
    lowered = None

    if example_inputs is not None:
        try:
            after = compile_to_graph(fn, example_inputs)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Compile error: {e}")
        try:
            lowered = lower_to_graph(fn, example_inputs)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Lower error: {e}")

    return {"before": before, "after": after, "lowered": lowered}


@app.post("/benchmark")
def run_benchmark(req: CompileRequest):
    namespace = {"torch": torch}
    try:
        exec(req.code, namespace)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Code error: {e}")

    fn = namespace.get("fn")
    example_inputs = namespace.get("example_inputs")

    if fn is None:
        raise HTTPException(400, "Your code must define a function named 'fn'.")
    if example_inputs is None:
        raise HTTPException(400, "Your code must define 'example_inputs' to benchmark.")

    try:
        return benchmark_fn(fn, example_inputs)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Benchmark error: {e}")
