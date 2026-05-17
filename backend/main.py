import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from compiler import trace_to_graph, compile_to_graph, lower_to_graph, DEVICE

app = FastAPI()

# Allow the frontend (running on a different port) to call this API.
# Without this, browsers block cross-origin requests by default.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Vite's default dev port
    allow_methods=["POST"],
    allow_headers=["*"],
)


class CompileRequest(BaseModel):
    code: str  # raw Python source string from the editor


@app.get("/device")
def get_device():
    return {"device": DEVICE}


class CompileResponse(BaseModel):
    before: dict
    after: dict | None = None
    lowered: dict | None = None  # AOT decomposed IR — what Triton would compile


@app.post("/compile", response_model=CompileResponse)
def compile_graph(req: CompileRequest):
    namespace = {"torch": torch}
    try:
        exec(req.code, namespace)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Code error: {e}")

    fn = namespace.get("fn")
    if fn is None:
        raise HTTPException(
            status_code=400,
            detail="Your code must define a function named 'fn'.",
        )

    try:
        before = trace_to_graph(fn, [])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Trace error: {e}")

    # 'after' is only produced when the user also defines example_inputs.
    example_inputs = namespace.get("example_inputs")
    after = None
    if example_inputs is not None:
        try:
            after = compile_to_graph(fn, example_inputs)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Compile error: {e}")

    lowered = None
    if example_inputs is not None:
        try:
            lowered = lower_to_graph(fn, example_inputs)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Lower error: {e}")

    return {"before": before, "after": after, "lowered": lowered}
