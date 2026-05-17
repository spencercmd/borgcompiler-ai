import os
import time
import collections
import threading
import torch
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from compiler import trace_to_graph, compile_to_graph, lower_to_graph, benchmark_fn, DEVICE

app = FastAPI()

# ---------------------------------------------------------------------------
# Per-IP rate limiting (in-memory sliding window)
# Limits:  /compile  — 15 req / 60 s
#          /benchmark — 4 req / 300 s  (expensive: GPU warmup + 100 iterations)
# ---------------------------------------------------------------------------
_lock = threading.Lock()
_windows: dict[str, collections.deque] = collections.defaultdict(collections.deque)

def _check_rate(key: str, limit: int, window_s: int) -> None:
    now = time.monotonic()
    with _lock:
        dq = _windows[key]
        while dq and dq[0] < now - window_s:
            dq.popleft()
        if len(dq) >= limit:
            raise HTTPException(
                status_code=429,
                detail=f"Rate limit exceeded. Max {limit} requests per {window_s}s.",
            )
        dq.append(now)

def _ip(request: Request) -> str:
    return request.headers.get("x-forwarded-for", request.client.host).split(",")[0].strip()

_API_KEY = os.getenv("API_KEY")  # None in local dev → auth disabled

def _check_auth(request: Request) -> None:
    if not _API_KEY:
        return  # local dev: no key required
    if request.headers.get("x-api-key") != _API_KEY:
        raise HTTPException(status_code=403, detail="Forbidden")

# ALLOWED_ORIGINS: comma-separated list of allowed origins.
# Defaults to localhost dev server; set to the Vercel frontend URL in production.
_raw = os.getenv("ALLOWED_ORIGINS", "http://localhost:5173")
ALLOWED_ORIGINS = [o.strip() for o in _raw.split(",")]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "x-api-key"],
)


class CompileRequest(BaseModel):
    code: str


class CompileResponse(BaseModel):
    before: dict
    after: dict | None = None
    lowered: dict | None = None


@app.get("/device")
def get_device(request: Request):
    _check_auth(request)
    return {"device": DEVICE}


@app.post("/compile", response_model=CompileResponse)
def compile_graph(req: CompileRequest, request: Request):
    _check_auth(request)
    _check_rate(f"compile:{_ip(request)}", limit=15, window_s=60)
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
def run_benchmark(req: CompileRequest, request: Request):
    _check_auth(request)
    _check_rate(f"benchmark:{_ip(request)}", limit=4, window_s=300)
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
