## compiler internals

### FX graph node.meta["val"]
Dynamo's shape propagation pass populates `node.meta["val"]` with a FakeTensor — a fake tensor that has the real shape/dtype but no actual data. `symbolic_trace` doesn't run Dynamo, so this field is absent there. Always guard with `.get("val")`.

### _collect_inputs — why recursive?
`output` nodes wrap their args in nested tuples: `((node,),)`. `call_function` nodes have flat args: `(node1, node2)`. A simple `isinstance(arg, fx.Node)` check misses the nested case and leaves the output node disconnected. Recursive traversal handles both.

### benchmark compile_time_ms
The first call to a compiled function triggers Dynamo tracing + Inductor codegen. This can take 2–10 seconds. Measuring it separately lets you distinguish "compilation cost (one-time)" from "steady-state inference latency." Production systems (vLLM, TensorRT) do explicit warmup passes to absorb this before serving traffic.

### aot_eager vs inductor on CPU
`aot_eager` decomposes ops into primitives but doesn't fuse or codegen kernels. For small tensors this is strictly slower than eager — more kernel launches, more dispatch overhead, no fusion benefit. The crossover to faster-than-eager only happens with `inductor` + Triton on GPU, where fused kernels eliminate memory round-trips.

---

When you write:

x = torch.relu(x)

x = x * 2

x = x + bias

Python executes each line the moment it sees it. It has **no idea** that `* 2` and `+ bias` are coming. It can't fuse anything because it's always looking one step ahead.  

TorchDynamo solves this by intercepting Python's bytecode before it executes, tracing through your function to build a complete graph of every operation. Now the compiler sees the whole picture and can:

Fuse adjacent element-wise ops
Reorder operations
Eliminate redundant reads/writes
Choose optimal tile sizes for your specific GPU
The two matmuls can't be fused. Here's why:


scores = Q @ K.T          # ← produces a big NxN matrix
scores = scores / sqrt(d) # ← depends on that full matrix
scores = softmax(scores)  # ← depends on the FULL ROW (needs the max value)
out = scores @ V          # ← depends on softmax output
There's a data dependency chain. You can't start softmax until Q @ K.T is completely done, and you can't start scores @ V until softmax is done. Fusing requires that operations can be interleaved on the same data tile — but a matmul produces a large intermediate result that another matmul needs whole.


# FX graph — a list of nodes where each node is one operation:

# Your code:
def f(x, w):
    return torch.relu(x @ w)

# FX graph (simplified):
# node 0: placeholder 'x'
# node 1: placeholder 'w'  
# node 2: call_function(torch.mm, args=(x, w))
# node 3: call_function(torch.relu, args=(node2,))
# node 4: output(node3)

Each node knows its inputs (edges in the graph). This is exactly what we'll render visually.

A graph of \(f(x)\) is the visual representation of a mathematical function, where the input \(x\) is plotted on the horizontal axis and the output \(f(x)\) (or \(y\)) is plotted on the vertical axis. Each coordinate on the curve represents a unique ordered pair \((x, f(x))\)

Each node, data structure is a dict:

dictionary (or a dataclass/object, which is a dictionary with a name).

node = {
    "id": "node_2",
    "op": "torch.mm",
    "inputs": ["node_0", "node_1"],  # references to other nodes by id
    "shape": [512, 512],             # output tensor shape
}

So the full FX graph is:

graph = [
    {"id": "node_0", "op": "placeholder", "name": "x",       "inputs": []},
    {"id": "node_1", "op": "placeholder", "name": "w",       "inputs": []},
    {"id": "node_2", "op": "torch.mm",    "inputs": ["node_0", "node_1"]},
    {"id": "node_3", "op": "torch.relu",  "inputs": ["node_2"]},
    {"id": "node_4", "op": "output",      "inputs": ["node_3"]},
]

Good instinct on the constants — let me make it concrete:

If we included 8 as an input ID, our edges list would contain {"source": "8", "target": "scale_node"}. But there's no node with id: "8" in the nodes list. React Flow would have an arrow pointing to nothing — a dangling edge. It would either crash or render broken arrows.

The isinstance(arg, fx.Node) filter is exactly the guard: "only draw edges to things that are actual nodes in the graph."

__call__ is the dunder method that makes any object behave like a function when you use (). But it doesn't turn a string into a function.

exec() takes a string and executes it as Python code.

Think of exec as a mini-interpreter inside your interpreter. You hand it a string, it runs it, and any names defined inside (like my_fn) land in the namespace dict you provide.

Even if Triton wanted to ship pre-compiled kernels, it couldn't. The kernels are specialized to your specific inputs at the moment you call them. The tile sizes, loop bounds, and thread grid dimensions are all baked in at compile time — they depend on the exact tensor shapes you pass. A kernel compiled for [512, 512] is different code than one for [4, 4].

No package can pre-compile every possible shape combination. So Triton compiles on first call, caches the result, and reuses it on subsequent calls with the same shapes. That's what the triton-cache volume in our docker-compose.yml preserves — so you don't recompile on every container restart.

This is also why production ML serving systems like vLLM do a "warmup" pass when they start — they're triggering Triton compilation for the shapes they'll see in production, so the first real user request hits the cache, not a compile.

That's exactly right. More precisely:

torch.max has two behaviors depending on arguments — it either returns the single max scalar or returns (values, indices) as a named tuple. That ambiguity makes it hard for a compiler to reason about — it doesn't know what shape or type the output is without inspecting the call site
aten.amax is unambiguous: always returns a tensor of reduced values, no indices, no overloading. The compiler knows exactly what it produces. It's part of the ATen (A Tensor Library) operator set — the "assembly language" of PyTorch, where every op has a single well-defined signature.

matmul is the most hand-optimized operation in all of computing. NVIDIA has entire teams that do nothing but tune matrix multiplication — it maps directly onto Tensor Cores, which are dedicated hardware units that compute D = A×B + C at peak throughput. The algorithm (Strassen, Winograd, blocking strategies) is baked into cuBLAS at the hardware level.

So matmul is a compiler primitive — a leaf node that gets handed directly to cuBLAS/cuDNN without inspection. Same with conv2d, attention (via FlashAttention), and var_mean.
You decompose:   softmax, layer_norm, gelu, dropout
                 (high-level, can be expressed as fusable primitives)

You don't decompose: matmul, conv2d, var_mean, attention
                     (hardware-optimized — decomposing loses performance)


The first call to a compiled function triggers compilation — that takes 2-10 seconds. If you measure that call, you're benchmarking the compiler, not the kernel. You need to warm up before measuring.

On GPU, PyTorch launches kernels asynchronously — the CPU doesn't wait for them to finish. If you time with time.perf_counter() without syncing, you measure the time to launch the kernel, not run it. Always call torch.cuda.synchronize() before stopping the timer.

The first few runs after warmup may still be slow (cache misses, OS scheduling). Use the median over many runs, not the mean — a single slow outlier can inflate the mean significantly.

The first few runs after warmup may still be slow (cache misses, OS scheduling). Use the median over many runs, not the mean — a single slow outlier can inflate the mean significantly.

The median is the 50th percentile value. One OS interrupt, one cache miss, one GC pause doesn't move it. With the mean, a single 500ms spike across 100 runs inflates every reported number by 5ms. In ML benchmarking, those spikes happen constantly.

On CPU without MSVC, torch.compile(backend="inductor") fails — we saw this. So we'll use aot_eager as the CPU compiled backend. It applies graph-level optimizations (operator fusion at the Python level, dead code elimination) but no kernel compilation. The speedup will be modest on CPU, sometimes even negative — and that's a useful result to show. It teaches you that torch.compile isn't always faster, especially on small tensors where compilation overhead dominates.

AOT eager doesn't stay at the Python level like eager mode does. It decomposes softmax into primitive ops first (that's the "lowered IR" stage you already saw — amax, sub, exp, sum, div). Then on every call it dispatches those individually.

More kernel launches, not fewer. Eager torch.softmax calls one optimized C++ kernel. AOT eager calls 5 primitive kernels sequentially. Each launch has a fixed cost (~microseconds). For a 4×4 tensor, the compute per kernel is nanoseconds. You pay launch cost 5× for nearly no compute.

2. Dispatch indirection on every call. Each call goes through the AOT dispatcher, which resolves the graph structure, checks tensor shapes against the compiled specialization, and routes each node. For large tensors this amortizes to zero. For tiny tensors it's the dominant cost.
Think of it this way: you hired a stage manager who, before every performance, reads the full script out loud, assigns each actor to their mark, and signals "go." For a Broadway production (large tensor), this overhead is imperceptible. For a two-line improv sketch (4×4 softmax), the stage manager takes longer than the show.

On small tensors: compiled = slower (overhead dominates)
On large tensors with Triton: compiled = 3–5× faster (fusion + parallelism dominate)

Think about the direction: you need to make the tensor bigger, not smaller. The question is: at what size does actual compute time grow large enough to dwarf the fixed dispatch overhead?

Here's the math framing:

total_time = dispatch_overhead (fixed) + compute_time (scales with tensor size)
For compiled (Triton), compute_time scales more efficiently — fusion eliminates memory round-trips.

So the crossover happens when:

compute_time(compiled) < compute_time(eager)
...and both are large enough that dispatch_overhead is negligible.

The answer isn't a specific number you'd know offhand — it depends on hardware. But the shape: you need enough elements that the GPU (or CPU) is actually saturated doing math, not just spinning up kernels.

# 1. warmup  (discard these)
for _ in range(warmup):
    fn(*inputs)

# 2. measure  (keep these)
for _ in range(iters):
    if cuda: torch.cuda.synchronize()
    t0 = time.perf_counter()
    fn(*inputs)
    if cuda: torch.cuda.synchronize()
    t1 = time.perf_counter()
    times.append(t1 - t0)

return median(times)

## react 

useState doesn't just store data — it creates a subscription. React tracks which component called useState, and when the setter is invoked, React knows exactly which component tree to re-evaluate.

## docker

The pytorch/pytorch:2.7.0-cuda12.8-cudnn9-devel image contains:

Ubuntu base OS (~200MB)
CUDA Toolkit 12.8 — compiler, libraries, headers (~3GB)
cuDNN 9 — deep learning primitives (~500MB)
PyTorch itself (~1.5GB)
Python + pip
The devel (development) variant includes the full CUDA compiler toolchain so Triton can compile kernels at runtime inside the container. A runtime variant would be half the size but Triton's JIT would fail.


When you do ./frontend:/app, Docker replaces the entire /app directory with your Windows host files. That overwrites the node_modules the container installed during docker build — gone.

The anonymous volume - /app/node_modules creates an exception: Docker mounts a container-managed volume on top of the bind mount for just that subdirectory. So:

/app               ← your Windows source files (bind mount)
/app/node_modules  ← container's own Linux packages (anonymous volume wins here)
Without it, two things break:

node_modules disappears (overwritten by the host mount)
Even if you had node_modules on Windows, they'd have wrong-platform native binaries — some npm packages compile platform-specific .node addons. Windows binaries don't run in a Linux container.

 misused useState to trigger a side effect. That should be useEffect:

 useEffect is one of React's most fundamental hooks.

useEffect(fn, deps) runs fn whenever any value in deps changes. The dependency array controls when the effect fires:

The [] tells React: "this effect has no dependencies that ever change, so run it once and never again." Perfect for a one-time server check.

