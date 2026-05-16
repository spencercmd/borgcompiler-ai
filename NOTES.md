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

## react 

useState doesn't just store data — it creates a subscription. React tracks which component called useState, and when the setter is invoked, React knows exactly which component tree to re-evaluate.