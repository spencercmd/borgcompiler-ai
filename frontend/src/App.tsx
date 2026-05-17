import { useState, useEffect } from "react";
import Editor from "@monaco-editor/react";
import { GraphView } from "./components/GraphView";
import type { Node, Edge } from "@xyflow/react";

const DEFAULT_CODE = `def fn(x, w):
    return torch.relu(x @ w)

# Define example_inputs to enable the Compare view.
# Dynamo specializes the compiled graph to these shapes.
example_inputs = [torch.randn(4, 4), torch.randn(4, 4)]
`;

interface GraphData {
  nodes: Node[];
  edges: Edge[];
}

function toReactFlow(rawNodes: any[], rawEdges: any[]): GraphData {
  return {
    nodes: rawNodes.map((n: any) => ({
      id: n.id,
      type: "op",
      data: { label: n.target, op: n.op, shape: n.shape ?? null },
      position: { x: 0, y: 0 },
    })),
    edges: rawEdges.map((e: any) => ({
      id: `${e.source}->${e.target}`,
      source: e.source,
      target: e.target,
      animated: true,
    })),
  };
}

const API = import.meta.env.VITE_API_URL ?? "http://localhost:8000";

export default function App() {
  const [code, setCode] = useState(DEFAULT_CODE);
  const [before, setBefore] = useState<GraphData | null>(null);
  const [after, setAfter] = useState<GraphData | null>(null);
  const [compare, setCompare] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [device, setDevice] = useState<string | null>(null);

  // Fetch active compute device once on mount
  useEffect(() => {
    fetch(`${API}/device`).then(r => r.json()).then(d => setDevice(d.device)).catch(() => {});
  }, []);

  async function compile() {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(`${API}/compile`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ code }),
      });
      const data = await res.json();
      if (!res.ok) {
        setError(data.detail ?? "Unknown error");
        return;
      }
      setBefore(toReactFlow(data.before.nodes, data.before.edges));
      if (data.after) {
        setAfter(toReactFlow(data.after.nodes, data.after.edges));
      } else {
        setAfter(null);
        setCompare(false);
      }
    } catch (e: any) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }

  const canCompare = after !== null;

  return (
    <div style={styles.root}>
      <header style={styles.header}>
        <div>
          <h1 style={styles.title}>LLM Compiler Explorer</h1>
          <p style={styles.subtitle}>Write a PyTorch function → see its computation graph</p>
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
          {device && (
            <span style={{
              padding: "4px 10px", borderRadius: 6, fontSize: 12, fontWeight: 600,
              background: device === "cuda" ? "#14532d" : "#1e293b",
              color: device === "cuda" ? "#4ade80" : "#64748b",
            }}>
              {device === "cuda" ? "GPU" : "CPU"}
            </span>
          )}
          {canCompare && (
            <button
              style={{ ...styles.button, background: compare ? "#0ea5e9" : "#334155" }}
              onClick={() => setCompare((c) => !c)}
            >
              {compare ? "◀ Single view" : "⇔ Compare"}
            </button>
          )}
        </div>
      </header>
      <div style={styles.body}>
        <div style={styles.left}>
          <div style={styles.editorWrap}>
            <Editor
              height="100%"
              defaultLanguage="python"
              value={code}
              onChange={(v) => setCode(v ?? "")}
              theme="vs-dark"
              options={{ minimap: { enabled: false }, fontSize: 14 }}
            />
          </div>
          <button
            style={{ ...styles.button, opacity: loading ? 0.6 : 1 }}
            onClick={compile}
            disabled={loading}
          >
            {loading ? "Compiling…" : "▶ Compile"}
          </button>
          {error && <div style={styles.error}>{error}</div>}
        </div>
        <div style={styles.right}>
          {!before ? (
            <div style={styles.placeholder}>Hit Compile to visualize the graph</div>
          ) : compare && after ? (
            <div style={styles.splitPane}>
              <div style={styles.pane}>
                <div style={styles.paneLabel}>Before (FX trace)</div>
                <GraphView key="before" nodes={before.nodes} edges={before.edges} />
              </div>
              <div style={styles.divider} />
              <div style={styles.pane}>
                <div style={styles.paneLabel}>After (Dynamo export)</div>
                <GraphView key="after" nodes={after.nodes} edges={after.edges} />
              </div>
            </div>
          ) : (
            <GraphView nodes={before.nodes} edges={before.edges} />
          )}
        </div>
      </div>
    </div>
  );
}

const styles: Record<string, React.CSSProperties> = {
  root: { display: "flex", flexDirection: "column", height: "100vh", background: "#0f172a", color: "#e2e8f0", fontFamily: "system-ui, sans-serif" },
  header: { display: "flex", alignItems: "center", justifyContent: "space-between", padding: "16px 24px", borderBottom: "1px solid #1e293b" },
  title: { margin: 0, fontSize: 20, fontWeight: 700, color: "#f8fafc" },
  subtitle: { margin: "4px 0 0", fontSize: 13, color: "#64748b" },
  body: { display: "flex", flex: 1, overflow: "hidden" },
  left: { display: "flex", flexDirection: "column", width: 420, borderRight: "1px solid #1e293b", padding: 16, gap: 12 },
  editorWrap: { flex: 1, borderRadius: 8, overflow: "hidden", border: "1px solid #1e293b" },
  button: { padding: "10px 20px", background: "#6366f1", color: "#fff", border: "none", borderRadius: 8, cursor: "pointer", fontSize: 14, fontWeight: 600 },
  error: { padding: 12, background: "#450a0a", color: "#fca5a5", borderRadius: 8, fontSize: 13 },
  right: { flex: 1, display: "flex", flexDirection: "column", overflow: "hidden" },
  placeholder: { display: "flex", alignItems: "center", justifyContent: "center", height: "100%", color: "#334155", fontSize: 14 },
  splitPane: { display: "flex", height: "100%" },
  pane: { flex: 1, display: "flex", flexDirection: "column", overflow: "hidden", position: "relative" },
  paneLabel: { padding: "8px 16px", fontSize: 12, fontWeight: 600, color: "#64748b", borderBottom: "1px solid #1e293b", background: "#0f172a" },
  divider: { width: 1, background: "#1e293b" },
};
