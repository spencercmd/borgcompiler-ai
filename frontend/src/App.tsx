import { useState, useEffect } from "react";
import Editor from "@monaco-editor/react";
import { GraphView } from "./components/GraphView";
import { BenchmarkPanel } from "./components/BenchmarkPanel";
import type { Node, Edge } from "@xyflow/react";

const DEFAULT_CODE = `def fn(x):
    return torch.softmax(x, dim=-1)

example_inputs = [torch.randn(4, 4)]
`;

interface GraphData {
  nodes: Node[];
  edges: Edge[];
}

type Stage = "after" | "lowered" | "perf";

interface BenchmarkResult {
  eager_ms: number;
  compiled_ms: number;
  compile_time_ms: number;
  speedup: number;
  backend: string;
  eager_runs: number[];
  compiled_runs: number[];
}

const STAGE_LABELS: Record<Stage, { label: string; desc: string }> = {
  after:   { label: "Dynamo Export",  desc: "Shape-specialized, canonicalized" },
  lowered: { label: "Lowered IR",     desc: "AOT decomposed — what Triton compiles" },
  perf:    { label: "Performance",    desc: "Eager vs compiled latency" },
};

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
  const [before,  setBefore]  = useState<GraphData | null>(null);
  const [after,   setAfter]   = useState<GraphData | null>(null);
  const [lowered, setLowered] = useState<GraphData | null>(null);
  const [benchResult, setBenchResult] = useState<BenchmarkResult | null>(null);
  const [stage, setStage]     = useState<Stage>("after");
  const [error,   setError]   = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [benchLoading, setBenchLoading] = useState(false);
  const [device,  setDevice]  = useState<string | null>(null);

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
      if (!res.ok) { setError(data.detail ?? "Unknown error"); return; }

      setBefore(toReactFlow(data.before.nodes, data.before.edges));
      setAfter(data.after ? toReactFlow(data.after.nodes, data.after.edges) : null);
      setLowered(data.lowered ? toReactFlow(data.lowered.nodes, data.lowered.edges) : null);
    } catch (e: any) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }

  async function benchmark() {
    setBenchLoading(true);
    setError(null);
    try {
      const res = await fetch(`${API}/benchmark`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ code }),
      });
      const data = await res.json();
      if (!res.ok) { setError(data.detail ?? "Benchmark error"); return; }
      setBenchResult(data);
      setStage("perf");
    } catch (e: any) {
      setError(e.message);
    } finally {
      setBenchLoading(false);
    }
  }

  const stageData: Record<Exclude<Stage, "perf">, GraphData | null> = { after, lowered };
  const hasCompiled = after !== null || lowered !== null;

  return (
    <div style={styles.root}>
      <header style={styles.header}>
        <div>
          <h1 style={styles.title}>BorgCompiler</h1>
          <p style={styles.subtitle}>Write a PyTorch function → trace the compilation pipeline</p>
        </div>
        {device && (
          <span style={{
            padding: "4px 10px", borderRadius: 6, fontSize: 12, fontWeight: 600,
            background: device === "cuda" ? "#14532d" : "#1e293b",
            color: device === "cuda" ? "#4ade80" : "#64748b",
          }}>
            {device === "cuda" ? "⚡ GPU" : "CPU"}
          </span>
        )}
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
          <div style={{ display: "flex", gap: 8 }}>
            <button
              style={{ ...styles.button, flex: 1, opacity: loading ? 0.6 : 1 }}
              onClick={compile}
              disabled={loading}
            >
              {loading ? "Compiling…" : "▶ Compile"}
            </button>
            <button
              style={{ ...styles.button, flex: 1, background: "#0f766e", opacity: benchLoading ? 0.6 : 1 }}
              onClick={benchmark}
              disabled={benchLoading}
            >
              {benchLoading ? "Benchmarking…" : "⚡ Benchmark"}
            </button>
          </div>
          {error && <div style={styles.error}>{error}</div>}
        </div>

        {!before ? (
          <div style={{ ...styles.right, alignItems: "center", justifyContent: "center" }}>
            <div style={styles.placeholder}>Hit Compile to visualize the pipeline</div>
          </div>
        ) : (
          <div style={styles.right}>
            <div style={styles.pane}>
              <div style={styles.paneLabel}>
                <span style={styles.paneLabelTitle}>① FX Trace</span>
                <span style={styles.paneLabelDesc}>Symbolic — no shapes</span>
              </div>
              <GraphView key="before" nodes={before.nodes} edges={before.edges} />
            </div>

            <div style={styles.divider} />

            <div style={styles.pane}>
              <div style={styles.paneLabel}>
                {hasCompiled ? (
                  <>
                    <div style={{ display: "flex", gap: 6 }}>
                      {(Object.keys(STAGE_LABELS) as Stage[]).map((s, i) => (
                        <button
                          key={s}
                          onClick={() => setStage(s)}
                          style={{
                            ...styles.stageBtn,
                            background: stage === s ? "#334155" : "transparent",
                            color: stage === s ? "#e2e8f0" : "#64748b",
                          }}
                        >
                          {["② Dynamo", "③ Lowered IR", "④ Perf"][i]}
                        </button>
                      ))}
                    </div>
                    <span style={styles.paneLabelDesc}>{STAGE_LABELS[stage].desc}</span>
                  </>
                ) : (
                  <span style={styles.paneLabelDesc}>Add example_inputs to see compiled stages</span>
                )}
              </div>
              {stage === "perf" ? (
                benchResult ? (
                  <div style={{ flex: 1, overflow: "auto" }}>
                    <BenchmarkPanel result={benchResult} />
                  </div>
                ) : (
                  <div style={{ display: "flex", flex: 1, alignItems: "center", justifyContent: "center", color: "#334155", fontSize: 13 }}>
                    Click ⚡ Benchmark to measure eager vs compiled latency
                  </div>
                )
              ) : stageData[stage as Exclude<Stage, "perf">] ? (
                <GraphView key={stage} nodes={stageData[stage as Exclude<Stage, "perf">]!.nodes} edges={stageData[stage as Exclude<Stage, "perf">]!.edges} />
              ) : (
                <div style={{ display: "flex", flex: 1, alignItems: "center", justifyContent: "center", color: "#334155", fontSize: 13 }}>
                  No data for this stage
                </div>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

const styles: Record<string, React.CSSProperties> = {
  root:         { display: "flex", flexDirection: "column", height: "100vh", background: "#0f172a", color: "#e2e8f0", fontFamily: "system-ui, sans-serif" },
  header:       { display: "flex", alignItems: "center", justifyContent: "space-between", padding: "16px 24px", borderBottom: "1px solid #1e293b" },
  title:        { margin: 0, fontSize: 20, fontWeight: 700, color: "#f8fafc" },
  subtitle:     { margin: "4px 0 0", fontSize: 13, color: "#64748b" },
  body:         { display: "flex", flex: 1, overflow: "hidden" },
  left:         { display: "flex", flexDirection: "column", width: 400, borderRight: "1px solid #1e293b", padding: 16, gap: 12 },
  editorWrap:   { flex: 1, borderRadius: 8, overflow: "hidden", border: "1px solid #1e293b" },
  button:       { padding: "10px 20px", background: "#6366f1", color: "#fff", border: "none", borderRadius: 8, cursor: "pointer", fontSize: 14, fontWeight: 600 },
  error:        { padding: 12, background: "#450a0a", color: "#fca5a5", borderRadius: 8, fontSize: 13 },
  right:        { flex: 1, display: "flex", overflow: "hidden" },
  placeholder:  { color: "#334155", fontSize: 14 },
  pane:         { flex: 1, display: "flex", flexDirection: "column", overflow: "hidden", position: "relative" },
  paneLabel:    { display: "flex", alignItems: "center", justifyContent: "space-between", padding: "6px 14px", fontSize: 12, borderBottom: "1px solid #1e293b", background: "#0f172a", minHeight: 36 },
  paneLabelTitle: { fontWeight: 700, color: "#94a3b8" },
  paneLabelDesc:  { color: "#475569", fontSize: 11 },
  divider:      { width: 1, background: "#1e293b" },
  stageBtn:     { padding: "3px 10px", border: "1px solid #334155", borderRadius: 6, cursor: "pointer", fontSize: 12, fontWeight: 600 },
};
