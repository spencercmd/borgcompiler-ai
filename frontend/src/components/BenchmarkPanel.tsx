import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
  Cell,
} from "recharts";

interface BenchmarkResult {
  eager_ms: number;
  compiled_ms: number;
  compile_time_ms: number;
  speedup: number;
  backend: string;
  eager_runs: number[];
  compiled_runs: number[];
}

interface Props {
  result: BenchmarkResult;
}

const EAGER_COLOR = "#6366f1";
const COMPILED_COLOR = "#10b981";
const SLOWER_COLOR = "#f59e0b";

export function BenchmarkPanel({ result }: Props) {
  const { eager_ms, compiled_ms, compile_time_ms, speedup, backend, eager_runs, compiled_runs } = result;
  const isFaster = speedup >= 1;
  const compiledColor = isFaster ? COMPILED_COLOR : SLOWER_COLOR;

  const barData = [
    { name: "Eager", ms: eager_ms, fill: EAGER_COLOR },
    { name: "Compiled", ms: compiled_ms, fill: compiledColor },
  ];

  // Build scatter data for both distributions side-by-side
  const maxLen = Math.max(eager_runs.length, compiled_runs.length);
  const distributionData = Array.from({ length: maxLen }, (_, i) => ({
    i,
    eager: eager_runs[i] ?? null,
    compiled: compiled_runs[i] ?? null,
  }));

  return (
    <div style={{ padding: "16px 20px", color: "#e2e8f0", display: "flex", flexDirection: "column", gap: 20 }}>
      {/* Summary row */}
      <div style={{ display: "flex", gap: 16, flexWrap: "wrap" }}>
        <StatCard label="Eager (median)" value={`${eager_ms} ms`} color={EAGER_COLOR} />
        <StatCard label="Compiled (median)" value={`${compiled_ms} ms`} color={compiledColor} />
        <StatCard label="Compile time (1×)" value={`${compile_time_ms} ms`} color="#94a3b8" />
        <StatCard
          label="Speedup"
          value={`${speedup}×`}
          color={isFaster ? COMPILED_COLOR : SLOWER_COLOR}
          note={isFaster ? "faster" : "slower"}
        />
        <StatCard label="Backend" value={backend} color="#94a3b8" />
      </div>

      {!isFaster && (
        <div style={{
          background: "#78350f22",
          border: "1px solid #f59e0b55",
          borderRadius: 6,
          padding: "8px 12px",
          fontSize: 13,
          color: "#fcd34d",
        }}>
          Compiled is slower here — small tensors + AOT tracing overhead outweigh any graph optimization.
          On GPU with Triton kernels, this typically inverts to 3–5× faster for large tensors.
        </div>
      )}

      {/* Median bar chart */}
      <div>
        <SectionLabel>Median latency (ms)</SectionLabel>
        <ResponsiveContainer width="100%" height={160}>
          <BarChart data={barData} margin={{ top: 4, right: 20, bottom: 0, left: 0 }}>
            <XAxis dataKey="name" tick={{ fill: "#94a3b8", fontSize: 12 }} axisLine={false} tickLine={false} />
            <YAxis tick={{ fill: "#94a3b8", fontSize: 11 }} axisLine={false} tickLine={false} />
            <Tooltip
              contentStyle={{ background: "#1e293b", border: "1px solid #334155", borderRadius: 6 }}
              labelStyle={{ color: "#e2e8f0" }}
              formatter={(v) => [`${v} ms`, ""]}
            />
            <Bar dataKey="ms" radius={[4, 4, 0, 0]}>
              {barData.map((entry, i) => (
                <Cell key={i} fill={entry.fill} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* Per-run distribution */}
      <div>
        <SectionLabel>Per-run latency — first 50 iterations (ms)</SectionLabel>
        <ResponsiveContainer width="100%" height={160}>
          <BarChart data={distributionData} margin={{ top: 4, right: 20, bottom: 0, left: 0 }}>
            <XAxis dataKey="i" hide />
            <YAxis tick={{ fill: "#94a3b8", fontSize: 11 }} axisLine={false} tickLine={false} />
            <Tooltip
              contentStyle={{ background: "#1e293b", border: "1px solid #334155", borderRadius: 6 }}
              formatter={(v, name) => [`${v} ms`, name as string]}
            />
            <ReferenceLine y={eager_ms} stroke={EAGER_COLOR} strokeDasharray="3 3" label={{ value: "eager p50", fill: EAGER_COLOR, fontSize: 11 }} />
            <ReferenceLine y={compiled_ms} stroke={compiledColor} strokeDasharray="3 3" label={{ value: "compiled p50", fill: compiledColor, fontSize: 11 }} />
            <Bar dataKey="eager" fill={EAGER_COLOR} opacity={0.7} radius={[2, 2, 0, 0]} />
            <Bar dataKey="compiled" fill={compiledColor} opacity={0.7} radius={[2, 2, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

function StatCard({ label, value, color, note }: { label: string; value: string; color: string; note?: string }) {
  return (
    <div style={{
      background: "#1e293b",
      border: "1px solid #334155",
      borderRadius: 8,
      padding: "10px 16px",
      minWidth: 110,
    }}>
      <div style={{ fontSize: 11, color: "#64748b", marginBottom: 4 }}>{label}</div>
      <div style={{ fontSize: 20, fontWeight: 700, color, fontFamily: "monospace" }}>{value}</div>
      {note && <div style={{ fontSize: 11, color }}>{note}</div>}
    </div>
  );
}

function SectionLabel({ children }: { children: React.ReactNode }) {
  return (
    <div style={{ fontSize: 12, color: "#64748b", marginBottom: 8, textTransform: "uppercase", letterSpacing: "0.05em" }}>
      {children}
    </div>
  );
}
