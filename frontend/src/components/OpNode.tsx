import { Handle, Position } from "@xyflow/react";

const OP_COLORS: Record<string, string> = {
  placeholder: "#6366f1",
  call_function: "#0ea5e9",
  call_method: "#0ea5e9",
  output: "#10b981",
};

export function OpNode({ data }: { data: any }) {
  const bg = OP_COLORS[data.op] ?? "#334155";
  return (
    <div
      style={{
        background: bg,
        borderRadius: 8,
        padding: "8px 14px",
        minWidth: 130,
        textAlign: "center",
        color: "#fff",
        fontSize: 12,
        fontFamily: "system-ui, sans-serif",
      }}
    >
      <Handle type="target" position={Position.Top} style={{ background: "rgba(255,255,255,0.4)", border: "none" }} />
      <div style={{ fontWeight: 600 }}>{data.label}</div>
      {data.shape && (
        <div style={{ fontSize: 10, opacity: 0.75, marginTop: 3 }}>{data.shape}</div>
      )}
      <Handle type="source" position={Position.Bottom} style={{ background: "rgba(255,255,255,0.4)", border: "none" }} />
    </div>
  );
}
