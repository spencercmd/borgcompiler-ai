import { useMemo } from "react";
import { ReactFlow, Background, Controls, type Node, type Edge } from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import { applyLayout } from "../lib/layout";
import { OpNode } from "./OpNode";

const nodeTypes = { op: OpNode };

interface Props {
  nodes: Node[];
  edges: Edge[];
}

export function GraphView({ nodes, edges }: Props) {
  const laidOutNodes = useMemo(
    () => applyLayout(nodes, edges),
    [nodes, edges]
  );

  return (
    <div style={{ width: "100%", flex: 1, minHeight: 0 }}>
      <ReactFlow
        nodes={laidOutNodes}
        edges={edges}
        nodeTypes={nodeTypes}
        fitView
        nodesDraggable={false}
      >
        <Background color="#1e293b" gap={20} />
        <Controls />
      </ReactFlow>
    </div>
  );
}
