import dagre from "@dagrejs/dagre";
import type { Node, Edge } from "@xyflow/react";

const NODE_W = 160;
const NODE_H = 48;

// Takes raw nodes/edges (no positions) and returns nodes with x/y assigned
// by Dagre's layered graph layout algorithm (Sugiyama style, top-to-bottom).
export function applyLayout(nodes: Node[], edges: Edge[]): Node[] {
  const g = new dagre.graphlib.Graph();
  // rankdir: TB = top-to-bottom flow (inputs at top, output at bottom)
  g.setGraph({ rankdir: "TB", nodesep: 60, ranksep: 60 });
  g.setDefaultEdgeLabel(() => ({}));

  nodes.forEach((n) => g.setNode(n.id, { width: NODE_W, height: NODE_H }));
  edges.forEach((e) => g.setEdge(e.source, e.target));

  dagre.layout(g);

  // Dagre gives us center coordinates; React Flow uses top-left corner.
  return nodes.map((n) => {
    const { x, y } = g.node(n.id);
    return { ...n, position: { x: x - NODE_W / 2, y: y - NODE_H / 2 } };
  });
}
