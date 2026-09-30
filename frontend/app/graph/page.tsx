"use client";

import { useEffect, useState, useCallback } from "react";
import dynamic from "next/dynamic";
import { fetchGraphOverview, fetchGraphData, GraphOverview, GraphData, GraphNode } from "@/lib/api";

// react-force-graph-2d renders to <canvas> and reads `window` at import
// time, so it must never run during Next.js server-side rendering.
const ForceGraph2D = dynamic(() => import("react-force-graph-2d"), { ssr: false });

const TYPE_COLORS: Record<string, string> = {
  technology: "#2563eb",
  organization: "#16a34a",
  material: "#ca8a04",
  location: "#dc2626",
  metric: "#7c3aed",
  policy: "#0891b2",
  unknown: "#64748b",
};

function colorForType(type: string): string {
  return TYPE_COLORS[type?.toLowerCase()] || TYPE_COLORS.unknown;
}

export default function GraphExplorerPage() {
  const [overview, setOverview] = useState<GraphOverview | null>(null);
  const [graphData, setGraphData] = useState<GraphData | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [selectedNode, setSelectedNode] = useState<GraphNode | null>(null);

  useEffect(() => {
    Promise.all([fetchGraphOverview(), fetchGraphData()])
      .then(([ov, data]) => {
        setOverview(ov);
        setGraphData(data);
      })
      .catch((e) => setError(e.message));
  }, []);

  const handleNodeClick = useCallback((node: any) => {
    setSelectedNode(node as GraphNode);
  }, []);

  if (error) {
    return (
      <p className="text-sm text-red-600">
        {error}. Make sure scripts/03_build_graph_index.py has been run at least once.
      </p>
    );
  }
  if (!overview || !graphData) {
    return <p className="text-sm text-slate-500">Loading graph…</p>;
  }

  const fgData = {
    nodes: graphData.nodes.map((n) => ({ ...n, val: 1 + n.chunk_count })),
    links: graphData.edges.map((e) => ({ source: e.source, target: e.target, relation: e.relation })),
  };

  return (
    <div className="space-y-4">
      <div>
        <h1 className="text-xl font-semibold mb-1">Knowledge Graph Explorer</h1>
        <p className="text-sm text-slate-500">
          {overview.num_nodes} entities, {overview.num_edges} relationships extracted from your corpus.
          {graphData.truncated && (
            <>
              {" "}
              Showing the {graphData.nodes.length} most-connected entities out of {graphData.total_nodes}{" "}
              for readability.
            </>
          )}
        </p>
      </div>

      <div className="grid md:grid-cols-4 gap-4">
        <div className="md:col-span-3 bg-white border rounded-md overflow-hidden" style={{ height: 600 }}>
          <ForceGraph2D
            graphData={fgData}
            nodeId="id"
            nodeLabel={(n: any) => `${n.name} (${n.type})`}
            nodeColor={(n: any) => colorForType(n.type)}
            nodeRelSize={4}
            linkLabel={(l: any) => l.relation}
            linkDirectionalArrowLength={4}
            linkDirectionalArrowRelPos={1}
            linkColor={() => "#cbd5e1"}
            onNodeClick={handleNodeClick}
            height={600}
          />
        </div>

        <div className="bg-white border rounded-md p-4 text-sm space-y-4">
          <div>
            <h3 className="font-semibold mb-2">Top entities by connections</h3>
            <ul className="space-y-1">
              {overview.top_nodes_by_degree.slice(0, 10).map((n) => (
                <li key={n.name} className="flex justify-between">
                  <span>{n.name}</span>
                  <span className="text-slate-400">{n.degree}</span>
                </li>
              ))}
            </ul>
          </div>

          <div>
            <h3 className="font-semibold mb-2">Entity types</h3>
            <ul className="space-y-1">
              {Object.entries(overview.node_types).map(([type, count]) => (
                <li key={type} className="flex items-center gap-2">
                  <span className="inline-block w-3 h-3 rounded-full" style={{ background: colorForType(type) }} />
                  <span className="capitalize">{type}</span>
                  <span className="text-slate-400 ml-auto">{count}</span>
                </li>
              ))}
            </ul>
          </div>

          {selectedNode && (
            <div>
              <h3 className="font-semibold mb-2">Selected entity</h3>
              <p className="font-medium">{selectedNode.name}</p>
              <p className="text-slate-500 capitalize">{selectedNode.type}</p>
              <p className="text-slate-500">Mentioned in {selectedNode.chunk_count} chunk(s)</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
