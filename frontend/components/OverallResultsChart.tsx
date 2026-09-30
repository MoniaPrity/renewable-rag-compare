"use client";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from "recharts";

const METHODS = ["vector", "graph", "graph_reranked"] as const;
type Method = (typeof METHODS)[number];

const METHOD_LABEL: Record<Method, string> = {
  vector: "Vector RAG",
  graph: "Raw Graph RAG",
  graph_reranked: "Reranked Graph RAG",
};

// recharts needs raw hex values (it can't resolve Tailwind class names as
// fill colors). These match tailwind.config.ts's theme.extend.colors
// exactly: vector = #2563eb, graph = #16a34a. graph_reranked isn't a
// registered token in tailwind.config.ts — page.tsx/ResultCard.tsx apply
// it via the built-in Tailwind "purple-600" class, so #9333ea (purple-600's
// hex) is used here to match.
const METHOD_FILL: Record<Method, string> = {
  vector: "#2563eb",
  graph: "#16a34a",
  graph_reranked: "#9333ea",
};

type MethodSummary = {
  avg_groundedness: number;
  avg_relevance: number;
  avg_completeness: number;
  avg_latency_seconds: number;
};

type Summary = Record<Method, MethodSummary>;

const METRICS: { key: keyof MethodSummary; label: string }[] = [
  { key: "avg_groundedness", label: "Groundedness" },
  { key: "avg_relevance", label: "Relevance" },
  { key: "avg_completeness", label: "Completeness" },
];

export default function OverallResultsChart({ summary }: { summary: Summary }) {
  // Reshape { vector: {...}, graph: {...}, graph_reranked: {...} } into
  // recharts' row-per-category shape:
  // [{ metric: "Groundedness", vector: 4.88, graph: 4.28, graph_reranked: 4.67 }, ...]
  const data = METRICS.map(({ key, label }) => {
    const row: Record<string, string | number> = { metric: label };
    for (const method of METHODS) {
      row[method] = summary[method]?.[key] ?? 0;
    }
    return row;
  });

  return (
    <div className="bg-white border rounded-md p-4">
      <h3 className="font-semibold mb-4">Overall results by method</h3>
      <ResponsiveContainer width="100%" height={320}>
        <BarChart data={data} margin={{ top: 8, right: 16, left: 0, bottom: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
          <XAxis dataKey="metric" tick={{ fontSize: 12 }} />
          <YAxis domain={[0, 5]} tick={{ fontSize: 12 }} />
          <Tooltip
            formatter={(value: number) => value.toFixed(2)}
            contentStyle={{ fontSize: 12 }}
          />
          <Legend
            formatter={(value: string) => METHOD_LABEL[value as Method] ?? value}
            wrapperStyle={{ fontSize: 12 }}
          />
          {METHODS.map((method) => (
            <Bar
              key={method}
              dataKey={method}
              name={method}
              fill={METHOD_FILL[method]}
              radius={[4, 4, 0, 0]}
            />
          ))}
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
