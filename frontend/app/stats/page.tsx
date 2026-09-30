"use client";
import { useEffect, useState } from "react";
import { fetchStats } from "@/lib/api";

const METHODS = ["vector", "graph", "graph_reranked"] as const;
type Method = (typeof METHODS)[number];

const METHOD_LABEL: Record<Method, string> = {
  vector: "Vector RAG",
  graph: "Raw Graph RAG",
  graph_reranked: "Reranked Graph RAG",
};

const METHOD_COLOR: Record<Method, string> = {
  vector: "text-vector",
  graph: "text-graph",
  graph_reranked: "text-purple-600",
};

const METRICS = ["groundedness", "relevance", "completeness"] as const;
type Metric = (typeof METRICS)[number];
const METRIC_LABEL: Record<Metric, string> = {
  groundedness: "Groundedness",
  relevance: "Relevance",
  completeness: "Completeness",
};

const PAIRS = [
  ["graph", "graph_reranked"],
  ["vector", "graph"],
  ["vector", "graph_reranked"],
] as const;
const PAIR_LABEL: Record<string, string> = {
  "graph_vs_graph_reranked": "Raw Graph RAG vs Reranked Graph RAG",
  "vector_vs_graph": "Vector RAG vs Raw Graph RAG",
  "vector_vs_graph_reranked": "Vector RAG vs Reranked Graph RAG",
};

const SECTION_LABEL: Record<string, string> = {
  overall: "All Questions",
  single_hop: "Single-Hop",
  multi_hop: "Multi-Hop",
};

function Section({ id, section }: { id: string; section: any }) {
  if (!section || section.n === 0) {
    return (
      <div className="bg-white border rounded-md p-4">
        <h3 className="font-semibold mb-1">{SECTION_LABEL[id] ?? id}</h3>
        <p className="text-sm text-slate-500">No results yet.</p>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      <h2 className="text-lg font-semibold">
        {SECTION_LABEL[id] ?? id}{" "}
        <span className="text-sm font-normal text-slate-500">(n={section.n})</span>
      </h2>

      {/* Bootstrap CI table */}
      <div className="bg-white border rounded-md overflow-x-auto">
        <table className="w-full text-xs text-left">
          <thead className="bg-slate-50">
            <tr>
              <th className="p-2">Method</th>
              {METRICS.map((m) => (
                <th key={m} className="p-2">{METRIC_LABEL[m]} (mean, 95% CI)</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {METHODS.map((method) => (
              <tr key={method} className="border-t align-top">
                <td className={`p-2 font-medium ${METHOD_COLOR[method]}`}>
                  {METHOD_LABEL[method]}
                </td>
                {METRICS.map((metric) => {
                  const v = section.bootstrap_ci?.[method]?.[metric];
                  if (!v) return <td key={metric} className="p-2">—</td>;
                  return (
                    <td key={metric} className="p-2">
                      {v.mean.toFixed(2)}
                      <span className="text-slate-500"> [{v.ci_95_low.toFixed(2)}, {v.ci_95_high.toFixed(2)}]</span>
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Wilcoxon table */}
      <div className="bg-white border rounded-md overflow-x-auto">
        <table className="w-full text-xs text-left">
          <thead className="bg-slate-50">
            <tr>
              <th className="p-2">Paired comparison</th>
              {METRICS.map((m) => (
                <th key={m} className="p-2">{METRIC_LABEL[m]} (p-value)</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {PAIRS.map(([a, b]) => {
              const key = `${a}_vs_${b}`;
              const row = section.wilcoxon?.[key];
              if (!row) return null;
              return (
                <tr key={key} className="border-t align-top">
                  <td className="p-2 font-medium">{PAIR_LABEL[key] ?? key}</td>
                  {METRICS.map((metric) => {
                    const v = row[metric];
                    if (!v || v.p_value === null) {
                      return (
                        <td key={metric} className="p-2 text-slate-400">
                          {v?.note ?? "—"}
                        </td>
                      );
                    }
                    const sig = v.p_value < 0.05;
                    return (
                      <td
                        key={metric}
                        className={`p-2 ${sig ? "text-green-700 font-semibold" : "text-slate-600"}`}
                      >
                        p={v.p_value.toFixed(4)}
                        {sig ? " *" : ""}
                      </td>
                    );
                  })}
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}

export default function StatsPage() {
  const [data, setData] = useState<Record<string, any> | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetchStats()
      .then(setData)
      .catch((e) => setError(e.message));
  }, []);

  if (error) {
    return (
      <div className="space-y-2">
        <h1 className="text-xl font-semibold">Statistical Analysis</h1>
        <p className="text-sm text-red-600">{error}</p>
        <p className="text-sm text-slate-500">
          Run{" "}
          <code className="bg-slate-100 px-1 rounded">python scripts/06_statistical_analysis.py</code>{" "}
          in the engine folder (after 04_run_evaluation.py), then reload this page.
        </p>
      </div>
    );
  }
  if (!data) {
    return <p className="text-sm text-slate-500">Loading…</p>;
  }

  return (
    <div className="space-y-8">
      <div>
        <h1 className="text-xl font-semibold">Statistical Analysis</h1>
        <p className="text-sm text-slate-500 mt-1">
          Bootstrap 95% confidence intervals (10,000 resamples) on mean judge scores, and
          paired Wilcoxon signed-rank tests between methods on the same questions.
          <span className="font-semibold text-green-700"> *</span> marks p &lt; 0.05.
          With small n, "not significant" means not yet distinguishable from noise —
          not "no effect."
        </p>
      </div>

      <Section id="overall" section={data.overall} />
      {data.single_hop && <Section id="single_hop" section={data.single_hop} />}
      {data.multi_hop && <Section id="multi_hop" section={data.multi_hop} />}
    </div>
  );
}
