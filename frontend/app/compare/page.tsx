"use client";
import { useEffect, useState } from "react";
import { fetchHistory, CompareResult } from "@/lib/api";
import OverallResultsChart from "@/components/OverallResultsChart";

const METHODS = ["vector", "graph", "graph_reranked"] as const;
type Method = (typeof METHODS)[number];

const METHOD_LABEL: Record<Method, string> = {
  vector: "Vector RAG",
  graph: "Raw Graph RAG",
  graph_reranked: "Reranked Graph RAG",
};

// text-vector / text-graph are presumably custom colors defined in your
// tailwind config / globals.css already. I'm using a built-in Tailwind
// color (purple) for the new method so this renders with zero config
// changes — swap "text-purple-600" for a custom "text-graph-reranked"
// class if you want it to match your existing design tokens.
const METHOD_COLOR: Record<Method, string> = {
  vector: "text-vector",
  graph: "text-graph",
  graph_reranked: "text-purple-600",
};

export default function ComparePage() {
  const [data, setData] = useState<{ results: CompareResult[]; summary: any } | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetchHistory().then(setData).catch((e) => setError(e.message));
  }, []);

  if (error) {
    return <p className="text-sm text-red-600">{error}</p>;
  }
  if (!data) {
    return <p className="text-sm text-slate-500">Loading…</p>;
  }
  if (!data.results.length) {
    return (
      <p className="text-sm text-slate-500">
        No evaluation results yet. Run{" "}
        <code className="bg-slate-100 px-1 rounded">python scripts/04_run_evaluation.py</code>{" "}
        in the engine folder, then reload this page.
      </p>
    );
  }

  const { summary } = data;

  return (
    <div className="space-y-6">
      <h1 className="text-xl font-semibold">Batch Evaluation Dashboard</h1>

      <OverallResultsChart summary={summary} />

      <div className="grid md:grid-cols-3 gap-4">
        {METHODS.map((method) => (
          <div key={method} className="bg-white border rounded-md p-4">
            <h3 className={`font-semibold mb-2 ${METHOD_COLOR[method]}`}>
              {METHOD_LABEL[method]} — averages
            </h3>
            <ul className="text-sm space-y-1">
              <li>Groundedness: {summary[method].avg_groundedness}/5</li>
              <li>Relevance: {summary[method].avg_relevance}/5</li>
              <li>Completeness: {summary[method].avg_completeness}/5</li>
              <li>Avg latency: {summary[method].avg_latency_seconds}s</li>
            </ul>
          </div>
        ))}
      </div>

      {data.summary_by_category && (
        <div className="space-y-4">
          <h2 className="text-lg font-semibold">By question type</h2>
          <div className="grid md:grid-cols-2 gap-4">
            {Object.entries(data.summary_by_category as Record<string, any>).map(([cat, catSummary]) => (
              <div key={cat} className="bg-white border rounded-md p-4">
                <h3 className="font-semibold mb-2 capitalize">{cat.replace("_", "-")}</h3>
                <table className="w-full text-xs text-left">
                  <thead>
                    <tr>
                      <th className="p-1">Method</th>
                      <th className="p-1">G</th>
                      <th className="p-1">R</th>
                      <th className="p-1">C</th>
                    </tr>
                  </thead>
                  <tbody>
                    {METHODS.map((method) => (
                      <tr key={method} className="border-t">
                        <td className={`p-1 ${METHOD_COLOR[method]}`}>{METHOD_LABEL[method]}</td>
                        <td className="p-1">{catSummary[method].avg_groundedness}</td>
                        <td className="p-1">{catSummary[method].avg_relevance}</td>
                        <td className="p-1">{catSummary[method].avg_completeness}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ))}
          </div>
        </div>
      )}

      <div className="bg-white border rounded-md overflow-x-auto">
        <table className="w-full text-xs text-left">
          <thead className="bg-slate-50">
            <tr>
              <th className="p-2">Question</th>
              {METHODS.map((method) => (
                <th key={method} className={`p-2 ${METHOD_COLOR[method]}`}>
                  {METHOD_LABEL[method]} (G/R/C)
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {data.results.map((r: any, i: number) => (
              <tr key={i} className="border-t align-top">
                <td className="p-2 max-w-xs">{r.question}</td>
                {METHODS.map((method) => {
                  const judge = r[`${method}_judge`];
                  return (
                    <td key={method} className="p-2">
                      {judge ? `${judge.groundedness}/${judge.relevance}/${judge.completeness}` : "—"}
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
