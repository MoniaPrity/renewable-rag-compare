"use client";
import { useState } from "react";
import QueryForm from "@/components/QueryForm";
import ResultCard from "@/components/ResultCard";
import { compareBoth, CompareResult } from "@/lib/api";

const METHODS = ["vector", "graph", "graph_reranked"] as const;
type Method = (typeof METHODS)[number];

const METHOD_LABEL: Record<Method, string> = {
  vector: "Vector",
  graph: "Graph",
  graph_reranked: "Reranked Graph",
};

export default function HomePage() {
  const [result, setResult] = useState<CompareResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function handleAsk(question: string) {
    setLoading(true);
    setError(null);
    try {
      const res = await compareBoth(question);
      setResult(res);
    } catch (e: any) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-xl font-semibold mb-1">Ask a renewable-energy question</h1>
        <p className="text-sm text-slate-500 mb-4">
          Runs the same question through Vector RAG (ChromaDB + embeddings), raw Graph RAG
          (NetworkX knowledge graph), and Cross-Encoder-Reranked Graph RAG side by side.
        </p>
        <QueryForm onSubmit={handleAsk} loading={loading} />
      </div>

      {error && <p className="text-sm text-red-600">{error}</p>}

      {result && (
        <div className="grid md:grid-cols-3 gap-4">
          <ResultCard answer={result.vector_answer} accent="vector" />
          <ResultCard answer={result.graph_answer} accent="graph" />
          <ResultCard answer={result.graph_reranked_answer} accent="graph_reranked" />
        </div>
      )}

      {result && (
        <div className="bg-white border rounded-md p-4 text-sm">
          <h3 className="font-semibold mb-2">LLM-as-judge scores</h3>
          <table className="w-full text-left text-xs">
            <thead>
              <tr className="text-slate-500">
                <th className="py-1">Method</th>
                <th>Groundedness</th>
                <th>Relevance</th>
                <th>Completeness</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td className="py-1 text-vector font-medium">Vector</td>
                <td>{result.vector_judge.groundedness}/5</td>
                <td>{result.vector_judge.relevance}/5</td>
                <td>{result.vector_judge.completeness}/5</td>
              </tr>
              <tr>
                <td className="py-1 text-graph font-medium">Graph</td>
                <td>{result.graph_judge.groundedness}/5</td>
                <td>{result.graph_judge.relevance}/5</td>
                <td>{result.graph_judge.completeness}/5</td>
              </tr>
              <tr>
                <td className="py-1 text-purple-600 font-medium">Reranked Graph</td>
                <td>{result.graph_reranked_judge.groundedness}/5</td>
                <td>{result.graph_reranked_judge.relevance}/5</td>
                <td>{result.graph_reranked_judge.completeness}/5</td>
              </tr>
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
