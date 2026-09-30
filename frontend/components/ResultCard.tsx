import { RAGAnswer } from "@/lib/api";

type Accent = "vector" | "graph" | "graph_reranked";

const BORDER_COLOR: Record<Accent, string> = {
  vector: "border-vector",
  graph: "border-graph",
  graph_reranked: "border-purple-600",
};

const LABEL: Record<Accent, string> = {
  vector: "Vector RAG",
  graph: "Raw Graph RAG",
  graph_reranked: "Reranked Graph RAG",
};

export default function ResultCard({ answer, accent }: { answer: RAGAnswer; accent: Accent }) {
  const color = BORDER_COLOR[accent];
  const label = LABEL[accent];
  const isGraphMethod = accent === "graph" || accent === "graph_reranked";

  return (
    <div className={`border-l-4 ${color} bg-white rounded-md shadow-sm p-4`}>
      <div className="flex justify-between items-center mb-2">
        <h3 className="font-semibold">{label}</h3>
        <span className="text-xs text-slate-500">{answer.latency_seconds}s</span>
      </div>
      <p className="text-sm whitespace-pre-wrap mb-3">{answer.answer}</p>

      {isGraphMethod && Array.isArray((answer.extra as any)?.matched_entities) && (
        <div className="text-xs text-slate-500 mb-1">
          Matched entities: {(answer.extra as any).matched_entities.join(", ") || "none"}
        </div>
      )}

      {accent === "graph_reranked" && (answer.extra as any)?.candidate_pool_size != null && (
        <div className="text-xs text-slate-500 mb-2">
          Reranked from a pool of {(answer.extra as any).candidate_pool_size} candidate chunks
        </div>
      )}

      <details className="text-xs text-slate-600">
        <summary className="cursor-pointer select-none">
          {answer.contexts.length} retrieved passages
        </summary>
        <ul className="mt-2 space-y-2">
          {answer.contexts.map((c, i) => (
            <li key={c.chunk_id} className="border rounded p-2 bg-slate-50">
              <div className="font-mono text-[10px] text-slate-400 mb-1">
                [{i + 1}] {c.source} · score {c.score.toFixed(2)}
              </div>
              {c.text.slice(0, 300)}
              {c.text.length > 300 ? "…" : ""}
            </li>
          ))}
        </ul>
      </details>
    </div>
  );
}
