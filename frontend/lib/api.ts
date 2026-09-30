const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export interface RetrievedContext {
  chunk_id: string;
  text: string;
  score: number;
  source?: string;
}

export interface RAGAnswer {
  method: "vector" | "graph" | "graph_reranked";
  question: string;
  answer: string;
  contexts: RetrievedContext[];
  latency_seconds: number;
  extra: Record<string, unknown>;
}

export interface JudgeScore {
  groundedness: number;
  relevance: number;
  completeness: number;
  rationale: string;
}

export interface CompareResult {
  question: string;
  vector_answer: RAGAnswer;
  graph_answer: RAGAnswer;
  graph_reranked_answer: RAGAnswer;
  vector_judge: JudgeScore;
  graph_judge: JudgeScore;
  graph_reranked_judge: JudgeScore;
}

export interface GraphOverview {
  num_nodes: number;
  num_edges: number;
  node_types: Record<string, number>;
  top_nodes_by_degree: { name: string; degree: number }[];
}

export interface GraphNode {
  id: string;
  name: string;
  type: string;
  chunk_count: number;
}

export interface GraphEdge {
  source: string;
  target: string;
  relation: string;
}

export interface GraphData {
  nodes: GraphNode[];
  edges: GraphEdge[];
  truncated: boolean;
  total_nodes: number;
  total_edges: number;
}

// Shape written by scripts/06_statistical_analysis.py to stats_results.json,
// served as-is by /api/stats. Kept loose (any) for the nested per-metric
// bootstrap/wilcoxon objects since their shape is already self-describing
// JSON and not worth re-typing field-by-field here.
export interface StatsSection {
  label: string;
  n: number;
  bootstrap_ci: Record<string, Record<string, {
    mean: number;
    ci_95_low: number;
    ci_95_high: number;
    n: number;
  }>>;
  wilcoxon: Record<string, Record<string, {
    statistic: number | null;
    p_value: number | null;
    note?: string;
  }>>;
}

export interface StatsResults {
  overall: StatsSection;
  single_hop?: StatsSection;
  multi_hop?: StatsSection;
  [key: string]: StatsSection | undefined;
}

async function postJSON<T>(path: string, body: unknown): Promise<T> {
  const res = await fetch(`${API_URL}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const detail = await res.text();
    throw new Error(`Request to ${path} failed (${res.status}): ${detail}`);
  }
  return res.json();
}

async function getJSON<T>(path: string): Promise<T> {
  const res = await fetch(`${API_URL}${path}`, { cache: "no-store" });
  if (!res.ok) {
    const detail = await res.text();
    throw new Error(`Request to ${path} failed (${res.status}): ${detail}`);
  }
  return res.json();
}

export function queryVector(question: string, top_k = 5) {
  return postJSON<RAGAnswer>("/api/vector/query", { question, top_k });
}

export function queryGraph(question: string, top_k = 5) {
  return postJSON<RAGAnswer>("/api/graph/query", { question, top_k });
}

export function queryGraphReranked(question: string, top_k = 5) {
  return postJSON<RAGAnswer>("/api/graph_reranked/query", { question, top_k });
}

export function compareBoth(question: string, top_k = 5) {
  return postJSON<CompareResult>("/api/compare", { question, top_k });
}

export async function fetchHistory(): Promise<{
  results: CompareResult[];
  summary: any;
  summary_by_category?: Record<string, any>;
}> {
  const res = await fetch(`${API_URL}/api/compare/history`, { cache: "no-store" });
  if (!res.ok) throw new Error("Failed to load evaluation history");
  return res.json();
}

export function fetchGraphOverview() {
  return getJSON<GraphOverview>("/api/graph/overview");
}

export function fetchGraphData(limitNodes = 150) {
  return getJSON<GraphData>(`/api/graph/data?limit_nodes=${limitNodes}`);
}

export async function fetchStats(): Promise<StatsResults> {
  // Deliberately NOT using getJSON here: a 404 means "haven't run
  // 06_statistical_analysis.py yet", which the /stats page wants to show
  // as a helpful instruction, not an opaque thrown error. Any other
  // non-OK status still throws.
  const res = await fetch(`${API_URL}/api/stats`, { cache: "no-store" });
  if (!res.ok) {
    let message = `Request to /api/stats failed (${res.status})`;
    try {
      const body = await res.json();
      if (body?.detail) message = body.detail;
    } catch {
      // response wasn't JSON — fall back to the generic message above
    }
    throw new Error(message);
  }
  return res.json();
}
