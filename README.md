# Renewable Energy: Vector RAG vs Graph RAG vs Reranked Graph RAG

CSE499B capstone project. Research question: **Does cross-encoder reranking
close the multi-hop performance gap between knowledge-graph retrieval and
vector RAG in renewable energy domain QA?**

Three retrieval methods are compared over a Wikipedia-based renewable-energy
corpus:

1. **Vector RAG**: ChromaDB + `all-MiniLM-L6-v2` embeddings, top-K chunks.
2. **Raw Graph RAG**: NetworkX knowledge graph (LLM-built), 2-hop ego-graph
   traversal from fuzzy-matched entities, no ranking (deliberate baseline).
3. **Reranked Graph RAG**: same traversal, but the full candidate pool is
   scored by a cross-encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`) before
   keeping the top K.

Results are validated with bootstrap 95% confidence intervals and paired
Wilcoxon tests, split by single-hop and multi-hop questions.

Embeddings, vector search, graph traversal, and reranking all run locally on
CPU. Only the LLM calls (answer generation, graph building, judging) go to the
Groq API.

```
renewable-rag-compare/
├── engine/              FastAPI backend (RAG pipelines + evaluation)
│   ├── app/             API + pipeline code
│   ├── scripts/         offline scripts (corpus, indexes, evaluation, stats)
│   └── data/            generated corpus/indexes (not committed)
├── packages/shared/     shared Pydantic schemas (installed into engine's venv)
└── frontend/            Next.js 15 UI (query, graph explorer, dashboards)
```

---
## 0. Requirements

| Tool | Version |
|---|---|
| Python | 3.12 |
| Node.js | 20 LTS or newer (https://nodejs.org) |
| Groq API key | free tier at https://console.groq.com |

No Docker, database server, or GPU needed.

---
## 1. Get the project

```powershell
git clone https://github.com/MoniaPrity/renewable-rag-compare.git
cd renewable-rag-compare
```

Open the folder in VS Code and use the built-in PowerShell terminal (Ctrl+`).

---
## 2. Backend setup

```powershell
cd engine
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
```

If PowerShell blocks activation, run once:
`Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned`

Open `engine\.env` and set your **own** key (never commit this file):

```
GROQ_API_KEY=your_key_here
GROQ_MODEL=<same model as in .env.example>
```

### Build the data (run in order)

```powershell
python scripts/01_collect_corpus.py          # Wikipedia category-tree crawl
python scripts/02_build_vector_index.py      # chunks + Chroma index
python scripts/03_build_graph_index.py       # LLM entity/relation extraction
python scripts/07_filter_graph.py --dry-run  # sanity check, then run without --dry-run
```

Step 01 crawls the Wikipedia category tree from renewable-energy root
categories (depth 2, max 300 pages). Step 03 calls the LLM once per chunk, so
it is the slowest step. Use `LIMIT = 20` at the top of the script while
testing.

### Run the evaluation

```powershell
python scripts/05_generate_questions.py --single-hop-target 20 --multi-hop-target 20
python scripts/04_run_evaluation.py --delay 1.0   # resumable if interrupted
python scripts/06_statistical_analysis.py
```

Outputs go to `engine/data/processed/` (`eval_results.json`,
`stats_results.json`).

### Start the API

```powershell
uvicorn app.main:app --reload --port 8000
```

Swagger docs: http://localhost:8000/docs. The active Groq model is logged at
startup.

---
## 3. Frontend setup

In a **new terminal**:

```powershell
cd frontend
npm install
copy .env.local.example .env.local
npm run dev
```

Re-run `npm install` after any `package.json` change, otherwise Next.js throws
"Module not found" errors.

| URL | Page |
|---|---|
| http://localhost:3000 | Ask a question, compare the three methods |
| http://localhost:3000/graph | Knowledge graph explorer |
| http://localhost:3000/stats | Statistical analysis |
| http://localhost:3000/compare | Evaluation dashboard with results chart |

---
## 4. Everyday workflow

```powershell
# Terminal 1 - backend
cd engine
.venv\Scripts\Activate.ps1
uvicorn app.main:app --reload --port 8000

# Terminal 2 - frontend
cd frontend
npm run dev
```

Re-run the scripts in `engine/scripts/` only if you change the corpus,
chunking, or graph.

---
## 5. Key result

On multi-hop questions, Reranked Graph RAG beats Raw Graph RAG on every metric
(groundedness / relevance / completeness: 5.00 / 5.00 / 5.00 vs
4.60 / 4.80 / 4.60), and the gap to Vector RAG is no longer statistically
significant on multi-hop questions. See `/stats` for the full analysis.

---
## 6. Troubleshooting

- **`GROQ_API_KEY is not set`**: copy `.env.example` to `.env` in `engine/` and
  add your key.
- **Frontend connection/CORS error**: make sure `uvicorn` is running on port
  8000.
- **`Module not found: Can't resolve 'recharts'`**: run `npm install` in
  `frontend/` and restart the dev server.
- **Groq rate-limit errors**: increase `--delay` in `04_run_evaluation.py`; the
  run is resumable.
- **Graph building is slow**: expected (one API call per chunk); use `LIMIT`
  while iterating.