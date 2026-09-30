import time
from shared.schemas import CompareResult, JudgeScore
from app import vector_rag, graph_rag
from app.llm_client import generate_json

# Groq's free tier caps tokens-per-minute, not just requests-per-minute.
# compare_question() fires 6 calls back-to-back (3 answers + 3 judges);
# without pacing, that burst alone can exceed the per-minute budget even
# with a delay between separate questions. Adjust upward if you still see
# 429s; this is deliberately small since each call already takes a few
# seconds end-to-end.
INTRA_QUESTION_DELAY = 1.5

JUDGE_SYSTEM_PROMPT = """You are grading a RAG system's answer to a question about
renewable energy. Score 1 (poor) to 5 (excellent) on:
- groundedness: is the answer actually supported by the given context passages?
- relevance: does the answer address the actual question asked?
- completeness: does it cover the key points a good answer should have?

Return JSON: {"groundedness": int, "relevance": int, "completeness": int, "rationale": "one sentence"}
"""


def _judge(question: str, answer_text: str, context_texts: list[str]) -> JudgeScore:
    context_block = "\n\n".join(context_texts) or "(no context retrieved)"
    prompt = (
        f"Question: {question}\n\nContext passages given to the system:\n{context_block}"
        f"\n\nSystem's answer:\n{answer_text}"
    )
    data = generate_json(prompt, system=JUDGE_SYSTEM_PROMPT)
    # Defensive clamp: despite the prompt saying "1 to 5", the judge LLM
    # occasionally returns 0 (e.g. when it considers an answer a total
    # failure) or otherwise out-of-range ints. Clamp instead of letting a
    # single bad judge call crash the entire batch eval run.
    for field in ("groundedness", "relevance", "completeness"):
        if field in data:
            try:
                data[field] = max(1, min(5, int(data[field])))
            except (TypeError, ValueError):
                data[field] = 1
    return JudgeScore(**data)


def compare_question(question: str, top_k: int = 5) -> CompareResult:
    vector_answer = vector_rag.answer(question, top_k=top_k)
    time.sleep(INTRA_QUESTION_DELAY)
    graph_answer = graph_rag.answer(question, top_k=top_k)
    time.sleep(INTRA_QUESTION_DELAY)
    # Same 2-hop candidate pool as graph_answer, but chunk selection is
    # reranked by a local Cross-Encoder instead of an unordered-set slice.
    graph_reranked_answer = graph_rag.answer(question, top_k=top_k, use_reranker=True)
    time.sleep(INTRA_QUESTION_DELAY)

    vector_judge = _judge(question, vector_answer.answer, [c.text for c in vector_answer.contexts])
    time.sleep(INTRA_QUESTION_DELAY)
    graph_judge = _judge(question, graph_answer.answer, [c.text for c in graph_answer.contexts])
    time.sleep(INTRA_QUESTION_DELAY)
    graph_reranked_judge = _judge(
        question, graph_reranked_answer.answer, [c.text for c in graph_reranked_answer.contexts]
    )

    return CompareResult(
        question=question,
        vector_answer=vector_answer,
        graph_answer=graph_answer,
        graph_reranked_answer=graph_reranked_answer,
        vector_judge=vector_judge,
        graph_judge=graph_judge,
        graph_reranked_judge=graph_reranked_judge,
    )
