"use client";
import { useState } from "react";

export default function QueryForm({
  onSubmit,
  loading,
}: {
  onSubmit: (question: string) => void;
  loading: boolean;
}) {
  const [question, setQuestion] = useState("");

  return (
    <form
      onSubmit={(e) => {
        e.preventDefault();
        if (question.trim().length > 3) onSubmit(question.trim());
      }}
      className="flex gap-2"
    >
      <input
        value={question}
        onChange={(e) => setQuestion(e.target.value)}
        placeholder="e.g. How does grid energy storage support renewable integration?"
        className="flex-1 border rounded-md px-3 py-2 text-sm"
      />
      <button
        type="submit"
        disabled={loading}
        className="bg-slate-900 text-white text-sm px-4 py-2 rounded-md disabled:opacity-50"
      >
        {loading ? "Thinking…" : "Ask all"}
      </button>
    </form>
  );
}
