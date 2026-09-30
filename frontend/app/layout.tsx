import type { Metadata } from "next";
import "./globals.css";
import Link from "next/link";

export const metadata: Metadata = {
  title: "Vector RAG vs Graph RAG — Renewable Energy",
  description: "Comparing Vector RAG and Graph RAG over a renewable energy corpus",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>
        <nav className="border-b bg-white px-6 py-4 flex gap-6 items-center">
          <span className="font-semibold">🌱 Renewable RAG Compare</span>
          <Link href="/" className="text-sm text-slate-600 hover:text-slate-900">
            Ask
          </Link>
          <Link href="/compare" className="text-sm text-slate-600 hover:text-slate-900">
            Evaluation Dashboard
          </Link>
          <Link href="/stats" className="text-sm text-slate-600 hover:text-slate-900">
            Statistics
          </Link>
          <Link href="/graph" className="text-sm text-slate-600 hover:text-slate-900">
            Knowledge Graph
          </Link>
        </nav>
        <main className="max-w-6xl mx-auto p-6">{children}</main>
      </body>
    </html>
  );
}
