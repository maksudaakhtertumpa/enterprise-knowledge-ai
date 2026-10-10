// Pure presentation helpers (no React, no network); unit-tested in format.test.ts.
//
// Security rule for the whole UI: document text, quotes, filenames and model
// answers are untrusted. They are only ever rendered as React text nodes (which
// React escapes). Highlighting is done by splitting text into segments here and
// wrapping some segments in <mark> elements, never by building HTML strings.

import type { Citation, SourceRef } from "@/lib/types";

export type Span = [start: number, end: number];
export interface Segment {
  text: string;
  hit: boolean;
}

export function truncate(text: string | null | undefined, limit: number): string {
  const flat = (text ?? "").split(/\s+/).filter(Boolean).join(" ");
  return flat.length <= limit ? flat : flat.slice(0, limit - 1).trimEnd() + "…";
}

export function sourceLabel(source: Partial<SourceRef> | null | undefined): string {
  if (!source) return "unknown source";
  const parts = [source.filename || "untitled"];
  const { page_start: start, page_end: end } = source;
  if (start) parts.push(!end || end === start ? `p. ${start}` : `pp. ${start}–${end}`);
  if (source.heading) parts.push(truncate(source.heading, 60));
  return parts.join(" · ");
}

// ------------------------------------------------------------------ highlighting
export function mergeSpans(spans: Span[]): Span[] {
  const sorted = spans.filter(([s, e]) => e > s).sort((a, b) => a[0] - b[0] || a[1] - b[1]);
  const out: Span[] = [];
  for (const [s, e] of sorted) {
    const last = out[out.length - 1];
    if (last && s <= last[1]) last[1] = Math.max(last[1], e);
    else out.push([s, e]);
  }
  return out;
}

/** Split text into plain and highlighted segments (overlapping spans merged). */
export function segments(text: string, spans: Span[]): Segment[] {
  const out: Segment[] = [];
  let pos = 0;
  for (const [rawStart, rawEnd] of mergeSpans(spans)) {
    const start = Math.max(rawStart, pos);
    const end = Math.min(rawEnd, text.length);
    if (start >= end) continue;
    if (start > pos) out.push({ text: text.slice(pos, start), hit: false });
    out.push({ text: text.slice(start, end), hit: true });
    pos = end;
  }
  if (pos < text.length || out.length === 0) out.push({ text: text.slice(pos), hit: false });
  return out;
}

function escapeRegExp(s: string): string {
  return s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

const WORD = /[\p{L}\p{N}_]+/gu;

/**
 * Locate a verified quote in passage text: exact first, then tolerant of case,
 * whitespace and punctuation between words (PDF text often differs in exactly
 * those ways). Returns null rather than guessing a wrong span.
 */
export function findQuote(text: string, quote: string): Span | null {
  if (!quote.trim()) return null;
  const i = text.indexOf(quote);
  if (i >= 0) return [i, i + quote.length];
  const words = quote.match(WORD);
  if (!words) return null;
  const re = new RegExp(words.map(escapeRegExp).join("[^\\p{L}\\p{N}_]+"), "iu");
  const m = re.exec(text);
  return m ? [m.index, m.index + m[0].length] : null;
}

/** Spans of words starting with any matched BM25 term (terms may be stems). */
export function termSpans(text: string, terms: string[]): Span[] {
  const clean = [...new Set(terms.map((t) => t.toLowerCase()))]
    .filter((t) => /^[\p{L}\p{N}_]+$/u.test(t))
    .sort((a, b) => b.length - a.length);
  if (clean.length === 0) return [];
  const re = new RegExp(
    `(?<![\\p{L}\\p{N}_])(?:${clean.map(escapeRegExp).join("|")})[\\p{L}\\p{N}_]*`,
    "giu",
  );
  return [...text.matchAll(re)].map((m) => [m.index, m.index + m[0].length] as Span);
}

// ------------------------------------------------------------------ answers
export type AnswerPart =
  { kind: "text"; text: string } | { kind: "cite"; marker: number; label: string };

/**
 * Split an answer into text and citation markers. Only markers that belong to
 * a verified citation become chips; anything else stays literal text.
 */
export function answerParts(answer: string, citations: Citation[]): AnswerPart[] {
  const labels = new Map(citations.map((c) => [c.marker, sourceLabel(c.source)]));
  const parts: AnswerPart[] = [];
  let pos = 0;
  for (const m of answer.matchAll(/\[(\d{1,3})\]/g)) {
    const marker = Number(m[1]);
    const label = labels.get(marker);
    if (label === undefined) continue;
    if (m.index > pos) parts.push({ kind: "text", text: answer.slice(pos, m.index) });
    parts.push({ kind: "cite", marker, label });
    pos = m.index + m[0].length;
  }
  if (pos < answer.length) parts.push({ kind: "text", text: answer.slice(pos) });
  return parts;
}

export const REASONS: Record<string, string> = {
  no_relevant_passages: "Nothing in your documents matched this question.",
  low_retrieval_score: "The best matching passages were not relevant enough to answer from.",
  model_declined:
    "The model read the most relevant passages and said they don't answer this question.",
  no_verified_citations:
    "The model drafted an answer, but none of its citations could be verified against the passages, so it was withheld.",
  model_output_invalid:
    "The model's reply was not in the expected format, so it was withheld. Asking again usually works.",
};

export function reasonText(reason: string | null | undefined): string {
  return REASONS[reason ?? ""] ?? "The documents don't contain enough evidence to answer.";
}

export const REJECTION_REASONS: Record<string, string> = {
  unknown_passage: "cited a passage it was not shown",
  quote_not_found: "quote not found in the passage",
  quote_too_short: "quote too short to verify",
};

// ------------------------------------------------------------------ errors
export interface FriendlyError {
  title: string;
  hint?: string;
}

// ------------------------------------------------------------------ API keys
export const KEY_PREFIX = "eka_";
export const KEY_LENGTH = 47; // "eka_" + 43 random URL-safe characters

/** Remove every kind of whitespace a copy-paste can add (spaces, line breaks, zero-width). */
export function cleanKey(raw: string): string {
  return raw.replace(/[\s\u200B-\u200D\u2060\uFEFF]/g, "");
}

/** Why a pasted value can't be an API key, or null if it looks right. Never echoes the key. */
export function keyProblem(raw: string): string | null {
  const key = cleanKey(raw);
  if (!key) return null;
  if (!key.startsWith(KEY_PREFIX)) {
    return `Keys start with “${KEY_PREFIX}”. What's in the box doesn't, so it may be something else (for example the workspace ID or a command).`;
  }
  if (!/^[A-Za-z0-9_-]+$/.test(key)) {
    return "This contains characters a key never has. Copy only the key itself.";
  }
  if (key.length !== KEY_LENGTH) {
    return `Keys are ${KEY_LENGTH} characters long; this is ${key.length}. Part of it may be missing, or two keys were pasted together.`;
  }
  return null;
}

/** What went wrong and how to fix it, for an API error code. */
export function friendlyError(code: string, message: string): FriendlyError {
  switch (code) {
    case "unreachable":
      return {
        title: "Cannot reach the API server.",
        hint: "Start it in another terminal with `make run`, then try again.",
      };
    case "unauthorized":
      return {
        title: "The API doesn't recognise this key.",
        hint: 'It may have been revoked, or created for a different data folder. Create a new one with `python -m app.cli create-workspace "My team"`.',
      };
    case "invalid_key_format":
      return {
        title: "That doesn't look like a complete API key.",
        hint: `Keys start with ${KEY_PREFIX} and are ${KEY_LENGTH} characters long. Copy the whole key and paste it again.`,
      };
    case "session_expired":
      return { title: "Your session has ended.", hint: "Connect again with your API key." };
    case "llm_unavailable":
      return {
        title: "The local language model (Ollama) is not running.",
        hint: "Open the Ollama app, then ask again. Search still works without it.",
      };
    case "llm_model_missing":
      return { title: message };
    case "llm_timeout":
      return {
        title: "The local model took too long to answer.",
        hint: "Try again; the first question after a restart also loads the model.",
      };
    case "answering_disabled":
      return { title: "Question answering is turned off on this server (EKA_LLM_BACKEND=none)." };
    case "index_out_of_date":
      return {
        title: "The search index was built with a different embedding model.",
        hint: "Stop the API and run `python -m app.cli reindex`, then start it again.",
      };
    case "reranking_disabled":
      return { title: "Reranking is disabled on this server.", hint: "Turn reranking off." };
    case "unsupported_type":
      return { title: message, hint: "Upload PDF, DOCX or TXT files." };
    case "not_found":
      return { title: "Not found. It may have been deleted." };
    case "validation_error":
      return { title: "The request was not valid." };
    case "forbidden":
      return { title: "The request was blocked (it did not come from this app)." };
    case "rate_limited":
      return { title: message, hint: "Wait a moment and try again." };
    case "internal_error":
      return { title: "The server hit an unexpected error.", hint: "Check the API terminal log." };
    default:
      return { title: message || "Something went wrong." };
  }
}

// ------------------------------------------------------------------ formatting
export function humanSize(n: number | null | undefined): string {
  if (n === null || n === undefined) return "–";
  let size = n;
  for (const unit of ["B", "KB", "MB", "GB"]) {
    if (size < 1024 || unit === "GB")
      return unit === "B" ? `${size} B` : `${size.toFixed(1)} ${unit}`;
    size /= 1024;
  }
  return `${size.toFixed(1)} GB`;
}

export function fmtMs(ms: number | null | undefined): string {
  if (ms === null || ms === undefined) return "–";
  return ms < 1000 ? `${Math.round(ms)} ms` : `${(ms / 1000).toFixed(1)} s`;
}

export function pct(x: number | null | undefined): string {
  return x === null || x === undefined ? "–" : `${Math.round(x * 100)}%`;
}

export function fmtTime(iso: string | null | undefined): string {
  if (!iso) return "–";
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return iso;
  const pad = (n: number) => String(n).padStart(2, "0");
  return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())} ${pad(d.getHours())}:${pad(d.getMinutes())}`;
}

export function statusText(status: string): string {
  return status.replaceAll("_", " ");
}
