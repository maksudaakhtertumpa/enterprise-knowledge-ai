// Response shapes of the FastAPI backend (app/schemas/*.py). Kept by hand and
// small on purpose; the end-to-end tests run the UI against the real API.

export type FileType = "pdf" | "docx" | "txt";
export type DocumentStatus = "pending" | "processing" | "ready" | "failed";
export type AnswerStatus = "answered" | "insufficient_evidence";
export type SearchMode = "bm25" | "dense" | "hybrid";

export interface Workspace {
  id: string;
  name: string;
}

export interface DocumentOut {
  id: string;
  filename: string;
  file_type: FileType;
  size_bytes: number;
  sha256: string;
  status: DocumentStatus;
  error: string | null;
  title: string | null;
  page_count: number | null;
  char_count: number | null;
  chunk_count: number;
  created_at: string;
  updated_at: string;
  processed_at: string | null;
}

export interface Page<T> {
  items: T[];
  total: number;
  limit: number;
  offset: number;
}

export interface ChunkOut {
  id: string;
  ordinal: number;
  text: string;
  heading: string | null;
  page_start: number | null;
  page_end: number | null;
  char_start: number;
  char_end: number;
  token_count: number;
}

export interface SourceRef {
  document_id: string;
  filename: string;
  title: string | null;
  file_type: FileType;
  heading: string | null;
  page_start: number | null;
  page_end: number | null;
}

export interface SearchResult {
  rank: number;
  score: number;
  chunk_id: string;
  text: string;
  source: SourceRef & { char_start: number; char_end: number };
  matched_terms: string[];
  retriever_ranks: Record<string, number> | null;
  retriever_scores: Record<string, number> | null;
  retrieval_rank: number | null;
}

export interface SearchResponse {
  query: string;
  mode: SearchMode;
  top_k: number;
  reranked: boolean;
  embedding_model: string | null;
  reranker_model: string | null;
  timings_ms: Record<string, number>;
  results: SearchResult[];
}

export interface Citation {
  marker: number;
  chunk_id: string;
  quote: string;
  match: "exact" | "fuzzy";
  quote_char_start: number | null;
  quote_char_end: number | null;
  source: SourceRef;
}

export interface Evidence {
  marker: number;
  chunk_id: string;
  text: string;
  score: number;
  rerank_score: number | null;
  suspicious: boolean;
  source: SourceRef;
}

export interface RejectedCitation {
  marker: number;
  quote: string;
  reason: "unknown_passage" | "quote_not_found" | "quote_too_short";
}

export interface Verification {
  citations_proposed: number;
  citations_verified: number;
  sentences_total: number;
  sentences_removed: number;
  rejected: RejectedCitation[];
  entailment?: Entailment | null;
}

export interface Entailment {
  model: string;
  mode: "remove" | "flag";
  threshold: number;
  sentences_removed: number;
  sentences_flagged: number;
  sentences: { text: string; score: number; supported: boolean }[];
}

export interface AskResponse {
  id: string | null;
  question: string;
  status: AnswerStatus;
  answer: string | null;
  reason: string | null;
  citations: Citation[];
  evidence: Evidence[];
  verification: Verification | null;
  model: string | null;
  llm_backend: string | null;
  embedding_model: string | null;
  reranker_model: string | null;
  usage: { prompt_tokens: number | null; completion_tokens: number | null; cost_usd: number };
  warnings: string[];
  timings_ms: Record<string, number>;
}

export interface HistoryItem {
  id: string;
  created_at: string;
  question: string | null;
  status: AnswerStatus;
  reason: string | null;
  answer: string | null;
  model: string | null;
  latency_ms: number;
  citations_verified: number;
}

export interface EvidenceContext {
  chunk_id: string;
  text: string;
  before: string;
  after: string;
  char_start: number;
  char_end: number;
  source: SourceRef;
}

export interface Analytics {
  window_days: number;
  since: string;
  questions: {
    total: number;
    answered: number;
    insufficient_evidence: number;
    answer_rate: number | null;
    refusal_reasons: Record<string, number>;
  };
  latency: { p50_ms: number; p95_ms: number; mean_ms: number; max_ms: number } | null;
  citations: {
    proposed: number;
    verified: number;
    acceptance_rate: number | null;
    avg_per_answer: number | null;
  };
  tokens: { prompt: number; completion: number; cost_usd: number };
  models: Record<string, number>;
  per_day: { date: string; answered: number; insufficient_evidence: number }[];
  documents: {
    total: number;
    by_status: Record<DocumentStatus, number>;
    chunks: number;
    size_bytes: number;
  };
}

export interface ReadinessCheck {
  name: string;
  ok: boolean;
  detail: string | null;
  required: boolean;
}

export interface Readiness {
  status: "ready" | "not_ready";
  checks: ReadinessCheck[];
}

export interface ErrorBody {
  code: string;
  message: string;
  details?: unknown;
  request_id?: string | null;
}
