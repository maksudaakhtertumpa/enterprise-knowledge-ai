// Browser-side client. Talks only to this site's own /api/* routes; the API key
// stays in the server-side session and is never visible here.

import type {
  Analytics,
  AskResponse,
  ChunkOut,
  DocumentOut,
  ErrorBody,
  EvidenceContext,
  HistoryItem,
  Page,
  Readiness,
  SearchMode,
  SearchResponse,
  Workspace,
} from "@/lib/types";

export class ApiError extends Error {
  constructor(
    public status: number,
    public code: string,
    message: string,
    public requestId: string | null = null,
    public details: unknown = null,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

async function toError(resp: Response): Promise<ApiError> {
  try {
    const body = (await resp.json()) as { error?: ErrorBody };
    if (body.error?.code) {
      return new ApiError(
        resp.status,
        body.error.code,
        body.error.message,
        body.error.request_id ?? null,
        body.error.details ?? null,
      );
    }
  } catch {
    // not our envelope; fall through
  }
  return new ApiError(resp.status, "http_error", `HTTP ${resp.status}`);
}

type Query = Record<string, string | number | undefined>;

function qs(params?: Query): string {
  if (!params) return "";
  const entries = Object.entries(params).filter(([, v]) => v !== undefined) as [string, string][];
  return entries.length ? `?${new URLSearchParams(entries.map(([k, v]) => [k, String(v)]))}` : "";
}

async function call<T>(
  method: "GET" | "POST" | "DELETE",
  path: string,
  options: {
    json?: unknown;
    form?: FormData;
    query?: Query;
    base?: string;
    accept?: number[]; // non-2xx statuses whose body is still the result
  } = {},
): Promise<T> {
  const init: RequestInit = { method, cache: "no-store", headers: {} };
  if (options.json !== undefined) {
    init.body = JSON.stringify(options.json);
    (init.headers as Record<string, string>)["Content-Type"] = "application/json";
  } else if (options.form) {
    init.body = options.form;
  }
  let resp: Response;
  try {
    resp = await fetch(`${options.base ?? "/api/eka"}/${path}${qs(options.query)}`, init);
  } catch {
    throw new ApiError(0, "unreachable", "Cannot reach the web server.");
  }
  if (!resp.ok && !options.accept?.includes(resp.status)) {
    const error = await toError(resp);
    if (resp.status === 401 && !options.base && typeof window !== "undefined") {
      // Session expired or key revoked: a full page load to the connect page,
      // so no workspace data stays in memory.
      // eslint-disable-next-line @next/next/no-location-assign-relative-destination
      window.location.assign("/connect?expired=1");
    }
    throw error;
  }
  if (resp.status === 204) return undefined as T;
  return (await resp.json()) as T;
}

export const api = {
  // session (this site's server)
  connect: (key: string) =>
    call<{ workspace: Workspace }>("POST", "session", { json: { key }, base: "/api" }),
  disconnect: () => call<void>("DELETE", "session", { base: "/api" }),
  health: () => call<Readiness>("GET", "health", { base: "/api", accept: [503] }),

  // backend, through the authenticated proxy
  workspace: () => call<Workspace>("GET", "workspace"),
  listDocuments: (limit = 100, offset = 0) =>
    call<Page<DocumentOut>>("GET", "documents", { query: { limit, offset } }),
  async allDocuments(max = 1000): Promise<DocumentOut[]> {
    const items: DocumentOut[] = [];
    while (items.length < max) {
      const page = await api.listDocuments(100, items.length);
      items.push(...page.items);
      if (page.items.length === 0 || items.length >= page.total) break;
    }
    return items.slice(0, max);
  },
  getDocument: (id: string) => call<DocumentOut>("GET", `documents/${id}`),
  chunks: (id: string, limit = 10, offset = 0) =>
    call<Page<ChunkOut> & { document_id: string }>("GET", `documents/${id}/chunks`, {
      query: { limit, offset },
    }),
  upload: (file: File) => {
    const form = new FormData();
    form.append("file", file, file.name);
    return call<DocumentOut>("POST", "documents", { form });
  },
  deleteDocument: (id: string) => call<void>("DELETE", `documents/${id}`),
  reprocess: (id: string) => call<DocumentOut>("POST", `documents/${id}/reprocess`),
  search: (body: {
    query: string;
    mode?: SearchMode;
    rerank?: boolean;
    top_k?: number;
    document_ids?: string[];
  }) => call<SearchResponse>("POST", "search", { json: body }),
  ask: (body: {
    question: string;
    passages?: number;
    document_ids?: string[];
    mode?: SearchMode;
    rerank?: boolean;
  }) => call<AskResponse>("POST", "ask", { json: body }),
  history: (limit = 20, offset = 0) =>
    call<Page<HistoryItem>>("GET", "ask/history", { query: { limit, offset } }),
  historyEntry: (id: string) => call<AskResponse>("GET", `ask/history/${id}`),
  evidence: (chunkId: string, context = 300) =>
    call<EvidenceContext>("GET", `evidence/${chunkId}`, { query: { context } }),
  analytics: (days = 30) => call<Analytics>("GET", "analytics", { query: { days } }),
};
