"use client";

import { useEffect, useRef, useState } from "react";
import Link from "next/link";
import { Bot, Loader2, SendHorizontal, Trash2, User } from "lucide-react";

import { AnswerView } from "@/components/answer-view";
import { DocumentFilter } from "@/components/document-filter";
import { ErrorAlert } from "@/components/error-alert";
import { PageHeader } from "@/components/page-header";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Skeleton } from "@/components/ui/skeleton";
import { Textarea } from "@/components/ui/textarea";
import { asApiError, useLoad } from "@/hooks/use-load";
import { type ApiError, api } from "@/lib/api";
import type { AskResponse, SearchMode } from "@/lib/types";

interface Turn {
  id: number;
  question: string;
  response?: AskResponse;
  error?: ApiError;
}

const DEFAULT = "default";

function Elapsed() {
  const [seconds, setSeconds] = useState(0);
  useEffect(() => {
    const t = setInterval(() => setSeconds((s) => s + 1), 1000);
    return () => clearInterval(t);
  }, []);
  return <span className="tabular-nums">{seconds} s</span>;
}

export function AskView() {
  const docs = useLoad(() => api.allDocuments(), "docs");
  const ready = (docs.data ?? []).filter((d) => d.status === "ready");
  const [turns, setTurns] = useState<Turn[]>([]);
  const [question, setQuestion] = useState("");
  const [selected, setSelected] = useState<string[]>([]);
  const [passages, setPassages] = useState("5");
  const [mode, setMode] = useState(DEFAULT);
  const [rerank, setRerank] = useState(DEFAULT);
  const [pending, setPending] = useState(false);
  const bottom = useRef<HTMLDivElement>(null);
  const nextId = useRef(1);

  useEffect(() => {
    bottom.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [turns.length, pending]);

  async function ask(e?: React.FormEvent) {
    e?.preventDefault();
    const q = question.trim();
    if (!q || pending) return;
    const id = nextId.current++;
    setTurns((t) => [...t, { id, question: q }]);
    setQuestion("");
    setPending(true);
    try {
      const response = await api.ask({
        question: q,
        passages: Number(passages),
        document_ids: selected.length ? selected : undefined,
        mode: mode === DEFAULT ? undefined : (mode as SearchMode),
        rerank: rerank === DEFAULT ? undefined : rerank === "on",
      });
      setTurns((t) => t.map((x) => (x.id === id ? { ...x, response } : x)));
    } catch (err) {
      setTurns((t) => t.map((x) => (x.id === id ? { ...x, error: asApiError(err) } : x)));
    } finally {
      setPending(false);
    }
  }

  return (
    <div className="flex min-h-[calc(100dvh-8rem)] flex-col">
      <PageHeader
        title="Ask"
        description="Answers use only your documents. Every sentence cites a passage, and every citation is checked against the source before it is shown. Each question is answered on its own (no memory of earlier questions)."
      >
        {turns.length > 0 && (
          <Button variant="ghost" size="sm" onClick={() => setTurns([])} disabled={pending}>
            <Trash2 /> Clear
          </Button>
        )}
      </PageHeader>

      {docs.error && <ErrorAlert error={docs.error} className="mb-4" />}
      {docs.loading && !docs.data && <Skeleton className="mb-4 h-10 w-full" />}
      {docs.data && ready.length === 0 && (
        <Alert className="mb-4">
          <AlertTitle>No searchable documents yet</AlertTitle>
          <AlertDescription>
            <p>
              Upload PDF, DOCX or TXT files on the{" "}
              <Link href="/documents" className="text-foreground underline underline-offset-4">
                Documents
              </Link>{" "}
              page first.
            </p>
          </AlertDescription>
        </Alert>
      )}

      <div className="flex-1 space-y-6">
        {turns.map((turn) => (
          <div key={turn.id} className="space-y-3" data-testid="turn">
            <div className="flex items-start gap-3">
              <div className="flex size-7 shrink-0 items-center justify-center rounded-full bg-muted">
                <User className="size-4" />
              </div>
              <p className="pt-0.5 font-medium whitespace-pre-wrap">{turn.question}</p>
            </div>
            <div className="flex items-start gap-3">
              <div className="flex size-7 shrink-0 items-center justify-center rounded-full bg-primary text-primary-foreground">
                <Bot className="size-4" />
              </div>
              <div className="min-w-0 flex-1">
                {turn.response && <AnswerView response={turn.response} />}
                {turn.error && <ErrorAlert error={turn.error} />}
                {!turn.response && !turn.error && (
                  <div className="space-y-2" data-testid="thinking">
                    <p className="flex items-center gap-2 text-sm text-muted-foreground">
                      <Loader2 className="size-4 animate-spin" />
                      Finding evidence and asking the local model… <Elapsed />
                    </p>
                    <Skeleton className="h-4 w-3/4" />
                    <Skeleton className="h-4 w-1/2" />
                  </div>
                )}
              </div>
            </div>
          </div>
        ))}
        <div ref={bottom} />
      </div>

      <form
        onSubmit={ask}
        className="mt-6 space-y-2 border-t bg-background pt-4 pb-2 md:sticky md:bottom-0"
      >
        <div className="flex flex-wrap items-center gap-2">
          <DocumentFilter documents={ready} selected={selected} onChange={setSelected} />
          <Select value={passages} onValueChange={setPassages}>
            <SelectTrigger size="sm" aria-label="Evidence passages">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {Array.from({ length: 10 }, (_, i) => String(i + 1)).map((n) => (
                <SelectItem key={n} value={n}>
                  {n} passage{n === "1" ? "" : "s"}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
          <Select value={mode} onValueChange={setMode}>
            <SelectTrigger size="sm" aria-label="Retrieval mode">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value={DEFAULT}>Retrieval: server default</SelectItem>
              <SelectItem value="dense">Dense (semantic)</SelectItem>
              <SelectItem value="hybrid">Hybrid (RRF)</SelectItem>
              <SelectItem value="bm25">BM25 (keyword)</SelectItem>
            </SelectContent>
          </Select>
          <Select value={rerank} onValueChange={setRerank}>
            <SelectTrigger size="sm" aria-label="Reranking">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value={DEFAULT}>Rerank: server default</SelectItem>
              <SelectItem value="on">Rerank on</SelectItem>
              <SelectItem value="off">Rerank off</SelectItem>
            </SelectContent>
          </Select>
        </div>
        <div className="flex items-end gap-2">
          <Textarea
            aria-label="Question"
            placeholder="Ask a question about your documents"
            value={question}
            maxLength={1000}
            disabled={ready.length === 0}
            onChange={(e) => setQuestion(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                void ask();
              }
            }}
            className="max-h-40 min-h-11"
          />
          <Button
            type="submit"
            size="icon"
            aria-label="Send"
            disabled={pending || !question.trim() || ready.length === 0}
          >
            {pending ? <Loader2 className="animate-spin" /> : <SendHorizontal />}
          </Button>
        </div>
      </form>
    </div>
  );
}
