"use client";

import { useState } from "react";
import { ChevronLeft, ChevronRight, X } from "lucide-react";

import { AnswerView } from "@/components/answer-view";
import { ErrorAlert } from "@/components/error-alert";
import { PageHeader } from "@/components/page-header";
import { StatusBadge } from "@/components/status-badge";
import { Button } from "@/components/ui/button";
import { Card, CardAction, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { useLoad } from "@/hooks/use-load";
import { api } from "@/lib/api";
import { fmtMs, fmtTime, reasonText, truncate } from "@/lib/format";

const PAGE_SIZE = 15;

function Entry({ id, onClose }: { id: string; onClose: () => void }) {
  const entry = useLoad(() => api.historyEntry(id), id);
  return (
    <Card data-testid="history-entry">
      <CardHeader>
        <CardTitle className="leading-snug whitespace-pre-wrap">
          {entry.data?.question ?? "Loading…"}
        </CardTitle>
        <CardAction>
          <Button variant="ghost" size="icon" aria-label="Close" onClick={onClose}>
            <X />
          </Button>
        </CardAction>
      </CardHeader>
      <CardContent>
        {entry.error &&
          (entry.error.code === "not_found" ? (
            <p className="text-sm text-muted-foreground">
              The full answer for this entry was not stored (history text is off).
            </p>
          ) : (
            <ErrorAlert error={entry.error} />
          ))}
        {entry.loading && !entry.data && <Skeleton className="h-24 w-full" />}
        {entry.data && <AnswerView response={entry.data} />}
      </CardContent>
    </Card>
  );
}

export function HistoryView() {
  const [page, setPage] = useState(0);
  const [openId, setOpenId] = useState<string>();
  const history = useLoad(() => api.history(PAGE_SIZE, page * PAGE_SIZE), `history:${page}`);
  const data = history.data;
  const pages = data ? Math.max(1, Math.ceil(data.total / PAGE_SIZE)) : 1;

  return (
    <div className="space-y-4">
      <PageHeader
        title="History"
        description="Questions asked in this workspace (by anyone with a key to it), newest first."
      />
      {openId && <Entry id={openId} onClose={() => setOpenId(undefined)} />}
      {history.error && <ErrorAlert error={history.error} />}
      {history.loading && !data && <Skeleton className="h-48 w-full" />}
      {data && data.items.length === 0 && (
        <p className="rounded-lg border border-dashed p-6 text-center text-sm text-muted-foreground">
          No questions yet. Ask one on the Ask page.
        </p>
      )}
      <ul className="space-y-2">
        {data?.items.map((item) => (
          <li key={item.id} data-testid="history-item">
            <button
              type="button"
              disabled={item.question === null}
              onClick={() => setOpenId(item.id)}
              className="w-full rounded-lg border p-4 text-left transition-colors hover:bg-accent/50 disabled:cursor-default disabled:hover:bg-transparent"
            >
              <div className="flex flex-wrap items-start justify-between gap-2">
                <p className="font-medium">
                  {truncate(item.question ?? "(question text not stored)", 160)}
                </p>
                <StatusBadge status={item.status} />
              </div>
              <p className="mt-1 text-sm text-muted-foreground">
                {item.status === "answered" ? truncate(item.answer, 200) : reasonText(item.reason)}
              </p>
              <p className="mt-2 text-xs text-muted-foreground">
                {fmtTime(item.created_at)} · {fmtMs(item.latency_ms)}
                {item.citations_verified > 0 && ` · ${item.citations_verified} citation(s)`}
                {item.model && ` · ${item.model}`}
              </p>
            </button>
          </li>
        ))}
      </ul>
      {data && pages > 1 && (
        <div className="flex items-center justify-between">
          <Button
            variant="outline"
            size="sm"
            disabled={page === 0}
            onClick={() => setPage(page - 1)}
          >
            <ChevronLeft /> Newer
          </Button>
          <span className="text-xs text-muted-foreground">
            Page {page + 1} of {pages} · {data.total} questions
          </span>
          <Button
            variant="outline"
            size="sm"
            disabled={page >= pages - 1}
            onClick={() => setPage(page + 1)}
          >
            Older <ChevronRight />
          </Button>
        </div>
      )}
    </div>
  );
}
