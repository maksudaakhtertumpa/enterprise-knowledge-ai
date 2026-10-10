"use client";

import { useState } from "react";

import { ErrorAlert } from "@/components/error-alert";
import { PageHeader } from "@/components/page-header";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import { useLoad } from "@/hooks/use-load";
import { api } from "@/lib/api";
import { fmtMs, humanSize, pct, reasonText } from "@/lib/format";
import type { Analytics } from "@/lib/types";

function Metric({ label, value, help }: { label: string; value: string; help?: string }) {
  return (
    <Card className="gap-1 py-4" data-testid="metric">
      <CardContent className="px-4">
        <p className="text-xs text-muted-foreground" title={help}>
          {label}
        </p>
        <p className="text-2xl font-semibold tabular-nums" data-testid={`metric-${label}`}>
          {value}
        </p>
      </CardContent>
    </Card>
  );
}

/** Stacked bars per day, drawn with plain elements (no chart library needed). */
function PerDayChart({ days }: { days: Analytics["per_day"] }) {
  const max = Math.max(1, ...days.map((d) => d.answered + d.insufficient_evidence));
  return (
    <div>
      <div className="flex h-40 items-end gap-[2px]" role="img" aria-label="Questions per day">
        {days.map((d) => {
          const total = d.answered + d.insufficient_evidence;
          return (
            <Tooltip key={d.date}>
              <TooltipTrigger asChild>
                <div className="flex h-full min-w-0 flex-1 flex-col justify-end">
                  <div
                    className="rounded-t-[2px] bg-warning"
                    style={{ height: `${(d.insufficient_evidence / max) * 100}%` }}
                  />
                  <div
                    className={total ? "bg-success" : "bg-muted"}
                    style={{ height: total ? `${(d.answered / max) * 100}%` : "2px" }}
                  />
                </div>
              </TooltipTrigger>
              <TooltipContent>
                {d.date}: {d.answered} answered, {d.insufficient_evidence} not enough evidence
              </TooltipContent>
            </Tooltip>
          );
        })}
      </div>
      <div className="mt-2 flex justify-between text-xs text-muted-foreground">
        <span>{days[0]?.date}</span>
        <span>{days[days.length - 1]?.date}</span>
      </div>
      <div className="mt-2 flex gap-4 text-xs text-muted-foreground">
        <span className="flex items-center gap-1">
          <span className="size-2.5 rounded-sm bg-success" /> Answered
        </span>
        <span className="flex items-center gap-1">
          <span className="size-2.5 rounded-sm bg-warning" /> Not enough evidence
        </span>
      </div>
    </div>
  );
}

export function AnalyticsView() {
  const [days, setDays] = useState("30");
  const stats = useLoad(() => api.analytics(Number(days)), days);
  const data = stats.data;

  return (
    <div className="space-y-6">
      <PageHeader
        title="Analytics"
        description="Usage of this workspace from its question log. These describe real traffic, not accuracy: answer quality is measured separately with labelled questions (eval/README.md)."
      >
        <ToggleGroup
          type="single"
          value={days}
          onValueChange={(v) => v && setDays(v)}
          aria-label="Window"
        >
          {["7", "30", "90"].map((d) => (
            <ToggleGroupItem key={d} value={d}>
              {d} days
            </ToggleGroupItem>
          ))}
        </ToggleGroup>
      </PageHeader>
      {stats.error && <ErrorAlert error={stats.error} />}
      {!data && !stats.error && <Skeleton className="h-64 w-full" />}
      {data && (
        <>
          <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
            <Metric label="Questions" value={String(data.questions.total)} />
            <Metric
              label="Answered"
              value={pct(data.questions.answer_rate)}
              help="Share of questions that got a verified answer"
            />
            <Metric label="Median time" value={fmtMs(data.latency?.p50_ms)} />
            <Metric label="95th percentile" value={fmtMs(data.latency?.p95_ms)} />
            <Metric
              label="Citations verified"
              value={pct(data.citations.acceptance_rate)}
              help="Share of the citations the model proposed that passed verification"
            />
            <Metric
              label="Citations per answer"
              value={data.citations.avg_per_answer?.toFixed(1) ?? "–"}
            />
            <Metric
              label="Tokens (in + out)"
              value={(data.tokens.prompt + data.tokens.completion).toLocaleString()}
            />
            <Metric
              label="Cost"
              value={`$${data.tokens.cost_usd.toFixed(2)}`}
              help="Models run locally"
            />
          </div>

          {data.questions.total > 0 ? (
            <div className="grid gap-6 lg:grid-cols-3">
              <Card className="lg:col-span-2">
                <CardHeader>
                  <CardTitle>Questions per day</CardTitle>
                </CardHeader>
                <CardContent>
                  <PerDayChart days={data.per_day} />
                </CardContent>
              </Card>
              <Card>
                <CardHeader>
                  <CardTitle>Why questions went unanswered</CardTitle>
                  <CardDescription>
                    Refusing is intended when the documents don&apos;t contain the answer.
                  </CardDescription>
                </CardHeader>
                <CardContent className="space-y-3 text-sm">
                  {Object.keys(data.questions.refusal_reasons).length === 0 && (
                    <p className="text-muted-foreground">
                      Every question in this window was answered.
                    </p>
                  )}
                  {Object.entries(data.questions.refusal_reasons).map(([reason, n]) => (
                    <div key={reason} className="flex justify-between gap-3">
                      <span className="text-muted-foreground">{reasonText(reason)}</span>
                      <span className="font-semibold tabular-nums">{n}</span>
                    </div>
                  ))}
                  <div className="border-t pt-3">
                    <p className="mb-1 font-medium">Models</p>
                    {Object.entries(data.models).map(([model, n]) => (
                      <p key={model} className="text-muted-foreground">
                        {model}: {n}
                      </p>
                    ))}
                  </div>
                </CardContent>
              </Card>
            </div>
          ) : (
            <p className="rounded-lg border border-dashed p-6 text-center text-sm text-muted-foreground">
              No questions in this window yet.
            </p>
          )}

          <div>
            <h2 className="mb-3 font-semibold">Documents</h2>
            <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
              <Metric label="Documents" value={String(data.documents.total)} />
              <Metric label="Ready" value={String(data.documents.by_status.ready ?? 0)} />
              <Metric label="Searchable chunks" value={data.documents.chunks.toLocaleString()} />
              <Metric label="Stored" value={humanSize(data.documents.size_bytes)} />
            </div>
            {(data.documents.by_status.failed ?? 0) > 0 && (
              <p className="mt-2 text-sm text-destructive">
                {data.documents.by_status.failed} document(s) failed to process. See the Documents
                page.
              </p>
            )}
          </div>
        </>
      )}
    </div>
  );
}
