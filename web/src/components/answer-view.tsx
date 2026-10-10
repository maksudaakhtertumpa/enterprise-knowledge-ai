"use client";

import { useRef, useState } from "react";
import { ChevronDown, FileText, SearchX, ShieldAlert } from "lucide-react";

import { ErrorAlert } from "@/components/error-alert";
import { Highlighted } from "@/components/highlighted";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible";
import { Label } from "@/components/ui/label";
import { Skeleton } from "@/components/ui/skeleton";
import { Switch } from "@/components/ui/switch";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import { ApiError, api } from "@/lib/api";
import {
  REJECTION_REASONS,
  answerParts,
  findQuote,
  fmtMs,
  reasonText,
  sourceLabel,
} from "@/lib/format";
import type { AskResponse, Citation, EvidenceContext } from "@/lib/types";
import { cn } from "@/lib/utils";

function CitationChip({
  marker,
  label,
  onClick,
}: {
  marker: number;
  label: string;
  onClick: () => void;
}) {
  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <button
          type="button"
          onClick={onClick}
          data-testid="citation-chip"
          aria-label={`Source ${marker}: ${label}`}
          className="mx-0.5 inline-flex h-5 min-w-5 items-center justify-center rounded-full border border-cite/50 bg-cite/15 px-1.5 align-[0.1em] text-[0.7rem] font-semibold transition-colors hover:bg-cite/30"
        >
          {marker}
        </button>
      </TooltipTrigger>
      <TooltipContent>{label}</TooltipContent>
    </Tooltip>
  );
}

function QuoteInPassage({ text, quote }: { text: string; quote: string }) {
  const span = findQuote(text, quote);
  return <Highlighted text={text} spans={span ? [span] : []} />;
}

function SourceCard({
  citation,
  passage,
  open,
  onOpenChange,
}: {
  citation: Citation;
  passage: string | undefined;
  open: boolean;
  onOpenChange: (open: boolean) => void;
}) {
  const [inContext, setInContext] = useState(false);
  const [context, setContext] = useState<EvidenceContext>();
  const [error, setError] = useState<ApiError>();
  const [loading, setLoading] = useState(false);

  async function toggleContext(on: boolean) {
    setInContext(on);
    if (!on || context) return;
    setLoading(true);
    try {
      setContext(await api.evidence(citation.chunk_id));
      setError(undefined);
    } catch (e) {
      setError(e instanceof ApiError ? e : new ApiError(0, "client_error", String(e)));
    } finally {
      setLoading(false);
    }
  }

  const switchId = `ctx-${citation.chunk_id}-${citation.marker}`;
  return (
    <Collapsible
      open={open}
      onOpenChange={onOpenChange}
      className="rounded-lg border"
      data-testid="source-card"
    >
      <CollapsibleTrigger className="flex w-full items-center gap-3 rounded-lg px-3 py-2.5 text-left text-sm hover:bg-muted/50">
        <Badge variant="outline" className="border-cite/50 bg-cite/10">
          {citation.marker}
        </Badge>
        <FileText className="size-4 shrink-0 text-muted-foreground" />
        <span className="min-w-0 flex-1 truncate font-medium">{sourceLabel(citation.source)}</span>
        {citation.match === "fuzzy" && <Badge variant="secondary">approximate quote</Badge>}
        <ChevronDown className={cn("size-4 shrink-0 transition-transform", open && "rotate-180")} />
      </CollapsibleTrigger>
      <CollapsibleContent className="space-y-3 border-t px-3 py-3">
        <div className="flex items-center gap-2">
          <Switch id={switchId} checked={inContext} onCheckedChange={toggleContext} />
          <Label htmlFor={switchId} className="text-xs font-normal text-muted-foreground">
            Show surrounding text in the document
          </Label>
        </div>
        {inContext && loading && <Skeleton className="h-24 w-full" />}
        {inContext &&
          error &&
          (error.code === "not_found" ? (
            <p className="text-sm text-muted-foreground">
              This source is no longer available (the document was deleted or is being reprocessed).
            </p>
          ) : (
            <ErrorAlert error={error} />
          ))}
        {inContext && context && (
          <p className="text-sm leading-relaxed" data-testid="source-context">
            {context.before && (
              <span className="whitespace-pre-wrap text-muted-foreground">
                {context.char_start > context.before.length ? "…" : ""}
                {context.before}
              </span>
            )}
            <QuoteInPassage text={context.text} quote={citation.quote} />
            {context.after && (
              <span className="whitespace-pre-wrap text-muted-foreground">{context.after}…</span>
            )}
          </p>
        )}
        {!inContext && (
          <>
            <p className="text-xs text-muted-foreground">
              Verified quote, highlighted in the passage the model was shown:
            </p>
            <p className="text-sm leading-relaxed" data-testid="source-passage">
              {passage ? (
                <QuoteInPassage text={passage} quote={citation.quote} />
              ) : (
                <Highlighted text={citation.quote} spans={[[0, citation.quote.length]]} />
              )}
            </p>
          </>
        )}
      </CollapsibleContent>
    </Collapsible>
  );
}

function Checks({ response }: { response: AskResponse }) {
  const v = response.verification;
  const [open, setOpen] = useState(false);
  return (
    <Collapsible open={open} onOpenChange={setOpen}>
      <CollapsibleTrigger className="flex items-center gap-1 text-xs text-muted-foreground hover:text-foreground">
        <ChevronDown className={cn("size-3.5 transition-transform", open && "rotate-180")} />
        How this answer was checked
      </CollapsibleTrigger>
      <CollapsibleContent className="mt-3 space-y-4 rounded-lg border p-4 text-sm">
        {v ? (
          <div className="grid grid-cols-3 gap-3">
            <Stat
              label="Citations verified"
              value={`${v.citations_verified}/${v.citations_proposed}`}
            />
            <Stat
              label="Sentences kept"
              value={`${v.sentences_total - v.sentences_removed}/${v.sentences_total}`}
            />
            <Stat label="Passages read" value={String(response.evidence.length)} />
          </div>
        ) : (
          <p className="text-muted-foreground">The model was not called (no relevant passages).</p>
        )}
        {v && v.sentences_removed > 0 && (
          <p className="text-muted-foreground">
            {v.sentences_removed} sentence(s) were removed because their citation could not be
            verified.
          </p>
        )}
        {v?.rejected.map((r, i) => (
          <div key={i} className="space-y-1">
            <p className="text-xs text-muted-foreground">
              Rejected citation [{r.marker}]: {REJECTION_REASONS[r.reason] ?? r.reason}
            </p>
            <p className="rounded bg-muted px-2 py-1 font-mono text-xs whitespace-pre-wrap">
              {r.quote}
            </p>
          </div>
        ))}
        {response.evidence.length > 0 && (
          <div className="space-y-3">
            <p className="font-medium">Passages the model was shown</p>
            {response.evidence.map((e) => (
              <div key={e.marker} className="space-y-1">
                <p className="text-xs text-muted-foreground">
                  [{e.marker}] {sourceLabel(e.source)} ·{" "}
                  {e.rerank_score !== null
                    ? `relevance ${e.rerank_score.toFixed(3)}`
                    : `score ${e.score.toFixed(3)}`}
                  {e.suspicious && " · instruction-like text (treated as data)"}
                </p>
                <p className="text-xs leading-relaxed whitespace-pre-wrap text-muted-foreground">
                  {e.text}
                </p>
              </div>
            ))}
          </div>
        )}
        {(response.embedding_model || response.reranker_model) && (
          <p className="text-xs text-muted-foreground">
            Retrieval:{" "}
            {[response.embedding_model, response.reranker_model].filter(Boolean).join(" + ")}
          </p>
        )}
      </CollapsibleContent>
    </Collapsible>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <p className="text-xs text-muted-foreground">{label}</p>
      <p className="text-lg font-semibold">{value}</p>
    </div>
  );
}

export function AnswerView({ response }: { response: AskResponse }) {
  const [open, setOpen] = useState<Record<number, boolean>>({});
  const cards = useRef<Record<number, HTMLDivElement | null>>({});
  const passages = new Map(response.evidence.map((e) => [e.marker, e.text]));
  const suspicious = response.evidence.filter((e) => e.suspicious).length;
  const v = response.verification;

  function openSource(marker: number) {
    const index = response.citations.findIndex((c) => c.marker === marker);
    if (index < 0) return;
    setOpen((o) => ({ ...o, [index]: true }));
    cards.current[index]?.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }

  return (
    <div className="space-y-4" data-testid="answer">
      {response.status === "answered" && response.answer ? (
        <p className="leading-7 whitespace-pre-wrap" data-testid="answer-text">
          {answerParts(response.answer, response.citations).map((part, i) =>
            part.kind === "text" ? (
              <span key={i}>{part.text}</span>
            ) : (
              <CitationChip
                key={i}
                marker={part.marker}
                label={part.label}
                onClick={() => openSource(part.marker)}
              />
            ),
          )}
        </p>
      ) : (
        <Alert variant="warning" data-testid="refusal">
          <SearchX />
          <AlertTitle>Not enough evidence to answer.</AlertTitle>
          <AlertDescription>{reasonText(response.reason)}</AlertDescription>
        </Alert>
      )}

      {suspicious > 0 && (
        <p className="flex items-center gap-1.5 text-xs text-muted-foreground">
          <ShieldAlert className="size-3.5 text-warning" />
          {suspicious} passage(s) contained instruction-like text. They were treated as data, not
          instructions.
        </p>
      )}
      {response.warnings.map((w, i) => (
        <p key={i} className="text-xs text-muted-foreground">
          Note: {w}
        </p>
      ))}

      {response.citations.length > 0 && (
        <div className="space-y-2">
          <p className="text-sm font-medium">Sources</p>
          {response.citations.map((c, i) => (
            <div key={`${c.marker}-${i}`} ref={(el) => void (cards.current[i] = el)}>
              <SourceCard
                citation={c}
                passage={passages.get(c.marker)}
                open={!!open[i]}
                onOpenChange={(value) => setOpen((o) => ({ ...o, [i]: value }))}
              />
            </div>
          ))}
        </div>
      )}

      <p className="text-xs text-muted-foreground" data-testid="answer-meta">
        {[
          response.model ?? "no model call",
          fmtMs(response.timings_ms.total_ms),
          v ? `${v.citations_verified}/${v.citations_proposed} citations verified` : null,
          response.usage.prompt_tokens !== null
            ? `${response.usage.prompt_tokens} + ${response.usage.completion_tokens ?? 0} tokens`
            : null,
          `$${response.usage.cost_usd.toFixed(2)}`,
        ]
          .filter(Boolean)
          .join(" · ")}
      </p>
      <Checks response={response} />
    </div>
  );
}
