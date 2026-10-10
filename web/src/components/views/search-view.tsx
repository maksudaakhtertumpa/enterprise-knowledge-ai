"use client";

import { useState } from "react";
import { Loader2, Search } from "lucide-react";

import { DocumentFilter } from "@/components/document-filter";
import { ErrorAlert } from "@/components/error-alert";
import { PageHeader } from "@/components/page-header";
import { SearchResultCard } from "@/components/search-result";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Switch } from "@/components/ui/switch";
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group";
import { asApiError, useLoad } from "@/hooks/use-load";
import { type ApiError, api } from "@/lib/api";
import { fmtMs } from "@/lib/format";
import type { SearchMode, SearchResponse } from "@/lib/types";
import { cn } from "@/lib/utils";

const MODES: Record<SearchMode, string> = {
  dense: "Dense (semantic)",
  hybrid: "Hybrid (BM25 + dense, RRF)",
  bm25: "BM25 (keyword)",
};
const ORDER: SearchMode[] = ["dense", "hybrid", "bm25"];

export function SearchView() {
  const docs = useLoad(() => api.allDocuments(), "docs");
  const ready = (docs.data ?? []).filter((d) => d.status === "ready");
  const [query, setQuery] = useState("");
  const [mode, setMode] = useState<SearchMode>("dense");
  const [rerank, setRerank] = useState(true);
  const [compare, setCompare] = useState(false);
  const [topK, setTopK] = useState("5");
  const [selected, setSelected] = useState<string[]>([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<ApiError>();
  const [hint, setHint] = useState<string>();
  const [results, setResults] = useState<SearchResponse[]>([]);

  async function run(e: React.FormEvent) {
    e.preventDefault();
    if (!query.trim()) {
      setHint("Type a query first.");
      return;
    }
    setHint(undefined);
    setBusy(true);
    setError(undefined);
    try {
      const modes = compare ? ORDER : [mode];
      setResults(
        await Promise.all(
          modes.map((m) =>
            api.search({
              query: query.trim(),
              mode: m,
              rerank,
              top_k: Number(topK),
              document_ids: selected.length ? selected : undefined,
            }),
          ),
        ),
      );
    } catch (err) {
      setError(asApiError(err));
    } finally {
      setBusy(false);
    }
  }

  return (
    <div>
      <PageHeader
        title="Search"
        description="Find passages without generating an answer: check what the retriever finds, and compare keyword, semantic and hybrid search."
      />
      <form onSubmit={run} className="space-y-4 rounded-xl border p-4">
        <div className="flex gap-2">
          <Input
            aria-label="Query"
            placeholder="e.g. How many days of annual leave?"
            value={query}
            maxLength={1000}
            onChange={(e) => setQuery(e.target.value)}
          />
          <Button type="submit" disabled={busy}>
            {busy ? <Loader2 className="animate-spin" /> : <Search />}
            Search
          </Button>
        </div>
        <div className="flex flex-wrap items-center gap-x-6 gap-y-3">
          <ToggleGroup
            type="single"
            value={mode}
            onValueChange={(v) => v && setMode(v as SearchMode)}
            disabled={compare}
            aria-label="Retriever"
          >
            {ORDER.map((m) => (
              <ToggleGroupItem key={m} value={m}>
                {MODES[m]}
              </ToggleGroupItem>
            ))}
          </ToggleGroup>
          <div className="flex items-center gap-2">
            <Switch id="rerank" checked={rerank} onCheckedChange={setRerank} />
            <Label htmlFor="rerank">Rerank (cross-encoder)</Label>
          </div>
          <div className="flex items-center gap-2">
            <Switch id="compare" checked={compare} onCheckedChange={setCompare} />
            <Label htmlFor="compare">Compare all three</Label>
          </div>
          <Select value={topK} onValueChange={setTopK}>
            <SelectTrigger size="sm" aria-label="Number of results">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {["3", "5", "10", "20"].map((n) => (
                <SelectItem key={n} value={n}>
                  {n} results
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
          <DocumentFilter documents={ready} selected={selected} onChange={setSelected} />
        </div>
        {hint && <p className="text-sm text-muted-foreground">{hint}</p>}
      </form>

      {error && <ErrorAlert error={error} className="mt-6" />}
      {docs.data && ready.length === 0 && (
        <p className="mt-6 text-sm text-muted-foreground">
          No searchable documents yet. Upload some on the Documents page first.
        </p>
      )}

      {results.length > 0 && (
        <div className={cn("mt-6 grid gap-6", results.length > 1 && "lg:grid-cols-3")}>
          {results.map((res) => (
            <section key={res.mode} className="min-w-0 space-y-3" aria-label={MODES[res.mode]}>
              <div>
                <h2 className="font-semibold">{MODES[res.mode]}</h2>
                <p className="text-xs text-muted-foreground">
                  {res.results.length} result(s) · {fmtMs(res.timings_ms.total_ms)}
                  {[res.embedding_model, res.reranker_model].some(Boolean) &&
                    ` · ${[res.embedding_model, res.reranker_model].filter(Boolean).join(" + ")}`}
                </p>
              </div>
              {res.results.length === 0 && (
                <p className="text-sm text-muted-foreground">No matching passages.</p>
              )}
              {res.results.map((r) => (
                <SearchResultCard
                  key={r.chunk_id}
                  result={r}
                  mode={res.mode}
                  reranked={res.reranked}
                  compact={results.length > 1}
                />
              ))}
            </section>
          ))}
        </div>
      )}
    </div>
  );
}
