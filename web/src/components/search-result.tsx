import { Highlighted } from "@/components/highlighted";
import { Card, CardContent } from "@/components/ui/card";
import { sourceLabel, termSpans, truncate } from "@/lib/format";
import type { SearchMode, SearchResult } from "@/lib/types";

export function SearchResultCard({
  result,
  mode,
  reranked,
  compact = false,
}: {
  result: SearchResult;
  mode: SearchMode;
  reranked: boolean;
  compact?: boolean;
}) {
  const text = compact ? truncate(result.text, 400) : result.text;
  const ranks = result.retriever_ranks;
  const meta = [
    `score ${result.score.toFixed(4)} ${reranked ? "(cross-encoder)" : `(${mode})`}`,
    result.retrieval_rank !== null ? `was #${result.retrieval_rank} before reranking` : null,
    ranks
      ? `BM25 ${ranks.bm25 ? `#${ranks.bm25}` : "–"} · dense ${ranks.dense ? `#${ranks.dense}` : "–"}`
      : null,
  ].filter(Boolean);
  return (
    <Card className="gap-2 py-4" data-testid="search-result">
      <CardContent className="space-y-2 px-4">
        <p className="text-sm font-medium">
          #{result.rank} · {sourceLabel(result.source)}
        </p>
        <p className="text-xs text-muted-foreground">{meta.join(" · ")}</p>
        <p className="text-sm leading-relaxed">
          <Highlighted text={text} spans={termSpans(text, result.matched_terms)} />
        </p>
      </CardContent>
    </Card>
  );
}
