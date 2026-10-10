"use client";

import { CheckCircle2, CircleAlert, RefreshCw, XCircle } from "lucide-react";

import { ErrorAlert } from "@/components/error-alert";
import { Button } from "@/components/ui/button";
import { Card, CardAction, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { useLoad } from "@/hooks/use-load";
import { api } from "@/lib/api";

const NAMES: Record<string, string> = {
  data_dir: "Data folder",
  database: "Database (SQLite)",
  vector_store: "Vector store (ChromaDB)",
  embedding_model: "Embedding model",
  reranker: "Reranker",
  llm: "Local LLM (Ollama)",
};

export function ReadinessPanel() {
  const { data, error, loading, reload } = useLoad(api.health, "health");
  return (
    <Card>
      <CardHeader>
        <CardTitle>Backend status</CardTitle>
        <CardAction>
          <Button variant="ghost" size="sm" onClick={() => reload()} disabled={loading}>
            <RefreshCw className={loading ? "animate-spin" : ""} />
            Refresh
          </Button>
        </CardAction>
      </CardHeader>
      <CardContent className="space-y-2 text-sm">
        {error && <ErrorAlert error={error} />}
        {!data && !error && <Skeleton className="h-28 w-full" />}
        {data && (
          <>
            <p className="font-medium" data-testid="readiness">
              {data.status === "ready" ? "The API is ready." : "The API is running but not ready."}
            </p>
            <ul className="space-y-1.5">
              {data.checks.map((c) => (
                <li key={c.name} className="flex items-start gap-2">
                  {c.ok ? (
                    <CheckCircle2 className="mt-0.5 size-4 shrink-0 text-success" />
                  ) : c.required ? (
                    <XCircle className="mt-0.5 size-4 shrink-0 text-destructive" />
                  ) : (
                    <CircleAlert className="mt-0.5 size-4 shrink-0 text-warning" />
                  )}
                  <span>
                    {NAMES[c.name] ?? c.name}
                    {!c.required && <span className="text-muted-foreground"> (optional)</span>}
                    {c.detail && <span className="text-muted-foreground"> — {c.detail}</span>}
                    {c.name === "llm" && !c.ok && (
                      <span className="block text-xs text-muted-foreground">
                        Answers need Ollama: open the Ollama app. Upload and search work without it.
                      </span>
                    )}
                  </span>
                </li>
              ))}
            </ul>
          </>
        )}
      </CardContent>
    </Card>
  );
}
