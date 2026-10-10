"use client";

import { useEffect, useRef, useState } from "react";
import { CheckCircle2, Info, Loader2, RotateCw, Trash2, Upload, XCircle } from "lucide-react";

import { ErrorAlert } from "@/components/error-alert";
import { PageHeader } from "@/components/page-header";
import { StatusBadge } from "@/components/status-badge";
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
  AlertDialogTrigger,
} from "@/components/ui/alert-dialog";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { Skeleton } from "@/components/ui/skeleton";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { asApiError, useLoad } from "@/hooks/use-load";
import { type ApiError, api } from "@/lib/api";
import { fmtTime, friendlyError, humanSize, sourceLabel } from "@/lib/format";
import type { DocumentOut } from "@/lib/types";
import { cn } from "@/lib/utils";

const BUSY = new Set(["pending", "processing"]);
const ALLOWED = [".pdf", ".docx", ".txt"];
const MAX_MB = 20; // the API enforces its own limit (EKA_MAX_UPLOAD_MB) either way
const CHUNKS_PER_PAGE = 10;

interface UploadResult {
  name: string;
  kind: "ok" | "duplicate" | "error";
  message: string;
}

function Uploader({ onUploaded }: { onUploaded: () => void }) {
  const input = useRef<HTMLInputElement>(null);
  const [dragging, setDragging] = useState(false);
  const [progress, setProgress] = useState<{ done: number; total: number }>();
  const [results, setResults] = useState<UploadResult[]>([]);

  async function upload(files: File[]) {
    if (files.length === 0) return;
    const out: UploadResult[] = [];
    setProgress({ done: 0, total: files.length });
    for (const [i, file] of files.entries()) {
      const ext = file.name.slice(file.name.lastIndexOf(".")).toLowerCase();
      if (!ALLOWED.includes(ext)) {
        out.push({ name: file.name, kind: "error", message: "Only PDF, DOCX and TXT files." });
      } else if (file.size > MAX_MB * 1024 * 1024) {
        out.push({ name: file.name, kind: "error", message: `Larger than ${MAX_MB} MB.` });
      } else {
        try {
          await api.upload(file);
          out.push({
            name: file.name,
            kind: "ok",
            message: "uploaded, processing in the background",
          });
        } catch (e) {
          const err = asApiError(e);
          const { title, hint } = friendlyError(err.code, err.message);
          out.push({
            name: file.name,
            kind: err.status === 409 ? "duplicate" : "error",
            message: hint ? `${title} ${hint}` : title,
          });
        }
      }
      setProgress({ done: i + 1, total: files.length });
    }
    setResults(out);
    setProgress(undefined);
    if (input.current) input.current.value = "";
    onUploaded();
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle>Add documents</CardTitle>
        <CardDescription>
          PDF, DOCX or TXT, up to {MAX_MB} MB each. Text-based PDFs only (scanned images are not
          read).
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        <label
          htmlFor="file-input"
          onDragOver={(e) => {
            e.preventDefault();
            setDragging(true);
          }}
          onDragLeave={() => setDragging(false)}
          onDrop={(e) => {
            e.preventDefault();
            setDragging(false);
            void upload(Array.from(e.dataTransfer.files));
          }}
          className={cn(
            "flex cursor-pointer flex-col items-center justify-center gap-2 rounded-lg border-2 border-dashed px-4 py-8 text-center text-sm transition-colors",
            dragging ? "border-primary bg-accent" : "hover:bg-accent/50",
            progress && "pointer-events-none opacity-60",
          )}
        >
          <Upload className="size-6 text-muted-foreground" />
          <span>
            <span className="font-medium">Choose files</span> or drag them here
          </span>
        </label>
        <input
          id="file-input"
          ref={input}
          type="file"
          multiple
          accept={ALLOWED.join(",")}
          className="sr-only"
          data-testid="file-input"
          onChange={(e) => void upload(Array.from(e.target.files ?? []))}
        />
        {progress && (
          <div className="space-y-1">
            <Progress value={(progress.done / progress.total) * 100} />
            <p className="text-xs text-muted-foreground">
              Uploading {progress.done}/{progress.total}…
            </p>
          </div>
        )}
        {results.length > 0 && (
          <ul className="space-y-1 text-sm" data-testid="upload-results">
            {results.map((r, i) => (
              <li key={i} className="flex items-start gap-2">
                {r.kind === "ok" ? (
                  <CheckCircle2 className="mt-0.5 size-4 shrink-0 text-success" />
                ) : r.kind === "duplicate" ? (
                  <Info className="mt-0.5 size-4 shrink-0 text-muted-foreground" />
                ) : (
                  <XCircle className="mt-0.5 size-4 shrink-0 text-destructive" />
                )}
                <span>
                  <span className="font-medium">{r.name}</span>: {r.message}
                </span>
              </li>
            ))}
          </ul>
        )}
      </CardContent>
    </Card>
  );
}

function Chunks({ doc }: { doc: DocumentOut }) {
  const [page, setPage] = useState(0);
  const pages = Math.max(1, Math.ceil(doc.chunk_count / CHUNKS_PER_PAGE));
  const chunks = useLoad(
    () => api.chunks(doc.id, CHUNKS_PER_PAGE, page * CHUNKS_PER_PAGE),
    `${doc.id}:${page}:${doc.updated_at}`,
  );
  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between">
        <p className="text-sm font-medium">Chunks ({doc.chunk_count})</p>
        {pages > 1 && (
          <div className="flex items-center gap-2 text-xs">
            <Button
              size="sm"
              variant="outline"
              disabled={page === 0}
              onClick={() => setPage(page - 1)}
            >
              Previous
            </Button>
            <span className="text-muted-foreground">
              {page + 1} / {pages}
            </span>
            <Button
              size="sm"
              variant="outline"
              disabled={page >= pages - 1}
              onClick={() => setPage(page + 1)}
            >
              Next
            </Button>
          </div>
        )}
      </div>
      {chunks.error && <ErrorAlert error={chunks.error} />}
      {chunks.loading && !chunks.data && <Skeleton className="h-32 w-full" />}
      {chunks.data?.items.map((c) => (
        <div key={c.id} className="space-y-1 rounded-md border p-3" data-testid="chunk">
          <p className="text-xs text-muted-foreground">
            {sourceLabel({
              filename: `#${c.ordinal}`,
              page_start: c.page_start,
              page_end: c.page_end,
              heading: c.heading,
            })}{" "}
            · {c.token_count} tokens · chars {c.char_start}–{c.char_end}
          </p>
          <p className="text-sm leading-relaxed whitespace-pre-wrap">{c.text}</p>
        </div>
      ))}
    </div>
  );
}

function Details({ doc, onChanged }: { doc: DocumentOut; onChanged: () => void }) {
  const [error, setError] = useState<ApiError>();
  const [busy, setBusy] = useState(false);

  async function act(fn: () => Promise<unknown>) {
    setBusy(true);
    try {
      await fn();
      setError(undefined);
      onChanged();
    } catch (e) {
      setError(asApiError(e));
    } finally {
      setBusy(false);
    }
  }

  return (
    <Card data-testid="document-details">
      <CardHeader>
        <CardTitle className="flex flex-wrap items-center gap-2 break-all">
          {doc.filename} <StatusBadge status={doc.status} />
        </CardTitle>
        {doc.title && <CardDescription>Title: {doc.title}</CardDescription>}
      </CardHeader>
      <CardContent className="space-y-5">
        <div className="grid grid-cols-2 gap-4 sm:grid-cols-4">
          {[
            ["Pages", doc.page_count ?? "–"],
            ["Chunks", doc.chunk_count],
            ["Characters", doc.char_count?.toLocaleString() ?? "–"],
            ["Size", humanSize(doc.size_bytes)],
          ].map(([label, value]) => (
            <div key={String(label)}>
              <p className="text-xs text-muted-foreground">{label}</p>
              <p className="text-lg font-semibold">{value}</p>
            </div>
          ))}
        </div>
        <p className="text-xs text-muted-foreground">
          Uploaded {fmtTime(doc.created_at)} · processed {fmtTime(doc.processed_at)} · SHA-256{" "}
          {doc.sha256.slice(0, 12)}…
        </p>
        {doc.status === "failed" && (
          <p className="text-sm text-destructive">
            Processing failed: {doc.error ?? "unknown error"}
          </p>
        )}
        {error && <ErrorAlert error={error} />}
        <div className="flex gap-2">
          <Button
            variant="outline"
            size="sm"
            disabled={busy || BUSY.has(doc.status)}
            onClick={() => act(() => api.reprocess(doc.id))}
          >
            <RotateCw /> Reprocess
          </Button>
          <AlertDialog>
            <AlertDialogTrigger asChild>
              <Button variant="outline" size="sm" disabled={busy || doc.status === "processing"}>
                <Trash2 /> Delete
              </Button>
            </AlertDialogTrigger>
            <AlertDialogContent>
              <AlertDialogHeader>
                <AlertDialogTitle>Delete {doc.filename}?</AlertDialogTitle>
                <AlertDialogDescription>
                  The document, its chunks and its search index entries are removed. This cannot be
                  undone.
                </AlertDialogDescription>
              </AlertDialogHeader>
              <AlertDialogFooter>
                <AlertDialogCancel>Cancel</AlertDialogCancel>
                <AlertDialogAction
                  className="bg-destructive text-white hover:bg-destructive/90"
                  onClick={() => act(() => api.deleteDocument(doc.id))}
                >
                  Delete
                </AlertDialogAction>
              </AlertDialogFooter>
            </AlertDialogContent>
          </AlertDialog>
        </div>
        {doc.status === "ready" && doc.chunk_count > 0 && <Chunks doc={doc} />}
      </CardContent>
    </Card>
  );
}

export function DocumentsView() {
  const docs = useLoad(() => api.allDocuments(), "documents");
  const [selectedId, setSelectedId] = useState<string>();
  const items = docs.data ?? [];
  const busy = items.some((d) => BUSY.has(d.status));
  const { reload } = docs;

  // Refresh every 2 s while anything is processing; stop when it's done.
  useEffect(() => {
    if (!busy) return;
    const t = setInterval(() => reload(), 2000);
    return () => clearInterval(t);
  }, [busy, reload]);

  const selected = items.find((d) => d.id === selectedId);
  const counts = {
    ready: items.filter((d) => d.status === "ready").length,
    busy: items.filter((d) => BUSY.has(d.status)).length,
    failed: items.filter((d) => d.status === "failed").length,
  };

  return (
    <div className="space-y-6">
      <PageHeader
        title="Documents"
        description="Upload, inspect and remove the documents this workspace searches."
      />
      <Uploader onUploaded={() => reload()} />

      <section className="space-y-3">
        <div className="flex flex-wrap items-center justify-between gap-2">
          <h2 className="font-semibold">Your documents</h2>
          {items.length > 0 && (
            <p
              className="flex items-center gap-2 text-xs text-muted-foreground"
              data-testid="doc-counts"
            >
              {busy && <Loader2 className="size-3.5 animate-spin" />}
              {items.length} document(s): {counts.ready} ready
              {counts.busy > 0 && `, ${counts.busy} processing (refreshing every 2 s)`}
              {counts.failed > 0 && `, ${counts.failed} failed`}
            </p>
          )}
        </div>
        {docs.error && <ErrorAlert error={docs.error} />}
        {docs.loading && !docs.data && <Skeleton className="h-40 w-full" />}
        {docs.data && items.length === 0 && (
          <p className="rounded-lg border border-dashed p-6 text-center text-sm text-muted-foreground">
            No documents yet. Upload a PDF, DOCX or TXT file above.
          </p>
        )}
        {items.length > 0 && (
          <div className="rounded-lg border">
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>File</TableHead>
                  <TableHead>Status</TableHead>
                  <TableHead className="hidden sm:table-cell">Type</TableHead>
                  <TableHead className="hidden sm:table-cell">Pages</TableHead>
                  <TableHead>Chunks</TableHead>
                  <TableHead className="hidden md:table-cell">Size</TableHead>
                  <TableHead className="hidden md:table-cell">Uploaded</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {items.map((d) => (
                  <TableRow
                    key={d.id}
                    data-state={d.id === selectedId ? "selected" : undefined}
                    className="cursor-pointer"
                    onClick={() => setSelectedId(d.id)}
                  >
                    <TableCell className="max-w-56 truncate font-medium">
                      <button
                        type="button"
                        className="truncate text-left hover:underline"
                        onClick={() => setSelectedId(d.id)}
                      >
                        {d.filename}
                      </button>
                    </TableCell>
                    <TableCell>
                      <StatusBadge status={d.status} />
                    </TableCell>
                    <TableCell className="hidden uppercase sm:table-cell">{d.file_type}</TableCell>
                    <TableCell className="hidden sm:table-cell">{d.page_count ?? "–"}</TableCell>
                    <TableCell>{d.chunk_count}</TableCell>
                    <TableCell className="hidden md:table-cell">
                      {humanSize(d.size_bytes)}
                    </TableCell>
                    <TableCell className="hidden md:table-cell">{fmtTime(d.created_at)}</TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </div>
        )}
        {items.length > 0 && !selected && (
          <p className="text-xs text-muted-foreground">
            Select a document to see its details and chunks.
          </p>
        )}
      </section>

      {selected && (
        <Details
          key={selected.id}
          doc={selected}
          onChanged={() => {
            reload();
          }}
        />
      )}
    </div>
  );
}
