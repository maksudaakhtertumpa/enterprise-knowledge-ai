"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { CheckCircle2, Eye, EyeOff, KeyRound, Loader2 } from "lucide-react";

import { ErrorAlert } from "@/components/error-alert";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { ApiError, api } from "@/lib/api";
import { KEY_LENGTH, cleanKey, keyProblem } from "@/lib/format";
import { cn } from "@/lib/utils";

export function ConnectForm({ expired }: { expired: boolean }) {
  const router = useRouter();
  const [key, setKey] = useState("");
  const [show, setShow] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<ApiError | undefined>(
    expired ? new ApiError(401, "session_expired", "Session ended") : undefined,
  );
  const cleaned = cleanKey(key);
  const problem = keyProblem(key);

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    if (!cleaned || problem) {
      setError(new ApiError(400, "invalid_key_format", "Not a complete key"));
      return;
    }
    setBusy(true);
    try {
      await api.connect(cleaned);
      setKey("");
      router.replace("/ask");
      router.refresh();
    } catch (err) {
      setError(err instanceof ApiError ? err : new ApiError(0, "client_error", String(err)));
      setBusy(false);
    }
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <KeyRound className="size-4" /> Connect to a workspace
        </CardTitle>
        <CardDescription>
          The key is checked by the API, then kept in an encrypted, httpOnly cookie. Page scripts
          can&apos;t read it, and it is never stored in the browser.
        </CardDescription>
      </CardHeader>
      <CardContent>
        <form onSubmit={submit} className="space-y-4">
          <div className="space-y-2">
            <Label htmlFor="api-key">Workspace API key</Label>
            <div className="flex gap-2">
              {/* A text field masked with CSS, not type="password": browsers' password
                  managers then don't autofill or offer to save it, which used to
                  replace or mangle pasted keys. */}
              <Input
                id="api-key"
                name="eka-workspace-key"
                type="text"
                inputMode="text"
                autoComplete="off"
                autoCorrect="off"
                autoCapitalize="none"
                spellCheck={false}
                data-1p-ignore
                data-lpignore="true"
                placeholder="eka_…"
                value={key}
                onChange={(e) => {
                  setKey(e.target.value);
                  setError(undefined);
                }}
                className={cn("font-mono", !show && "[-webkit-text-security:disc]")}
                aria-describedby="key-check"
              />
              <Button
                type="button"
                variant="outline"
                size="icon"
                onClick={() => setShow((s) => !s)}
                aria-label={show ? "Hide key" : "Show key"}
              >
                {show ? <EyeOff /> : <Eye />}
              </Button>
            </div>
            <p
              id="key-check"
              className="min-h-5 text-xs"
              data-testid="key-check"
              aria-live="polite"
            >
              {cleaned &&
                (problem ? (
                  <span className="text-destructive">{problem}</span>
                ) : (
                  <span className="flex items-center gap-1 text-muted-foreground">
                    <CheckCircle2 className="size-3.5 text-success" />
                    Looks like a complete key ({KEY_LENGTH} characters).
                  </span>
                ))}
            </p>
          </div>
          {error && <ErrorAlert error={error} />}
          <Button type="submit" disabled={busy}>
            {busy && <Loader2 className="animate-spin" />}
            Connect
          </Button>
        </form>
        <div className="mt-6 space-y-2 text-sm text-muted-foreground">
          <p>No key yet? In the project folder (venv active), run:</p>
          <pre className="overflow-x-auto rounded-md bg-muted px-3 py-2 font-mono text-xs">
            python -m app.cli create-workspace &quot;My team&quot;
          </pre>
          <p className="text-xs">The key is printed once. Only its hash is stored on the server.</p>
        </div>
      </CardContent>
    </Card>
  );
}
