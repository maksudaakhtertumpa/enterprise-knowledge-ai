import { AlertTriangle } from "lucide-react";

import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import type { ApiError } from "@/lib/api";
import { friendlyError } from "@/lib/format";

/** Our own hints mark commands with `backticks`; show those as code. */
function Hint({ text }: { text: string }) {
  return (
    <p>
      {text.split("`").map((part, i) =>
        i % 2 === 1 ? (
          <code key={i} className="rounded bg-muted px-1 py-0.5 font-mono text-xs">
            {part}
          </code>
        ) : (
          <span key={i}>{part}</span>
        ),
      )}
    </p>
  );
}

export function ErrorAlert({ error, className }: { error: ApiError; className?: string }) {
  const { title, hint } = friendlyError(error.code, error.message);
  return (
    <Alert variant="destructive" className={className}>
      <AlertTriangle />
      <AlertTitle className="line-clamp-none">{title}</AlertTitle>
      <AlertDescription>
        {hint && <Hint text={hint} />}
        {error.requestId && (
          <p className="text-xs opacity-80">
            Request ID: {error.requestId} (search the API log for it)
          </p>
        )}
      </AlertDescription>
    </Alert>
  );
}
