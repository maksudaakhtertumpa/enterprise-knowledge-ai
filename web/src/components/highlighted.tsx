import { cn } from "@/lib/utils";
import { segments, type Span } from "@/lib/format";

/** Untrusted text with some spans marked. Rendered as React text only (escaped). */
export function Highlighted({
  text,
  spans,
  className,
}: {
  text: string;
  spans: Span[];
  className?: string;
}) {
  return (
    <span className={cn("whitespace-pre-wrap", className)}>
      {segments(text, spans).map((seg, i) =>
        seg.hit ? (
          <mark key={i} className="rounded-sm bg-highlight/70 px-0.5 text-foreground">
            {seg.text}
          </mark>
        ) : (
          <span key={i}>{seg.text}</span>
        ),
      )}
    </span>
  );
}
