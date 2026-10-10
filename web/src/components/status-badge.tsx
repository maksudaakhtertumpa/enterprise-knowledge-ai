import { Badge } from "@/components/ui/badge";
import { statusText } from "@/lib/format";

const VARIANT = {
  ready: "success",
  answered: "success",
  processing: "secondary",
  pending: "outline",
  failed: "destructive",
  insufficient_evidence: "warning",
} as const;

export function StatusBadge({ status }: { status: string }) {
  const variant = VARIANT[status as keyof typeof VARIANT] ?? "outline";
  return <Badge variant={variant}>{statusText(status)}</Badge>;
}
