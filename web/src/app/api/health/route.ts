// Backend readiness for the status panel. Public on the API too; no key involved.
import { config } from "@/lib/server/config";
import { errorResponse } from "@/lib/server/proxy";

export const dynamic = "force-dynamic";

export async function GET() {
  try {
    const upstream = await fetch(`${config.apiUrl}/health/ready`, {
      cache: "no-store",
      signal: AbortSignal.timeout(10_000),
    });
    const body = await upstream.json();
    return Response.json(body, { status: upstream.status === 200 ? 200 : 503 });
  } catch {
    return errorResponse(502, "unreachable", "Cannot reach the API server.");
  }
}
