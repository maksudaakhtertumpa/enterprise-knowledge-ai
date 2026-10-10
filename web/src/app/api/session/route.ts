// Connect (POST), check (GET) and disconnect (DELETE) the browser session.
import { NextResponse, type NextRequest } from "next/server";

import { cleanKey } from "@/lib/format";
import { config } from "@/lib/server/config";
import { errorResponse, sameOrigin } from "@/lib/server/proxy";
import {
  COOKIE_NAME,
  cookieOptions,
  looksLikeKey,
  newSession,
  seal,
  unseal,
} from "@/lib/server/session";
import type { Workspace } from "@/lib/types";

export const dynamic = "force-dynamic";

export async function GET(request: NextRequest) {
  const session = unseal(request.cookies.get(COOKIE_NAME)?.value);
  if (!session) return errorResponse(401, "session_expired", "Not connected.");
  return NextResponse.json({ workspace: session.workspace });
}

export async function POST(request: NextRequest) {
  if (!sameOrigin(request)) return errorResponse(403, "forbidden", "Cross-site request blocked.");
  let key: unknown;
  try {
    key = ((await request.json()) as { key?: unknown }).key;
  } catch {
    return errorResponse(400, "validation_error", 'Send JSON: {"key": "eka_..."}');
  }
  key = typeof key === "string" ? cleanKey(key) : key;
  if (!looksLikeKey(key)) {
    // Not sent to the API at all: say so, so a bad paste isn't mistaken for a bad key.
    return errorResponse(400, "invalid_key_format", "That doesn't look like a complete API key.");
  }

  let upstream: Response;
  try {
    upstream = await fetch(`${config.apiUrl}/api/v1/workspace`, {
      headers: { "X-API-Key": key, Accept: "application/json" },
      cache: "no-store",
    });
  } catch {
    return errorResponse(502, "unreachable", "Cannot reach the API server.");
  }
  if (!upstream.ok) {
    // Pass the API's own error envelope through (401 for a wrong key).
    return new Response(upstream.body, {
      status: upstream.status,
      headers: { "content-type": upstream.headers.get("content-type") ?? "application/json" },
    });
  }
  const workspace = (await upstream.json()) as Workspace;
  const session = newSession(key, { id: workspace.id, name: workspace.name });
  const response = NextResponse.json({ workspace: session.workspace });
  response.cookies.set(COOKIE_NAME, seal(session), cookieOptions(config.sessionHours * 3600));
  return response;
}

export async function DELETE(request: NextRequest) {
  if (!sameOrigin(request)) return errorResponse(403, "forbidden", "Cross-site request blocked.");
  const response = new NextResponse(null, { status: 204 });
  response.cookies.set(COOKIE_NAME, "", cookieOptions(0));
  return response;
}
