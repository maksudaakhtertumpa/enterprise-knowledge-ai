// Backend-for-frontend proxy: the browser calls /api/eka/<path>; this server adds
// the workspace key from the sealed cookie and forwards to the FastAPI backend.
// Only the endpoints the UI uses are reachable, with the methods it uses.
import type { NextRequest } from "next/server";

import { config } from "@/lib/server/config";

const UUID = "[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}";

const ROUTES: [method: string, pattern: RegExp][] = [
  ["GET", /^workspace$/],
  ["GET", /^documents$/],
  ["POST", /^documents$/],
  ["GET", new RegExp(`^documents/${UUID}$`)],
  ["DELETE", new RegExp(`^documents/${UUID}$`)],
  ["GET", new RegExp(`^documents/${UUID}/chunks$`)],
  ["POST", new RegExp(`^documents/${UUID}/reprocess$`)],
  ["POST", /^search$/],
  ["POST", /^ask$/],
  ["GET", /^ask\/history$/],
  ["GET", new RegExp(`^ask/history/${UUID}$`)],
  ["GET", new RegExp(`^evidence/${UUID}$`)],
  ["GET", /^analytics$/],
];

export function isAllowed(method: string, segments: string[]): boolean {
  // Each segment must be plain (no "..", no encoded slashes, no empty parts).
  if (segments.length === 0 || segments.some((s) => !/^[A-Za-z0-9_-]+$/.test(s))) return false;
  const path = segments.join("/");
  return ROUTES.some(([m, re]) => m === method && re.test(path));
}

/**
 * State-changing requests must come from this site's own pages. SameSite=Strict
 * cookies already stop cross-site requests; this Origin check is a second layer.
 */
export function sameOrigin(request: Request): boolean {
  const origin = request.headers.get("origin");
  const host = request.headers.get("host");
  if (!origin || !host) return false;
  try {
    return new URL(origin).host === host;
  } catch {
    return false;
  }
}

export function errorResponse(status: number, code: string, message: string): Response {
  return Response.json({ error: { code, message, details: null, request_id: null } }, { status });
}

/** Forward one request to the API with the workspace key; pass back status and body. */
export async function forward(
  request: NextRequest,
  segments: string[],
  apiKey: string,
): Promise<Response> {
  const url = `${config.apiUrl}/api/v1/${segments.join("/")}${request.nextUrl.search}`;
  const headers = new Headers({ "X-API-Key": apiKey, Accept: "application/json" });
  const contentType = request.headers.get("content-type");
  if (contentType) headers.set("Content-Type", contentType);
  const requestId = request.headers.get("x-request-id");
  if (requestId) headers.set("X-Request-ID", requestId);

  const init: RequestInit & { duplex?: "half" } = {
    method: request.method,
    headers,
    cache: "no-store",
    redirect: "manual",
  };
  if (request.method !== "GET" && request.method !== "HEAD" && request.body) {
    init.body = request.body; // streamed: uploads are not buffered here
    init.duplex = "half";
  }

  let upstream: Response;
  try {
    upstream = await fetch(url, init);
  } catch {
    return errorResponse(502, "unreachable", "Cannot reach the API server.");
  }
  // Pass back only what the UI needs; never upstream cookies or server headers.
  const out = new Headers();
  for (const name of ["content-type", "x-request-id", "retry-after"]) {
    const value = upstream.headers.get(name);
    if (value) out.set(name, value);
  }
  out.set("Cache-Control", "no-store");
  return new Response(upstream.status === 204 ? null : upstream.body, {
    status: upstream.status,
    headers: out,
  });
}
