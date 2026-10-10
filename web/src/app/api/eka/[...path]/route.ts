// Authenticated proxy to the FastAPI backend (see lib/server/proxy.ts).
import { NextResponse, type NextRequest } from "next/server";

import { errorResponse, forward, isAllowed, sameOrigin } from "@/lib/server/proxy";
import { COOKIE_NAME, cookieOptions, unseal } from "@/lib/server/session";

export const dynamic = "force-dynamic";

type Context = { params: Promise<{ path: string[] }> };

async function handle(request: NextRequest, context: Context): Promise<Response> {
  const { path } = await context.params;
  if (!isAllowed(request.method, path)) {
    return errorResponse(404, "not_found", "Not found");
  }
  if (request.method !== "GET" && !sameOrigin(request)) {
    return errorResponse(403, "forbidden", "Cross-site request blocked.");
  }
  const session = unseal(request.cookies.get(COOKIE_NAME)?.value);
  if (!session) return errorResponse(401, "session_expired", "Not connected.");

  const response = await forward(request, path, session.key);
  if (response.status === 401) {
    // The key was revoked on the server: end the browser session too.
    const ended = new NextResponse(response.body, response);
    ended.cookies.set(COOKIE_NAME, "", cookieOptions(0));
    return ended;
  }
  return response;
}

export const GET = handle;
export const POST = handle;
export const DELETE = handle;
