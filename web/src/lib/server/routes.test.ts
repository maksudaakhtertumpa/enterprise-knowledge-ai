// @vitest-environment node
// The session and proxy route handlers, with the FastAPI backend replaced by a fake fetch.
import { NextRequest } from "next/server";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import * as proxyRoute from "@/app/api/eka/[...path]/route";
import * as sessionRoute from "@/app/api/session/route";
import { isAllowed } from "@/lib/server/proxy";
import { COOKIE_NAME, newSession, seal } from "@/lib/server/session";

const KEY = "eka_" + "K".repeat(43);
const SITE = "http://localhost:3000";
const DOC = "123e4567-e89b-12d3-a456-426614174000";

type Call = { url: string; init: RequestInit };
let calls: Call[];

function fakeBackend(handler: (url: string, init: RequestInit) => Response) {
  calls = [];
  vi.stubGlobal(
    "fetch",
    vi.fn(async (url: string, init: RequestInit) => {
      calls.push({ url, init });
      return handler(url, init);
    }),
  );
}

function req(
  path: string,
  init: { method?: string; body?: string; origin?: string | null; cookie?: string } = {},
) {
  const headers = new Headers({ host: "localhost:3000" });
  if (init.origin !== null) headers.set("origin", init.origin ?? SITE);
  if (init.body) headers.set("content-type", "application/json");
  if (init.cookie) headers.set("cookie", `${COOKIE_NAME}=${init.cookie}`);
  return new NextRequest(`${SITE}${path}`, {
    method: init.method ?? "GET",
    headers,
    body: init.body,
  });
}

const ctx = (path: string[]) => ({ params: Promise.resolve({ path }) });
const cookie = () => seal(newSession(KEY, { id: "w1", name: "Acme" }));

beforeEach(() => {
  process.env.EKA_API_URL = "http://api.test:8000";
});
afterEach(() => vi.unstubAllGlobals());

describe("POST /api/session", () => {
  it("checks the key with the API and sets a sealed httpOnly cookie", async () => {
    fakeBackend(() => Response.json({ id: "w1", name: "Acme" }));
    const resp = await sessionRoute.POST(
      req("/api/session", { method: "POST", body: JSON.stringify({ key: KEY }) }),
    );
    expect(resp.status).toBe(200);
    expect(await resp.json()).toEqual({ workspace: { id: "w1", name: "Acme" } });
    expect(calls[0]?.url).toBe("http://api.test:8000/api/v1/workspace");
    expect(new Headers(calls[0]?.init.headers).get("X-API-Key")).toBe(KEY);
    const setCookie = resp.headers.get("set-cookie") ?? "";
    expect(setCookie).toMatch(/HttpOnly/i);
    expect(setCookie).toMatch(/SameSite=strict/i);
    expect(setCookie).not.toContain(KEY);
  });

  it("passes a rejected key through as 401 without a cookie", async () => {
    fakeBackend(() =>
      Response.json(
        { error: { code: "unauthorized", message: "Missing or invalid API key" } },
        { status: 401 },
      ),
    );
    const resp = await sessionRoute.POST(
      req("/api/session", { method: "POST", body: JSON.stringify({ key: KEY }) }),
    );
    expect(resp.status).toBe(401);
    expect(resp.headers.get("set-cookie")).toBeNull();
  });

  it("refuses malformed keys without calling the API, with its own error code", async () => {
    fakeBackend(() => Response.json({}));
    for (const key of ["hello", KEY.slice(0, 16), `${KEY}${KEY}`]) {
      const resp = await sessionRoute.POST(
        req("/api/session", { method: "POST", body: JSON.stringify({ key }) }),
      );
      expect(resp.status).toBe(400);
      expect((await resp.json()).error.code).toBe("invalid_key_format");
    }
    expect(calls).toHaveLength(0);
  });

  it("strips whitespace a copy-paste adds around the key", async () => {
    fakeBackend(() => Response.json({ id: "w1", name: "Acme" }));
    const resp = await sessionRoute.POST(
      req("/api/session", { method: "POST", body: JSON.stringify({ key: ` ${KEY}\n` }) }),
    );
    expect(resp.status).toBe(200);
    expect(new Headers(calls[0]?.init.headers).get("X-API-Key")).toBe(KEY);
  });

  it("refuses cross-site and Origin-less requests", async () => {
    fakeBackend(() => Response.json({ id: "w1", name: "Acme" }));
    for (const origin of ["http://evil.example", null]) {
      const resp = await sessionRoute.POST(
        req("/api/session", { method: "POST", body: JSON.stringify({ key: KEY }), origin }),
      );
      expect(resp.status).toBe(403);
    }
    expect(calls).toHaveLength(0);
  });

  it("reports an unreachable API as 502 unreachable", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => Promise.reject(new TypeError("fetch failed"))),
    );
    const resp = await sessionRoute.POST(
      req("/api/session", { method: "POST", body: JSON.stringify({ key: KEY }) }),
    );
    expect(resp.status).toBe(502);
    expect((await resp.json()).error.code).toBe("unreachable");
  });

  it("DELETE clears the cookie", async () => {
    const resp = await sessionRoute.DELETE(req("/api/session", { method: "DELETE" }));
    expect(resp.status).toBe(204);
    expect(resp.headers.get("set-cookie")).toMatch(/Max-Age=0/i);
  });
});

describe("/api/eka proxy", () => {
  it("adds the key from the session and returns only safe headers", async () => {
    fakeBackend(() =>
      Response.json(
        { items: [] },
        { headers: { "x-request-id": "r1", "set-cookie": "upstream=1", server: "uvicorn" } },
      ),
    );
    const resp = await proxyRoute.GET(
      req(`/api/eka/documents?limit=5`, { cookie: cookie() }),
      ctx(["documents"]),
    );
    expect(resp.status).toBe(200);
    expect(calls[0]?.url).toBe("http://api.test:8000/api/v1/documents?limit=5");
    expect(new Headers(calls[0]?.init.headers).get("X-API-Key")).toBe(KEY);
    expect(resp.headers.get("x-request-id")).toBe("r1");
    expect(resp.headers.get("set-cookie")).toBeNull();
    expect(resp.headers.get("server")).toBeNull();
    expect(resp.headers.get("cache-control")).toBe("no-store");
  });

  it("needs a session", async () => {
    fakeBackend(() => Response.json({}));
    const resp = await proxyRoute.GET(req("/api/eka/workspace"), ctx(["workspace"]));
    expect(resp.status).toBe(401);
    expect(calls).toHaveLength(0);
  });

  it("forwards nothing outside the allowlist", async () => {
    fakeBackend(() => Response.json({}));
    const resp = await proxyRoute.GET(req("/api/eka/admin", { cookie: cookie() }), ctx(["admin"]));
    expect(resp.status).toBe(404);
    expect(calls).toHaveLength(0);
  });

  it("blocks state changes from other sites", async () => {
    fakeBackend(() => Response.json({}));
    const resp = await proxyRoute.POST(
      req("/api/eka/ask", {
        method: "POST",
        body: "{}",
        cookie: cookie(),
        origin: "http://evil.example",
      }),
      ctx(["ask"]),
    );
    expect(resp.status).toBe(403);
    expect(calls).toHaveLength(0);
  });

  it("ends the browser session when the API rejects the key (revoked)", async () => {
    fakeBackend(() =>
      Response.json(
        { error: { code: "unauthorized", message: "Missing or invalid API key" } },
        { status: 401 },
      ),
    );
    const resp = await proxyRoute.GET(
      req("/api/eka/workspace", { cookie: cookie() }),
      ctx(["workspace"]),
    );
    expect(resp.status).toBe(401);
    expect(resp.headers.get("set-cookie")).toMatch(/Max-Age=0/i);
  });

  it("streams POST bodies through and keeps the API's status", async () => {
    fakeBackend(() => new Response(null, { status: 204 }));
    const resp = await proxyRoute.DELETE(
      req(`/api/eka/documents/${DOC}`, { method: "DELETE", cookie: cookie() }),
      ctx(["documents", DOC]),
    );
    expect(resp.status).toBe(204);
    expect(calls[0]?.init.method).toBe("DELETE");
  });
});

describe("allowlist", () => {
  it.each([
    ["GET", ["workspace"]],
    ["GET", ["documents"]],
    ["POST", ["documents"]],
    ["GET", ["documents", DOC, "chunks"]],
    ["POST", ["documents", DOC, "reprocess"]],
    ["DELETE", ["documents", DOC]],
    ["POST", ["search"]],
    ["POST", ["ask"]],
    ["GET", ["ask", "history", DOC]],
    ["GET", ["evidence", DOC]],
    ["GET", ["analytics"]],
  ])("allows %s %j", (method, path) => {
    expect(isAllowed(method, path)).toBe(true);
  });

  it.each([
    ["DELETE", ["documents"]],
    ["GET", ["documents", "not-a-uuid"]],
    ["GET", ["documents", "..", "health"]],
    ["GET", ["documents", `${DOC}%2F..`]],
    ["GET", ["health", "ready"]],
    ["POST", ["workspace"]],
    ["GET", []],
    ["GET", ["ask", ""]],
  ])("refuses %s %j", (method, path) => {
    expect(isAllowed(method, path)).toBe(false);
  });
});
