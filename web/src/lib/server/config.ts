// Server-only settings for the web app (read from the environment of `next dev` / `next start`).
import { randomBytes } from "node:crypto";

function apiUrl(): string {
  const raw = process.env.EKA_API_URL ?? "http://localhost:8000";
  const url = new URL(raw); // throws on garbage, at first use
  if (url.protocol !== "http:" && url.protocol !== "https:") {
    throw new Error("EKA_API_URL must start with http:// or https://");
  }
  return raw.replace(/\/+$/, "");
}

// Next.js bundles route handlers and pages separately, so a plain module variable
// would exist twice in one process. A global symbol is shared by both.
const SECRET = Symbol.for("eka.web.sessionSecret");
type WithSecret = typeof globalThis & { [SECRET]?: string };

/**
 * Key for sealing the session cookie. Set EKA_WEB_SESSION_SECRET (32+ characters)
 * to keep sessions across restarts; otherwise a random one is made per process,
 * which is secure but logs everyone out when the web server restarts.
 */
export function sessionSecret(): string {
  const g = globalThis as WithSecret;
  if (g[SECRET]) return g[SECRET];
  const configured = process.env.EKA_WEB_SESSION_SECRET;
  if (configured && configured.length < 32) {
    throw new Error("EKA_WEB_SESSION_SECRET must be at least 32 characters");
  }
  g[SECRET] = configured || randomBytes(32).toString("base64url");
  return g[SECRET];
}

export const config = {
  get apiUrl() {
    return apiUrl();
  },
  // Secure cookies need HTTPS. Plain http://localhost is the local default; set
  // EKA_WEB_COOKIE_SECURE=true whenever the site is served over HTTPS (Phase 8).
  get cookieSecure() {
    return process.env.EKA_WEB_COOKIE_SECURE === "true";
  },
  sessionHours: 8,
};
