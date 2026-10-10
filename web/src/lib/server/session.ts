// The browser never sees the workspace API key. After the key is checked against
// the API, it is sealed (AES-256-GCM, authenticated) into an httpOnly, SameSite=Strict
// cookie that page JavaScript cannot read. Only this server can open it.
import { createCipheriv, createDecipheriv, createHash, randomBytes } from "node:crypto";

import type { Workspace } from "@/lib/types";
import { config, sessionSecret } from "@/lib/server/config";

export const COOKIE_NAME = "eka_session";
const AAD = Buffer.from("eka-session-v1");

export interface SessionData {
  key: string;
  workspace: Workspace;
  exp: number; // epoch milliseconds
}

function aesKey(): Buffer {
  return createHash("sha256").update(sessionSecret()).digest();
}

export function seal(data: SessionData): string {
  const iv = randomBytes(12);
  const cipher = createCipheriv("aes-256-gcm", aesKey(), iv);
  cipher.setAAD(AAD);
  const body = Buffer.concat([cipher.update(JSON.stringify(data), "utf8"), cipher.final()]);
  return Buffer.concat([iv, body, cipher.getAuthTag()]).toString("base64url");
}

/** The session, or null if the cookie is missing, tampered with, or expired. */
export function unseal(token: string | undefined | null, now = Date.now()): SessionData | null {
  if (!token || token.length > 4096) return null;
  try {
    const raw = Buffer.from(token, "base64url");
    if (raw.length < 12 + 16 + 1) return null;
    const decipher = createDecipheriv("aes-256-gcm", aesKey(), raw.subarray(0, 12));
    decipher.setAAD(AAD);
    decipher.setAuthTag(raw.subarray(raw.length - 16));
    const json = Buffer.concat([
      decipher.update(raw.subarray(12, raw.length - 16)),
      decipher.final(),
    ]).toString("utf8");
    const data = JSON.parse(json) as SessionData;
    if (typeof data.key !== "string" || typeof data.exp !== "number" || data.exp <= now) {
      return null;
    }
    return data;
  } catch {
    return null; // wrong key, tampered bytes, bad JSON: all "no session"
  }
}

export function newSession(key: string, workspace: Workspace, now = Date.now()): SessionData {
  return { key, workspace, exp: now + config.sessionHours * 3600 * 1000 };
}

export function cookieOptions(maxAgeSeconds: number) {
  return {
    httpOnly: true,
    sameSite: "strict" as const,
    secure: config.cookieSecure,
    path: "/",
    maxAge: maxAgeSeconds,
  };
}

/**
 * Exact key shape, checked before any network call: "eka_" + 43 URL-safe characters
 * (app/core/security.py generates exactly that), so a partial or doubled paste is
 * caught here with a clear message instead of a vague "not accepted".
 */
export function looksLikeKey(key: unknown): key is string {
  return typeof key === "string" && /^eka_[A-Za-z0-9_-]{43}$/.test(key);
}
