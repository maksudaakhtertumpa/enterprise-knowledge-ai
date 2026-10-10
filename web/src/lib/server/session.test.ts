// @vitest-environment node
import { describe, expect, it } from "vitest";

import { cookieOptions, looksLikeKey, newSession, seal, unseal } from "@/lib/server/session";

const KEY = "eka_" + "A".repeat(43);
const WS = { id: "w1", name: "Acme" };

describe("sealed session cookie", () => {
  it("round-trips and hides the key", () => {
    const token = seal(newSession(KEY, WS));
    expect(token).not.toContain(KEY);
    expect(Buffer.from(token, "base64url").toString("latin1")).not.toContain("eka_");
    expect(unseal(token)?.key).toBe(KEY);
    expect(unseal(token)?.workspace).toEqual(WS);
  });

  it("uses a fresh IV each time", () => {
    const s = newSession(KEY, WS);
    expect(seal(s)).not.toBe(seal(s));
  });

  it("rejects tampered, truncated, garbage and expired tokens", () => {
    const token = seal(newSession(KEY, WS));
    const raw = Buffer.from(token, "base64url");
    raw[20] = (raw[20] ?? 0) ^ 1;
    expect(unseal(raw.toString("base64url"))).toBeNull();
    expect(unseal(token.slice(0, 20))).toBeNull();
    expect(unseal("not-a-token")).toBeNull();
    expect(unseal("")).toBeNull();
    expect(unseal(undefined)).toBeNull();
    expect(unseal("x".repeat(5000))).toBeNull();
    const old = seal(newSession(KEY, WS, Date.now() - 9 * 3600 * 1000));
    expect(unseal(old)).toBeNull(); // sessions last 8 hours
  });

  it("sets an httpOnly, SameSite=Strict cookie", () => {
    expect(cookieOptions(60)).toMatchObject({ httpOnly: true, sameSite: "strict", path: "/" });
  });

  it("checks key shape before any network call", () => {
    expect(looksLikeKey(KEY)).toBe(true);
    for (const bad of [
      "",
      "eka_",
      "abc_" + "A".repeat(43),
      "eka_" + "A".repeat(200),
      "eka_a b c d e f g h i j k l",
      42,
      null,
    ]) {
      expect(looksLikeKey(bad)).toBe(false);
    }
  });
});
