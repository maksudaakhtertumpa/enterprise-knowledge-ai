import { describe, expect, it } from "vitest";

import {
  answerParts,
  cleanKey,
  keyProblem,
  findQuote,
  fmtMs,
  friendlyError,
  humanSize,
  mergeSpans,
  pct,
  reasonText,
  segments,
  sourceLabel,
  termSpans,
  truncate,
} from "@/lib/format";
import type { Citation } from "@/lib/types";

const SRC = {
  document_id: "d",
  filename: "handbook.pdf",
  title: null,
  file_type: "pdf" as const,
  heading: "Leave",
  page_start: 3,
  page_end: 3,
};
const CIT: Citation[] = [
  {
    marker: 1,
    chunk_id: "c",
    quote: "14 working days",
    match: "exact",
    quote_char_start: null,
    quote_char_end: null,
    source: SRC,
  },
];

describe("sourceLabel", () => {
  it("formats file, pages and section", () => {
    expect(sourceLabel(SRC)).toBe("handbook.pdf · p. 3 · Leave");
    expect(sourceLabel({ filename: "a.pdf", page_start: 2, page_end: 4 })).toBe("a.pdf · pp. 2–4");
    expect(sourceLabel({ filename: "notes.txt" })).toBe("notes.txt");
    expect(sourceLabel(null)).toBe("unknown source");
    expect(sourceLabel({ filename: "x", heading: "h".repeat(200) }).length).toBeLessThan(70);
  });
});

describe("answerParts", () => {
  it("turns verified markers into chips and leaves others as text", () => {
    const parts = answerParts("Staff get 14 working days [1]. Other claim [7].", CIT);
    expect(parts).toEqual([
      { kind: "text", text: "Staff get 14 working days " },
      { kind: "cite", marker: 1, label: "handbook.pdf · p. 3 · Leave" },
      { kind: "text", text: ". Other claim [7]." },
    ]);
  });
  it("keeps markup in answers as plain text parts", () => {
    const parts = answerParts("<img src=x onerror=alert(1)> [1]", CIT);
    expect(parts[0]).toEqual({ kind: "text", text: "<img src=x onerror=alert(1)> " });
  });
});

describe("highlighting", () => {
  it("merges overlapping spans and clips to the text", () => {
    expect(
      mergeSpans([
        [4, 7],
        [2, 5],
        [9, 9],
      ]),
    ).toEqual([[2, 7]]);
    expect(
      segments("abc", [
        [1, 1],
        [2, 99],
      ]),
    ).toEqual([
      { text: "ab", hit: false },
      { text: "c", hit: true },
    ]);
    expect(segments("", [])).toEqual([{ text: "", hit: false }]);
  });

  it("finds quotes exactly, then tolerant of case, whitespace and punctuation", () => {
    const text = "Staff receive 14 working\ndays of annual leave. Leave must be approved.";
    const exact = findQuote(text, "Leave must be approved");
    expect(exact && text.slice(...exact)).toBe("Leave must be approved");
    const loose = findQuote(text, "receive 14 working days");
    expect(loose && text.slice(...loose)).toBe("receive 14 working\ndays");
    const cased = findQuote(text, "STAFF RECEIVE");
    expect(cased && text.slice(...cased)).toBe("Staff receive");
    expect(findQuote(text, "fifteen days")).toBeNull();
    expect(findQuote(text, "   ")).toBeNull();
    expect(findQuote(text, "...")).toBeNull();
  });

  it("matches BM25 term prefixes and never builds a regex from symbols", () => {
    const text = "Leave policy: leaves and LEAVING; sleeve.";
    expect(termSpans(text, ["leav"]).map(([s, e]) => text.slice(s, e))).toEqual([
      "Leave",
      "leaves",
      "LEAVING",
    ]);
    expect(termSpans(text, ["(", ".*", ""])).toEqual([]);
    expect(termSpans("cuti tahunan", ["tahunan"])).toEqual([[5, 12]]);
  });
});

describe("messages", () => {
  it("explains refusals and errors with a next step", () => {
    expect(reasonText("no_relevant_passages")).toMatch(/matched/);
    expect(reasonText("brand_new")).toBeTruthy();
    expect(friendlyError("llm_unavailable", "x")).toEqual({
      title: "The local language model (Ollama) is not running.",
      hint: "Open the Ollama app, then ask again. Search still works without it.",
    });
    expect(friendlyError("unreachable", "x").hint).toContain("make run");
    expect(friendlyError("llm_model_missing", "Run: ollama pull m").title).toBe(
      "Run: ollama pull m",
    );
    expect(friendlyError("something_new", "Server said so")).toEqual({ title: "Server said so" });
  });

  it("formats numbers", () => {
    expect(humanSize(512)).toBe("512 B");
    expect(humanSize(2048)).toBe("2.0 KB");
    expect(humanSize(5 * 1024 ** 3)).toBe("5.0 GB");
    expect(humanSize(null)).toBe("–");
    expect(fmtMs(335.4)).toBe("335 ms");
    expect(fmtMs(5400)).toBe("5.4 s");
    expect(pct(0.967)).toBe("97%");
    expect(pct(null)).toBe("–");
    expect(truncate("a  b\nc", 10)).toBe("a b c");
    expect(truncate("x".repeat(20), 10)).toBe("x".repeat(9) + "…");
  });
});

describe("API key checks (connect form)", () => {
  const KEY = "eka_" + "Ab-_9".repeat(8) + "xyz"; // 47 characters

  it("accepts a whole key, ignoring whitespace a paste can add", () => {
    expect(KEY).toHaveLength(47);
    expect(keyProblem(KEY)).toBeNull();
    expect(keyProblem(`  ${KEY}\n`)).toBeNull();
    expect(cleanKey(`​${KEY} \r\n`)).toBe(KEY);
    expect(keyProblem("")).toBeNull(); // nothing typed yet: no message
  });

  it("explains what's wrong without echoing the key", () => {
    expect(keyProblem(KEY.slice(0, 16))).toMatch(/47 characters long; this is 16/);
    expect(keyProblem(KEY + KEY)).toMatch(/this is 94/);
    expect(keyProblem('{"id":"b2f1","name":"Demo"}')).toMatch(/start with “eka_”/);
    expect(keyProblem("b2f14a3f-0a0f-4ef9-b3d7-5c553a061a7a")).toMatch(/workspace ID/);
    expect(keyProblem("eka_abc.def")).toMatch(/characters a key never has/);
    for (const bad of [KEY.slice(0, 16), "eka_abc.def"]) {
      expect(keyProblem(bad)).not.toContain(bad.slice(4));
    }
  });
});
