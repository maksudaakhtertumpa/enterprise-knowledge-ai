// README screenshots, taken from the running app (scripts/screenshots.sh starts it).
//
//   EKA_SHOT_URL=http://localhost:3010 EKA_SHOT_KEY=eka_... node scripts/screenshots.mjs
//
// Uploads the demo policy documents, asks real questions, and saves JPEGs to
// docs/screenshots/. Everything shown is real output of the app as configured;
// nothing is staged or edited afterwards.
import { mkdirSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

import { chromium } from "@playwright/test";

const here = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(here, "..", "..");
const OUT = path.join(ROOT, "docs", "screenshots");
const CORPUS = path.join(ROOT, "eval", "datasets", "teratai", "corpus");
const FILES = [
  "leave-policy.txt",
  "benefits.txt",
  "it-security.txt",
  "travel-and-expense.txt",
  "remote-work.txt",
  "code-of-conduct.txt",
  "data-protection.txt",
  "facilities.txt",
  "onboarding.txt",
  "performance.txt",
  "procurement.txt",
].map((f) => path.join(CORPUS, f));
const BASE = process.env.EKA_SHOT_URL ?? "http://localhost:3010";
const KEY = process.env.EKA_SHOT_KEY;
if (!KEY) throw new Error("Set EKA_SHOT_KEY to a workspace API key");
const ANSWER_TIMEOUT = 180_000; // the first local-LLM answer also loads the model

mkdirSync(OUT, { recursive: true });
const launchOptions = process.env.PW_CHROMIUM_PATH
  ? { executablePath: process.env.PW_CHROMIUM_PATH }
  : {};
const browser = await chromium.launch(launchOptions);

async function newPage(options = {}) {
  const context = await browser.newContext({
    viewport: { width: 1280, height: 860 },
    deviceScaleFactor: 2,
    colorScheme: "light",
    ...options,
  });
  context.setDefaultTimeout(60_000); // fail loudly instead of hanging
  const page = await context.newPage();
  await page.goto(`${BASE}/connect`);
  await page.getByLabel("Workspace API key").fill(KEY);
  await page.getByRole("button", { name: "Connect" }).click();
  await page.waitForURL(/\/ask$/);
  return page;
}

async function shot(page, name, options = {}) {
  await page.waitForTimeout(400); // let transitions settle
  const file = path.join(OUT, `${name}.jpg`);
  await page.screenshot({ path: file, type: "jpeg", quality: 82, ...options });
  console.log(`saved docs/screenshots/${name}.jpg`);
}

async function ask(page, question) {
  await page.getByLabel("Question").fill(question);
  await page.getByLabel("Send").click();
  const turn = page.getByTestId("turn").last();
  await turn.getByTestId("thinking").waitFor({ state: "detached", timeout: ANSWER_TIMEOUT });
  return turn;
}

// ------------------------------------------------------------- documents
const page = await newPage();
await page.getByRole("link", { name: "Documents" }).first().click();
await page.getByTestId("file-input").setInputFiles(FILES);
await page
  .getByTestId("doc-counts")
  .filter({ hasText: `${FILES.length} ready` })
  .waitFor({ timeout: 300_000 });
// Reopen the page so the list shows from the top, without the upload log.
await page.getByRole("link", { name: "Ask" }).first().click();
await page.getByRole("link", { name: "Documents" }).first().click();
await page
  .getByTestId("doc-counts")
  .filter({ hasText: `${FILES.length} ready` })
  .waitFor();
await shot(page, "documents");

// ------------------------------------------------------------------- ask
await page.getByRole("link", { name: "Ask" }).first().click();
let turn = await ask(page, "How many days of annual leave do staff with less than two years get?");
const answer = turn.getByTestId("answer");
await answer.getByTestId("citation-chip").first().click();
await answer.getByTestId("source-card").first().locator("mark").first().waitFor();
await shot(page, "ask");

await answer.getByText("How this answer was checked").click();
await answer.getByTestId("entailment").scrollIntoViewIfNeeded();
await shot(page, "checks");

await page.getByRole("button", { name: "Clear" }).click(); // the refusal on its own
turn = await ask(page, "Can I bring my dog to the office?");
// Real models refuse this; offline stand-ins may not, so accept either.
await turn.getByTestId("refusal").or(turn.getByTestId("answer")).first().waitFor();
await shot(page, "refusal");

// ---------------------------------------------------------------- search
await page.getByRole("link", { name: "Search" }).first().click();
await page.getByLabel("Query").fill("Komputer riba saya hilang");
await page.getByLabel("Compare all three").click();
await page.getByRole("button", { name: "Search" }).click();
await page.getByRole("region", { name: "BM25 (keyword)" }).waitFor();
await page.getByTestId("search-result").first().waitFor();
await shot(page, "search");

// ------------------------------------------------------------- analytics
await page.getByRole("link", { name: "Analytics" }).first().click();
await page.getByTestId("metric-Questions").waitFor();
await shot(page, "analytics");

// ------------------------------------------------------- phone, dark mode
const phone = await newPage({
  viewport: { width: 390, height: 844 },
  deviceScaleFactor: 2,
  isMobile: true,
  hasTouch: true,
  colorScheme: "dark",
});
turn = await ask(phone, "Berapa hari cuti tahunan saya?");
await turn.getByTestId("answer").or(turn.getByTestId("refusal")).first().waitFor();
await shot(phone, "ask-mobile-dark");

await browser.close();
