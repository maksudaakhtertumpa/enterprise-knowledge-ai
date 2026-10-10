import { readFileSync } from "node:fs";
import path from "node:path";

import { expect, test, type Page } from "@playwright/test";

const CORPUS = path.resolve(__dirname, "..", "..", "eval", "datasets", "teratai", "corpus");
const FILES = ["leave-policy.txt", "it-security.txt", "facilities.txt", "benefits.txt"].map((f) =>
  path.join(CORPUS, f),
);
const KEY = () => process.env.EKA_E2E_KEY!;
const KEY2 = () => process.env.EKA_E2E_KEY2!;

test.describe.configure({ mode: "serial" });

async function connect(page: Page, key: string) {
  await page.goto("/connect");
  await page.getByLabel("Workspace API key").fill(key);
  await page.getByRole("button", { name: "Connect" }).click();
  await expect(page).toHaveURL(/\/ask$/);
}

async function ask(page: Page, question: string) {
  await page.getByLabel("Question").fill(question);
  await page.getByRole("button", { name: "Send" }).click();
  const turn = page.getByTestId("turn").last();
  await expect(turn.getByTestId("thinking")).toHaveCount(0, { timeout: 30_000 });
  return turn;
}

test("connect page shows backend status and rejects a wrong key", async ({ page }) => {
  await page.goto("/");
  await expect(page).toHaveURL(/\/connect$/); // no session -> connect
  await expect(page.getByTestId("readiness")).toHaveText("The API is ready.");
  await expect(page.getByText("Local LLM (Ollama)")).toBeVisible();

  // Well-formed but unknown: the API itself rejects it.
  await page.getByLabel("Workspace API key").fill("eka_" + "x".repeat(43));
  await page.getByRole("button", { name: "Connect" }).click();
  await expect(page.getByText("The API doesn't recognise this key.")).toBeVisible();
  await expect(page).toHaveURL(/\/connect$/);
});

test("the API key stays out of reach of page scripts", async ({ page, context }) => {
  await connect(page, KEY());
  await expect(page.getByTestId("workspace-name")).toContainText("Teratai HR");

  const cookie = (await context.cookies()).find((c) => c.name === "eka_session");
  expect(cookie?.httpOnly).toBe(true);
  expect(cookie?.sameSite).toBe("Strict");
  expect(cookie?.value).not.toContain(KEY()); // sealed, not the key itself

  const visible = await page.evaluate(() => ({
    cookie: document.cookie,
    local: JSON.stringify({ ...localStorage }),
    session: JSON.stringify({ ...sessionStorage }),
    html: document.documentElement.outerHTML,
  }));
  for (const value of Object.values(visible)) expect(value).not.toContain(KEY());
});

test("the proxy only forwards allowed calls from this site", async ({ page, request }) => {
  // No session cookie in this separate request context.
  const anon = await request.get("/api/eka/workspace");
  expect(anon.status()).toBe(401);

  await connect(page, KEY());
  const ok = await page.request.get("/api/eka/workspace");
  expect(ok.status()).toBe(200);
  // Not on the allowlist, or path tricks -> 404 without reaching the API.
  for (const p of [
    "/api/eka/admin",
    "/api/eka/documents/..%2F..%2Fhealth",
    "/api/eka/health/ready",
  ]) {
    expect((await page.request.get(p)).status()).toBe(404);
  }
  // A state-changing request from another origin is refused.
  const forged = await page.request.post("/api/eka/search", {
    headers: { Origin: "http://evil.example" },
    data: { query: "leave" },
  });
  expect(forged.status()).toBe(403);
});

test("upload documents, watch them process, inspect and reject bad files", async ({ page }) => {
  await connect(page, KEY());
  await page.getByRole("link", { name: "Documents" }).first().click();
  await expect(page.getByText("No documents yet.")).toBeVisible();

  await page.getByTestId("file-input").setInputFiles(FILES);
  const results = page.getByTestId("upload-results");
  for (const f of FILES) {
    await expect(results).toContainText(`${path.basename(f)}: uploaded`);
  }
  await expect(page.getByTestId("doc-counts")).toContainText("4 document(s): 4 ready", {
    timeout: 30_000,
  });

  // Duplicate and fake files get per-file messages; the batch continues.
  await page.getByTestId("file-input").setInputFiles([
    { name: "copy.txt", mimeType: "text/plain", buffer: readFileSync(FILES[0]!) },
    { name: "fake.pdf", mimeType: "application/pdf", buffer: Buffer.from("not a pdf") },
  ]);
  await expect(results).toContainText("already been uploaded");
  await expect(results).toContainText("The file is not a valid PDF.");

  await page.getByRole("button", { name: "leave-policy.txt" }).click();
  const details = page.getByTestId("document-details");
  await expect(details).toContainText("leave-policy.txt");
  await expect(details.getByTestId("chunk").first()).toBeVisible();
});

test("ask: cited answer, source with highlighted quote and document context", async ({ page }) => {
  await connect(page, KEY());
  const turn = await ask(
    page,
    "How many days of annual leave do staff with less than two years get?",
  );
  const answer = turn.getByTestId("answer");
  await expect(answer.getByTestId("answer-text")).toBeVisible();
  const chip = answer.getByTestId("citation-chip").first();
  await expect(chip).toBeVisible();
  await expect(answer.getByTestId("answer-meta")).toContainText("$0.00");

  await chip.click(); // opens the matching source card
  const card = answer.getByTestId("source-card").first();
  await expect(card.getByTestId("source-passage").locator("mark")).toHaveCount(1);
  await card.getByText("Show surrounding text in the document").click();
  await expect(card.getByTestId("source-context").locator("mark")).toHaveCount(1);

  await answer.getByText("How this answer was checked").click();
  await expect(answer.getByText("Passages the model was shown")).toBeVisible();
});

test("ask: a question the documents can't answer is refused with a reason", async ({ page }) => {
  await connect(page, KEY());
  const turn = await ask(page, "Xylophone quasar zeppelin?");
  await expect(turn.getByTestId("refusal")).toContainText("Not enough evidence to answer.");
});

test("untrusted document text never renders as HTML", async ({ page }) => {
  // Separate workspace: a planted document repeats an XSS payload and a tracking pixel.
  await connect(page, KEY2());
  await page.getByRole("link", { name: "Documents" }).first().click();
  const evil =
    'The parking fee is five ringgit per day <img src=x onerror="window.__pwned=1"> ' +
    "![pixel](https://attacker.example/leak.png) <script>window.__pwned=2</script>.\n";
  await page.getByTestId("file-input").setInputFiles({
    name: "parking<b>.txt",
    mimeType: "text/plain",
    buffer: Buffer.from(evil),
  });
  await expect(page.getByTestId("doc-counts")).toContainText("1 ready", { timeout: 30_000 });

  await page.getByRole("link", { name: "Ask" }).first().click();
  const turn = await ask(page, "What is the parking fee per day?");
  const answer = turn.getByTestId("answer");
  await expect(answer.getByTestId("answer-text")).toContainText("<img src=x onerror=");
  await expect(page.locator("main img")).toHaveCount(0);
  await expect(page.locator("main script")).toHaveCount(0);
  expect(
    await page.evaluate(() => (window as unknown as { __pwned?: number }).__pwned),
  ).toBeUndefined();
  // Workspace isolation in the UI: this workspace doesn't see the other one's files.
  await page.getByRole("link", { name: "Documents" }).first().click();
  await expect(page.getByText("leave-policy.txt")).toHaveCount(0);
});

test("search: highlighted BM25 terms and a three-way comparison", async ({ page }) => {
  await connect(page, KEY());
  await page.getByRole("link", { name: "Search" }).first().click();
  await page.getByLabel("Query").fill("annual leave carry forward");
  await page.getByRole("radio", { name: "BM25 (keyword)" }).click();
  await page.getByRole("button", { name: "Search" }).click();
  await expect(page.getByTestId("search-result").first().locator("mark").first()).toBeVisible();

  await page.getByLabel("Compare all three").click();
  await page.getByRole("button", { name: "Search" }).click();
  for (const name of ["Dense (semantic)", "Hybrid (BM25 + dense, RRF)", "BM25 (keyword)"]) {
    await expect(page.getByRole("region", { name })).toBeVisible();
  }
});

test("history replays answers; analytics counts them", async ({ page }) => {
  await connect(page, KEY());
  await page.getByRole("link", { name: "History" }).first().click();
  const items = page.getByTestId("history-item");
  await expect(items).toHaveCount(2);
  await items.filter({ hasText: "annual leave" }).getByRole("button").click();
  await expect(
    page.getByTestId("history-entry").getByTestId("citation-chip").first(),
  ).toBeVisible();

  await page.getByRole("link", { name: "Analytics" }).first().click();
  await expect(page.getByTestId("metric-Questions")).toHaveText("2");
  await expect(page.getByTestId("metric-Answered")).toHaveText("50%");
  await expect(page.getByTestId("metric-Cost")).toHaveText("$0.00");
  await expect(page.getByTestId("metric-Ready")).toHaveText("4");
});

test("delete asks for confirmation; disconnect ends the session", async ({ page }) => {
  await connect(page, KEY());
  await page.getByRole("link", { name: "Documents" }).first().click();
  await page.getByRole("button", { name: "benefits.txt" }).click();
  await page.getByTestId("document-details").getByRole("button", { name: "Delete" }).click();
  await expect(page.getByText("This cannot be undone.")).toBeVisible();
  await page.getByRole("alertdialog").getByRole("button", { name: "Delete" }).click();
  await expect(page.getByTestId("doc-counts")).toContainText("3 document(s)");

  await page.getByRole("button", { name: "Disconnect" }).first().click();
  await expect(page).toHaveURL(/\/connect$/);
  await page.goto("/ask");
  await expect(page).toHaveURL(/\/connect$/);
});

test("the key box explains a bad paste before anything is sent", async ({ page }) => {
  await page.goto("/connect");
  const box = page.getByLabel("Workspace API key");
  await box.fill(KEY().slice(0, 16));
  await expect(page.getByTestId("key-check")).toContainText("47 characters long; this is 16");
  await page.getByRole("button", { name: "Connect" }).click();
  await expect(page.getByText("That doesn't look like a complete API key.")).toBeVisible();
  await expect(page).toHaveURL(/\/connect$/);

  // Show/hide, and a key surrounded by whitespace from a sloppy copy still works.
  await expect(box).toHaveCSS("-webkit-text-security", "disc");
  await page.getByRole("button", { name: "Show key" }).click();
  await expect(box).not.toHaveCSS("-webkit-text-security", "disc");
  await box.fill(`  ${KEY()}\n`);
  await expect(page.getByTestId("key-check")).toContainText("Looks like a complete key");
  await page.getByRole("button", { name: "Connect" }).click();
  await expect(page).toHaveURL(/\/ask$/);
});
