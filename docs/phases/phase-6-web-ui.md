# Phase 6 — Web UI (Next.js + Tailwind CSS + shadcn/ui)

**Budget: $0.** Next.js, Tailwind CSS, shadcn/ui, Radix UI and Playwright are open source and run on your Mac. Nothing is hosted, and Next.js telemetry is turned off in every script (`NEXT_TELEMETRY_DISABLED=1`).

## What was built

```
Browser ──► Next.js (localhost:3000) ───────────────► FastAPI (localhost:8000)
            pages (React, Tailwind, shadcn/ui)         auth, workspace scope, retrieval,
            /api/session  sealed httpOnly cookie        local LLM, citation verification
            /api/eka/*    allowlisted proxy + key ──X-API-Key──►
```

The browser only ever talks to the Next.js server. The Next.js server holds the workspace key in a sealed cookie and adds it to calls to the API (a "backend for frontend").

| Page | What it does |
|---|---|
| **Connect** | Paste a workspace API key. The box checks the paste as you go ("Keys are 47 characters long; this is 16") and can show what you pasted. The API checks the key before anything is stored. Shows backend status: database, vector store, embedding model, reranker, and Ollama (marked optional). |
| **Ask** | Chat-style questions. Answers show **citation chips** `[1]` with the source in a tooltip. Clicking a chip opens that source, with the **verified quote highlighted inside the passage the model saw**. A switch shows the passage **in its surrounding document text**. "How this answer was checked" lists citations verified/proposed, sentences removed, rejected citations with reasons, every passage shown to the model with its relevance score, and any flagged instruction-like text. Refusals explain *why*. While waiting, a timer shows how long the local model has been working. Filter by document, choose 1–10 passages, override retrieval mode and reranking. |
| **Search** | Passages without an answer: dense, hybrid or BM25, reranking on/off, 3–20 results, document filter. Each result shows its score, its rank before reranking, per-retriever ranks for hybrid, and highlighted BM25 terms. **Compare all three** runs the three retrievers side by side. |
| **Documents** | Drag-and-drop or choose several PDF/DOCX/TXT files. Each file gets its own result (uploaded, duplicate, invalid) and the batch continues. The list **refreshes every 2 s while anything is processing** and stops when done. Select a document to see pages, chunks, size and hash, reprocess it, delete it (with a confirmation dialog), and page through its chunks. |
| **History** | Past questions, newest first, with status, latency, citations and model. Click one to replay the full answer with the same citation views. |
| **Analytics** | 7/30/90-day window: questions, answer rate, median and p95 latency, citation acceptance rate, citations per answer, tokens, **cost $0.00**, questions per day (answered vs not enough evidence), why questions went unanswered, models used, documents, searchable chunks and storage. |

| Area | Files |
|---|---|
| Pages and layouts (App Router) | `web/src/app/` — `connect/`, `(app)/ask|search|documents|history|analytics/` |
| Session: seal/unseal the key, cookie options | `web/src/lib/server/session.ts`, `config.ts` |
| Proxy: allowlist, Origin check, forwarding | `web/src/lib/server/proxy.ts`, `web/src/app/api/eka/[...path]/route.ts` |
| Connect / disconnect, backend status | `web/src/app/api/session/route.ts`, `web/src/app/api/health/route.ts` |
| Browser client (only calls this site) | `web/src/lib/api.ts`, `web/src/hooks/use-load.ts` |
| Answer view, chips, sources, checks | `web/src/components/answer-view.tsx`, `highlighted.tsx` |
| Pure helpers: highlighting, labels, messages | `web/src/lib/format.ts` |
| shadcn/ui components (copied in, as shadcn intends) | `web/src/components/ui/` |
| Security headers (CSP etc.) | `web/next.config.ts` |
| Usage analytics for the dashboard | `app/api/routes/analytics.py`, `app/schemas/analytics.py` |
| API + UI together | `scripts/dev.sh` (`make dev`) |

### New API endpoint
| Method | Path | Purpose |
|---|---|---|
| GET | `/api/v1/analytics?days=30` | Question counts, answer rate, refusal reasons, latency p50/p95/mean/max, citation acceptance, tokens (cost always $0), questions per model, per-day counts (zeros included), documents by status, searchable chunks, bytes stored |

## Run it

You need **Node.js 20.9 or newer** (free). On the Mac, install the LTS version from <https://nodejs.org>, or `brew install node@22`. Check with `node --version`.

```bash
cd ~/Projects/enterprise-knowledge-ai
source .venv/bin/activate
make web-install          # once: exact versions from web/package-lock.json
make dev                  # API on :8000 + web UI on :3000; Ctrl+C stops both
```
Open <http://localhost:3000>, paste the key from `python -m app.cli create-workspace "Demo"`, and upload documents. The 11 policy files in `eval/datasets/teratai/corpus/` make a good demo set.

| Command | What it does |
|---|---|
| `make dev` | API + web UI (development mode, hot reload) |
| `./scripts/dev.sh --offline` | Same with offline stand-ins: no model download, no Ollama. Uses its own `data-offline/` folder |
| `make run` / `make web` | API only / web UI only |
| `make web-check` | Prettier, ESLint, TypeScript, unit tests, production build |
| `make e2e` | Browser tests against a real API (first time: `cd web && npx playwright install chromium`, ~150 MB, free) |

Settings live in `web/.env.local` (git-ignored). `make web-install` and `make dev` create it once with a random session secret, so restarting the web server doesn't log you out. The web server listens on `127.0.0.1` only, so other devices on your Wi-Fi can't open it.

## Engineering decisions

**1. The browser never sees the API key (backend for frontend).** The key is checked against the API once. It is then sealed with AES-256-GCM (authenticated encryption) into a cookie that is `httpOnly` (page scripts can't read it), `SameSite=Strict` (not sent on cross-site requests) and expires after 8 hours. The Next.js server opens it and adds `X-API-Key` when forwarding. A cross-site scripting bug therefore couldn't steal the key, and nothing is kept in `localStorage`. An end-to-end test checks the cookie flags and that the key appears nowhere in `document.cookie`, browser storage or the page.

**2. The proxy is an allowlist, not an open pipe.** Only the 13 method + path pairs the UI uses are forwarded. IDs must be UUIDs, and path segments can't contain `..` or encoded slashes. State-changing requests must carry this site's own `Origin` (a second layer behind SameSite). Only `content-type` and `x-request-id` come back from the API; never its cookies or server headers. If the API rejects the key (revoked), the browser session is ended too. Uploads are streamed through, not buffered.

**3. No browser access to the API at all.** Since the browser never calls FastAPI, CORS is now **off by default** (`EKA_CORS_ORIGINS=[]`), and a test checks that no origin gets a CORS grant.

**4. Untrusted text is only ever React text.** Document text, quotes, filenames and model answers can contain HTML or Markdown, from an uploaded document or a prompt-injected model. React escapes text, so the UI never renders Markdown and never uses `dangerouslySetInnerHTML` or `innerHTML` (ESLint rules make both an error). Highlighting splits text into segments and wraps some in `<mark>`, never by building HTML. A Content Security Policy is a third layer: only this site's own scripts, and no images from other hosts, so even a smuggled tracking pixel couldn't load. Unit and browser tests plant `<img onerror>`, `<script>` and a remote image in a document, have it repeated in an answer, and check none of them run or render.

**5. shadcn/ui, the way it's meant to be used.** shadcn/ui is a set of component source files you copy into your project, built on Radix UI (accessible primitives) and Tailwind. The 20 components used are in `web/src/components/ui/`. The shadcn registry website was unreachable from the build sandbox, so they were written from the published new-york style rather than with `npx shadcn add`. `components.json` is included, so you can add more with the CLI on your Mac. Colours come from theme variables and follow the system light/dark setting.

**6. Server-side gate, client-side pages.** The `(app)` layout is a server component: it reads the sealed cookie and redirects to `/connect` before any page renders. The pages are client components that fetch through the proxy, with loading skeletons, a per-request elapsed timer for answers, and errors mapped to a fix (Ollama not running → "Open the Ollama app; search still works"; model missing → the `ollama pull` command; key rejected → how to create one; API down → `make run`). Request IDs are shown so the matching log line can be found.

**7. Analytics are computed by the API.** `/analytics` reads only metric columns, never question text, so it works with `EKA_ASK_HISTORY=false`. The page says these numbers describe usage, not accuracy, and points to the benchmark for quality. The chart is drawn with plain elements, so no chart library is needed.

**8. Honest UX.** Each question is answered on its own (the API has no conversation memory), and the Ask page says so. Approximate (fuzzy) quotes are labelled. If a quote can't be located in the passage, nothing is highlighted rather than the wrong span. Refusals look like a normal outcome with a reason.

**9. Pinned, audited dependencies.** Exact versions in `package.json` plus `package-lock.json`, installed with `npm ci`. Next.js 16.3.8 is used because the advisory database lists high-severity fixes over 16.3.7 and earlier. `npm audit --omit=dev` (what ships) reports 0 vulnerabilities and runs in CI. The one remaining advisory is in a development-only dependency of the ESLint plugin, with no fixed version yet.

**10. Lessons from the first run on the Mac.** The browser tests passed, but connecting by hand failed several times. The server log showed why: the connect request never reached the API, so what was in the key box wasn't a whole key. Three causes combined. The key box was a password field, which shows only dots. Double-clicking a key in Terminal selects only part of it, because keys contain `-`. Copying commands from the chat replaced the key on the clipboard. On top of that, the generic "not accepted" message couldn't tell a bad paste from a bad key. The fixes:
- A text box masked with CSS, with a **Show** button. Being a text box, browsers' password managers don't autofill it or offer to save it.
- A **live check** ("Keys are 47 characters long; this is 16"). It never repeats the key.
- **Separate errors** for "not a complete key" (never sent to the API) and "the API doesn't recognise this key".
- Whitespace from copying is stripped.
- A **fixed session secret** in `web/.env.local`, so a restart no longer logs you out.
- The dev server listens on **127.0.0.1** only.
- Next.js's development button no longer covers "Disconnect".

A browser test now covers a partial paste.

**11. A real bug the browser tests caught.** Without `EKA_WEB_SESSION_SECRET`, the server makes a random sealing key. Next.js bundles route handlers and pages separately, so a module-level variable existed twice in one process. Connecting "worked", but the page couldn't open the cookie and bounced back to Connect. Unit tests passed; the end-to-end test failed. The key now lives on a global symbol shared by both bundles.

## Testing
| Suite | Covers |
|---|---|
| `web/src/lib/format.test.ts` | Key checks (partial, doubled, workspace ID or JSON pasted instead, invalid characters, never echoed); citation markers → chips only for verified citations; markup in answers stays text; quote finding (exact, then case/whitespace/punctuation tolerant, never a wrong span); BM25 term highlighting without regex injection; span merging; labels, reasons, error hints, number formats |
| `web/src/lib/server/session.test.ts` | Seal/unseal round trip; key not visible in the token; fresh IV; tampered, truncated, oversized, garbage and expired tokens rejected; cookie flags; key shape |
| `web/src/lib/server/routes.test.ts` | Connect checks the key with the API and sets a sealed httpOnly SameSite=Strict cookie; rejected keys get no cookie; malformed keys never reach the API and get their own error code; whitespace around a key is stripped; cross-site and Origin-less requests refused; API down → 502; proxy adds the key and strips upstream cookies/headers; no session → 401; allowlist (11 allowed, 8 refused incl. `..` and encoded slashes); revoked key ends the session |
| `web/src/components/answer-view.test.tsx` | Chips, meta line ($0.00), flagged passages; chip opens its source with the quote in `<mark>`; refusal reason; **XSS payloads render as text**; rejected citations listed |
| `web/e2e/app.spec.ts` (Playwright, Chromium, 11 tests) | Real API + production build: status panel; unknown key; **partial paste explained before sending**, show/hide, whitespace-wrapped key accepted; **key out of reach of page scripts**; proxy allowlist and Origin check; upload 4 files with live status, duplicate and fake PDF; inspect chunks; cited answer → chip → highlighted quote → document context → checks; refusal; **planted XSS document in a second workspace** (and isolation from the first); search highlighting and three-way comparison; history replay; analytics numbers; delete confirmation; disconnect |
| `tests/api/test_analytics_api.py` (12) | Empty workspace, rates and percentiles, refusal reasons, window and per-day buckets, document stats, real `/ask` calls, history off, isolation, validation, auth |
| `tests/test_errors_and_middleware.py` | No CORS grant for any browser origin by default |

Also checked by looking at screenshots: every page, dark mode and a phone-width window. The question box is pinned to the bottom only on wider screens, because on a phone it covered the answer.

## Likely errors and fixes
| Symptom | Cause | Fix |
|---|---|---|
| `npm: command not found` | Node.js not installed | Install the LTS version from nodejs.org (or `brew install node@22`), open a new Terminal |
| `npm ci` complains about the Node version | Node older than 20.9 | Install the current LTS |
| Web page: "Cannot reach the API server" | API not started, or still loading models | `make dev` (or `make run` in another terminal); wait for "Application startup complete" |
| "That doesn't look like a complete API key" | Part of the key, or something else, was pasted | Click **Show** to see what's in the box. Copy the whole key (one line starting with `eka_`, 47 characters). Triple-click the line in Terminal; double-click stops at `-` |
| "The API doesn't recognise this key" | Key revoked, or made for a different data folder | Create a new key in the project folder: `python -m app.cli create-workspace "Name"` |
| Sent back to Connect after restarting | `web/.env.local` missing its secret | Run `./scripts/web-setup.sh` (also run by `make dev`) |
| Connect works but you're bounced back over HTTPS-less LAN access | `EKA_WEB_COOKIE_SECURE=true` without HTTPS | Keep it `false` for `http://localhost` |
| "The request was blocked (it did not come from this app)" | Opened the site through a different host name than it's served on | Use <http://localhost:3000> |
| `Port 3000 is already in use` | An earlier web server still running | Close that terminal, or `lsof -ti:3000 \| xargs kill` |
| `make e2e`: "Executable doesn't exist" | Playwright's Chromium not installed | `cd web && npx playwright install chromium` |
| Ollama not running | Answers need it | Open Ollama; documents and search keep working |
| Offline mode answers questions it shouldn't | The offline stand-in isn't semantic | Expected; use real models + Ollama |

## Acceptance criteria
Verified here (Linux sandbox, offline stand-in models):
- [x] Next.js + Tailwind CSS + shadcn/ui app replaces the Streamlit UI; the UI only reaches the API through its own server
- [x] Upload and management: multi-file drag-and-drop, per-file results, live status, details, chunks, reprocess, delete with confirmation
- [x] Ask: citation chips, verified quote highlighted in its passage, surrounding-document view, how-it-was-checked panel, refusals with reasons, elapsed timer
- [x] Search with scores, pre-rerank and per-retriever ranks, term highlighting, three-way comparison
- [x] History with replay; analytics backed by a tested `/analytics` endpoint
- [x] The key never reaches page scripts; proxy allowlist and Origin check; CSP; untrusted text can't render as HTML (unit, route and browser tests)
- [x] Python: 396 tests, Ruff clean, smoke test 37 checks. Web: Prettier, ESLint (0 warnings), TypeScript, 52 unit tests, production build, **11 browser tests**
- [x] Looked at in a real browser: all pages, dark mode, phone width

Verified on the MacBook Air M4:
- [x] `./scripts/verify.sh --quick` passes, including the web UI checks
- [x] `make e2e` passes (first version: 10 of 10 browser tests)
- [x] Connected in Chrome with real models and Ollama (first try failed on partial pastes; see decision 10, fixed and re-checked)
- [x] Uploaded the 11 Teratai policy files: 11 ready
- [x] Answerable question, "How many days of annual leave do staff with less than two years of service get?": "Staff with less than two years of service receive 14 working days per calendar year [1]". The chip opens leave-policy.txt › Annual Leave with the verified quote highlighted, and the surrounding-text view works. `qwen3.5:4b`, 1/1 citations verified, 1,061 + 61 tokens, 9.6 s for the first question (including loading the model), $0.00
- [x] Unanswerable question, "Can I bring my dog to the office?": "Not enough evidence to answer", with no invented answer

**Phase 6: complete.**
- [ ] `make dev`: upload the Teratai corpus, ask an answerable and an unanswerable question, open a source and its surrounding text

## Interview talking points
- "The browser never holds the API key. Next.js acts as a backend for frontend: the key is sealed with AES-GCM into an httpOnly, SameSite=Strict cookie, and the server adds it when proxying. Even an XSS bug couldn't steal it, and a Playwright test checks it appears nowhere in the page, cookies or storage."
- "The proxy is an allowlist of 13 method-and-path pairs with UUID checks, plus an Origin check on writes. Because the browser never calls the API directly, I turned CORS off on the backend."
- "Model output and document text are untrusted, so they're only rendered as React text, with ESLint banning `dangerouslySetInnerHTML` and a CSP that blocks remote images and scripts. Tests plant XSS payloads and a tracking pixel in a document and check nothing runs."
- "Clicking a citation shows the exact verified quote highlighted in the passage the model saw, then in the original document. Anyone can check every claim in one click."
- "End-to-end tests caught a bug unit tests couldn't: Next.js bundles route handlers and pages separately, so my per-process session key existed twice. Connecting looked fine but bounced back. A global symbol fixed it."
- "Everything is pinned and audited: exact versions, `npm ci`, and `npm audit` in CI. I picked Next.js 16.3.8 specifically because of high-severity advisories in earlier 16.x releases."
- "All my automated tests passed, but the first real user couldn't connect. The server log showed the request never reached the API, so it was a paste problem, not a key problem. I made the key box check the paste as you type, split the error into 'not a complete key' and 'not recognised', and stopped password managers from interfering. Logs that can tell those two cases apart saved a lot of guessing."
