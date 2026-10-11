// End-to-end tests: the real FastAPI backend (offline stand-in models, temporary
// data folder) + the production build of this web app, driven in Chromium.
//
//   npm run e2e        (first time: npx playwright install chromium)
import { mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";

import { defineConfig, devices } from "@playwright/test";

const API_PORT = 8765;
const WEB_PORT = 3100;
const python =
  process.env.EKA_PYTHON ??
  path.resolve(
    __dirname,
    "..",
    process.platform === "win32" ? ".venv\\Scripts\\python.exe" : ".venv/bin/python",
  );
// Set once in the main process; workers inherit it.
process.env.EKA_E2E_DATA ??= mkdtempSync(path.join(tmpdir(), "eka-e2e-"));
process.env.EKA_E2E_PYTHON = python;

const backendEnv = {
  EKA_DATA_DIR: process.env.EKA_E2E_DATA,
  EKA_EMBEDDING_BACKEND: "hashing",
  EKA_RERANKER_BACKEND: "hashing",
  EKA_LLM_BACKEND: "extractive",
  EKA_ENTAILMENT_BACKEND: "lexical",
  EKA_LOG_LEVEL: "WARNING",
};

export default defineConfig({
  testDir: "./e2e",
  globalSetup: "./e2e/global-setup.ts",
  fullyParallel: false,
  workers: 1,
  retries: process.env.CI ? 1 : 0,
  timeout: 60_000,
  reporter: process.env.CI ? [["list"], ["html", { open: "never" }]] : "list",
  use: {
    baseURL: `http://localhost:${WEB_PORT}`,
    trace: "retain-on-failure",
    screenshot: "only-on-failure",
  },
  projects: [
    {
      name: "chromium",
      use: {
        ...devices["Desktop Chrome"],
        // Use an already-installed Chromium if given (e.g. a sandbox without downloads).
        launchOptions: process.env.PW_CHROMIUM_PATH
          ? { executablePath: process.env.PW_CHROMIUM_PATH }
          : {},
      },
    },
  ],
  webServer: [
    {
      command: `"${python}" -m uvicorn app.main:app --port ${API_PORT}`,
      cwd: "..",
      env: backendEnv,
      url: `http://127.0.0.1:${API_PORT}/health`,
      reuseExistingServer: false,
      timeout: 120_000,
    },
    {
      command: `npx next build && npx next start --hostname 127.0.0.1 --port ${WEB_PORT}`,
      env: { EKA_API_URL: `http://127.0.0.1:${API_PORT}`, NEXT_TELEMETRY_DISABLED: "1" },
      url: `http://localhost:${WEB_PORT}/connect`,
      reuseExistingServer: false,
      timeout: 300_000,
    },
  ],
});
