// Create two workspaces with the project's own admin CLI; keys go to the tests via env.
import { execFileSync } from "node:child_process";
import path from "node:path";

function createWorkspace(name: string): string {
  const out = execFileSync(
    process.env.EKA_E2E_PYTHON!,
    ["-m", "app.cli", "create-workspace", name],
    {
      cwd: path.resolve(__dirname, "..", ".."),
      env: { ...process.env, EKA_DATA_DIR: process.env.EKA_E2E_DATA },
      encoding: "utf8",
    },
  );
  const line = out.split("\n").find((l) => l.startsWith("API key"));
  if (!line) throw new Error(`No API key in CLI output:\n${out}`);
  return line.trim().split(/\s+/).pop()!;
}

export default function globalSetup() {
  process.env.EKA_E2E_KEY = createWorkspace("Teratai HR");
  process.env.EKA_E2E_KEY2 = createWorkspace("Second Team");
}
