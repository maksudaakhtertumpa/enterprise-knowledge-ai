import { cookies } from "next/headers";
import { redirect } from "next/navigation";

import { ConnectForm } from "@/components/connect-form";
import { ReadinessPanel } from "@/components/readiness-panel";
import { COOKIE_NAME, unseal } from "@/lib/server/session";

export const dynamic = "force-dynamic";

export default async function ConnectPage({
  searchParams,
}: {
  searchParams: Promise<{ expired?: string }>;
}) {
  const session = unseal((await cookies()).get(COOKIE_NAME)?.value);
  if (session) redirect("/ask");
  const { expired } = await searchParams;
  return (
    <main className="mx-auto max-w-2xl space-y-6 px-4 py-10">
      <div className="space-y-1">
        <h1 className="text-3xl font-semibold tracking-tight">Enterprise Knowledge AI</h1>
        <p className="text-muted-foreground">
          Ask questions about your documents and get answers with verified citations. Runs locally,
          $0.
        </p>
      </div>
      <ConnectForm expired={expired === "1"} />
      <ReadinessPanel />
    </main>
  );
}
