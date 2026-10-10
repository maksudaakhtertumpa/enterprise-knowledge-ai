import { cookies } from "next/headers";
import { redirect } from "next/navigation";

import { AppShell } from "@/components/app-shell";
import { COOKIE_NAME, unseal } from "@/lib/server/session";

export const dynamic = "force-dynamic";

/** Every page in this group needs a session; the key itself never leaves the server. */
export default async function AppLayout({ children }: { children: React.ReactNode }) {
  const session = unseal((await cookies()).get(COOKIE_NAME)?.value);
  if (!session) redirect("/connect");
  return <AppShell workspace={session.workspace}>{children}</AppShell>;
}
