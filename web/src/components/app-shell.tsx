"use client";

import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import {
  BarChart3,
  FileText,
  History,
  LogOut,
  MessagesSquare,
  Search,
  type LucideIcon,
} from "lucide-react";

import { Button } from "@/components/ui/button";
import { api } from "@/lib/api";
import type { Workspace } from "@/lib/types";
import { cn } from "@/lib/utils";

const NAV: { href: string; label: string; icon: LucideIcon }[] = [
  { href: "/ask", label: "Ask", icon: MessagesSquare },
  { href: "/search", label: "Search", icon: Search },
  { href: "/documents", label: "Documents", icon: FileText },
  { href: "/history", label: "History", icon: History },
  { href: "/analytics", label: "Analytics", icon: BarChart3 },
];

export function AppShell({
  workspace,
  children,
}: {
  workspace: Workspace;
  children: React.ReactNode;
}) {
  const pathname = usePathname();
  const router = useRouter();

  async function disconnect() {
    try {
      await api.disconnect();
    } finally {
      router.replace("/connect");
      router.refresh();
    }
  }

  const links = NAV.map(({ href, label, icon: Icon }) => {
    const active = pathname === href || pathname.startsWith(`${href}/`);
    return (
      <Link
        key={href}
        href={href}
        aria-current={active ? "page" : undefined}
        className={cn(
          "flex items-center gap-2 rounded-md px-3 py-2 text-sm whitespace-nowrap transition-colors",
          active
            ? "bg-accent font-medium text-accent-foreground"
            : "text-muted-foreground hover:bg-accent/60 hover:text-foreground",
        )}
      >
        <Icon className="size-4" />
        {label}
      </Link>
    );
  });

  return (
    <div className="flex min-h-dvh flex-col md:flex-row">
      <aside className="border-b bg-sidebar md:sticky md:top-0 md:flex md:h-dvh md:w-60 md:shrink-0 md:flex-col md:border-r md:border-b-0">
        <div className="flex items-center justify-between gap-2 px-4 py-3 md:block md:py-5">
          <Link href="/ask" className="font-semibold tracking-tight">
            Enterprise Knowledge AI
          </Link>
          <p
            className="truncate text-xs text-muted-foreground md:mt-1"
            data-testid="workspace-name"
          >
            Workspace: <span className="font-medium text-foreground">{workspace.name}</span>
          </p>
        </div>
        <nav className="flex gap-1 overflow-x-auto px-2 pb-2 md:flex-1 md:flex-col md:overflow-visible md:pb-0">
          {links}
        </nav>
        <div className="hidden px-2 py-4 md:block">
          <Button variant="ghost" size="sm" className="w-full justify-start" onClick={disconnect}>
            <LogOut /> Disconnect
          </Button>
        </div>
      </aside>
      <main className="min-w-0 flex-1 px-4 py-6 md:px-8 md:py-8">
        <div className="mx-auto max-w-5xl">{children}</div>
        <div className="mt-10 md:hidden">
          <Button variant="outline" size="sm" onClick={disconnect}>
            <LogOut /> Disconnect
          </Button>
        </div>
      </main>
    </div>
  );
}
