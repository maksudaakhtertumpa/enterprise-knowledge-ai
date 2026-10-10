import type { Metadata } from "next";

import { TooltipProvider } from "@/components/ui/tooltip";

import "./globals.css";

export const metadata: Metadata = {
  title: "Enterprise Knowledge AI",
  description: "Ask questions about your documents and get answers with verified citations.",
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    // Browser extensions (e.g. writing assistants) add attributes to <html> before React
    // loads; this silences that one harmless mismatch on this element only.
    <html lang="en" suppressHydrationWarning>
      <body>
        <TooltipProvider delayDuration={150}>{children}</TooltipProvider>
      </body>
    </html>
  );
}
