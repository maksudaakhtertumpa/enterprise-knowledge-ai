import type { Metadata } from "next";

import { AskView } from "@/components/views/ask-view";

export const metadata: Metadata = { title: "Ask · Enterprise Knowledge AI" };

export default function AskPage() {
  return <AskView />;
}
