import type { Metadata } from "next";

import { DocumentsView } from "@/components/views/documents-view";

export const metadata: Metadata = { title: "Documents · Enterprise Knowledge AI" };

export default function DocumentsPage() {
  return <DocumentsView />;
}
