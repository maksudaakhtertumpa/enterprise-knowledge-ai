import type { Metadata } from "next";

import { SearchView } from "@/components/views/search-view";

export const metadata: Metadata = { title: "Search · Enterprise Knowledge AI" };

export default function SearchPage() {
  return <SearchView />;
}
