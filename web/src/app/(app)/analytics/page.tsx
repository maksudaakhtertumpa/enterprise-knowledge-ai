import type { Metadata } from "next";

import { AnalyticsView } from "@/components/views/analytics-view";

export const metadata: Metadata = { title: "Analytics · Enterprise Knowledge AI" };

export default function AnalyticsPage() {
  return <AnalyticsView />;
}
