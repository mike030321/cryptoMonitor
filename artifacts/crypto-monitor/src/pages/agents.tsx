import { Link } from "wouter";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { FamilyFleet } from "@/components/family-fleet";
import { BenchmarksPanel } from "@/components/benchmarks-panel";
import { ActivityBanner } from "@/components/activity-banner";
import { Archive, ArrowRight } from "lucide-react";

export default function Agents() {
  return (
    <div className="space-y-6" data-testid="agents-page">
      <div>
        <h1
          className="text-4xl md:text-5xl font-display font-bold tracking-tight gradient-text"
          data-testid="text-page-title"
        >
          AI Agents
        </h1>
        <p className="text-sm text-muted-foreground mt-2">
          The live fleet is the four deterministic strategy executors from the
          v1 registry. Personality bots are no longer part of the decision
          path — their history lives on the Archived Agents page.
        </p>
      </div>

      <ActivityBanner />

      <FamilyFleet />

      <BenchmarksPanel />

      <Card className="bg-card/50 border-border/40" data-testid="archived-agents-card">
        <CardHeader className="pb-2 flex flex-row items-center justify-between">
          <CardTitle className="text-sm font-mono uppercase tracking-wider flex items-center gap-2">
            <Archive className="w-4 h-4" />
            Archived agents
          </CardTitle>
          <Link
            href="/agents/archived"
            className="text-xs font-mono text-primary inline-flex items-center gap-1 hover:underline"
            data-testid="link-archived-agents"
          >
            View archive <ArrowRight className="w-3 h-3" />
          </Link>
        </CardHeader>
        <CardContent>
          <p className="text-xs font-mono text-muted-foreground">
            Legacy personality bots (Breakout Bella, Contrarian Clara, Hybrid-*,
            etc.) are retained read-only for historical analytics. They no
            longer trade through the live decision engine.
          </p>
        </CardContent>
      </Card>
    </div>
  );
}
