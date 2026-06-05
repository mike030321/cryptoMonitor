import { useEffect, type ReactNode } from "react";
import { Cpu } from "lucide-react";
import { Link } from "wouter";
import { Button } from "@/components/ui/button";
// Task #444 — `LlmSidecarToggleCard` deleted along with the LLM sidecar plane.
import { TrainingContractCard } from "@/components/training-contract-card";
import { TrainingPerSliceCard } from "@/components/training-per-slice-card";
import { TrainingProgressCard } from "@/components/training-progress-card";
import { DatasetCacheSizeCard } from "@/components/dataset-cache-size-card";
import { FailureAnalysisCard } from "@/components/failure-analysis-card";
import { ModelRegistryCard } from "@/components/model-registry-card";
import { RetrainCoinCard } from "@/components/retrain-coin-card";
import { QuantAccuracyCard } from "@/components/quant-accuracy-card";
import { QuantCoverageCard } from "@/components/quant-coverage-card";
import { DirectionalCallShareCard } from "@/components/directional-call-share-card";
import { BrainConvergenceCard } from "@/components/brain-convergence-card";
import { MetaBrainCycleStatsCard } from "@/components/meta-brain-cycle-stats-card";
import { GatesAlignmentCard } from "@/components/gates-alignment-card";
import { AnomalyCancelCard } from "@/components/anomaly-cancel-card";
import { JournalHealthCard } from "@/components/journal-health-card";
import { RegimeDistributionCard } from "@/components/regime-distribution-card";
import { SpecialistAccuracyCard } from "@/components/specialist-accuracy-card";
import { MetaAbstainRateCard } from "@/components/meta-abstain-rate-card";
import { MetaModelDiagnosticsCard } from "@/components/meta-model-diagnostics-card";
import { MetaBrainStatusCard } from "@/components/meta-brain-status-card";
import { RoleOutcomeCountsCard } from "@/components/role-outcome-counts-card";
import {
  PnlBreakdownCard,
  DriftTrackerCard,
  QuarantineEventsCard,
  FeatureLabCard,
} from "@/components/phase6-diagnostics-cards";

// Section wrapper that groups related diagnostic cards under a labelled
// heading. Replaces the pre-cutover flat scroll of 24 mixed cards so
// operators can find what they need by category instead of by memory.
function Section({
  id,
  title,
  blurb,
  children,
}: {
  id?: string;
  title: string;
  blurb: string;
  children: ReactNode;
}) {
  return (
    <section id={id} className="space-y-3" data-testid={id ? `section-${id}` : undefined}>
      <div className="border-b border-border/40 pb-2">
        <h2 className="text-lg font-display font-semibold tracking-tight">{title}</h2>
        <p className="text-xs text-muted-foreground mt-0.5">{blurb}</p>
      </div>
      <div className="space-y-6">{children}</div>
    </section>
  );
}

export default function Diagnostics() {
  // Allow other pages (e.g. dashboard "open coverage" links) to deep-link
  // straight to the relevant card via `/diagnostics#quant-coverage`.
  useEffect(() => {
    const hash = window.location.hash.replace(/^#/, "");
    if (!hash) return;
    const el = document.getElementById(hash);
    if (!el) return;
    const t = window.setTimeout(() => {
      el.scrollIntoView({ behavior: "smooth", block: "start" });
    }, 50);
    return () => window.clearTimeout(t);
  }, []);

  return (
    <div className="space-y-10" data-testid="diagnostics-page">
      <div className="flex flex-col sm:flex-row sm:items-end sm:justify-between gap-4">
        <div>
          <h1 className="text-3xl md:text-4xl font-display font-bold tracking-tight gradient-text flex items-center gap-3">
            <Cpu className="w-7 h-7" />
            Brain Diagnostics
          </h1>
          <p className="text-sm text-muted-foreground mt-2 max-w-2xl">
            Internal model health, calibration, and gate alignment for the
            quant brain — the only system authorised to open or close
            positions. Grouped by what an operator usually wants to check.
          </p>
        </div>
        <Link href="/">
          <Button variant="outline" className="rounded-full">← Back to Command Center</Button>
        </Link>
      </div>

      <Section
        title="Training health"
        blurb="What the latest training run produced and whether it cleared its quality bar before being promoted."
      >
        <TrainingProgressCard />
        <TrainingContractCard />
        <TrainingPerSliceCard />
        <FailureAnalysisCard />
        <BrainConvergenceCard />
        <MetaBrainCycleStatsCard />
        <RetrainCoinCard />
      </Section>

      <Section
        title="Live calibration"
        blurb="How the deployed quant model is scoring on resolved predictions, and how its threshold knobs are evolving."
      >
        <QuantAccuracyCard />
        {/* CalibrationHistoryCard removed (B-CALIB-CARD, 2026-04-24) — the
            underlying `quant_calibration_history` table is unprovisioned in
            this environment and there is no source feeding it. The card was
            permanently empty regardless of operator action, so it has been
            removed entirely along with the `/crypto/quant-calibration-history`
            proxy route. Re-introduce only when the schema + ingestion are in
            place. */}
        <DirectionalCallShareCard />
        <SpecialistAccuracyCard />
      </Section>

      <Section
        title="Meta gate"
        blurb="The veto layer that sits between specialist signals and the trade router. Both cards are complementary — distribution shows shape, abstain rate shows how often the gate fires."
      >
        {/* Task #532 / C-11 — render the new ml-engine learning-state
            proxy card at the top of the meta-gate section so operators
            can answer "is the meta-brain actually learning?" in one
            glance instead of cross-referencing two empty distribution
            cards. */}
        <MetaBrainStatusCard />
        {/* Task #578 — per-role outcome counts so operators can verify
            the timeframe-role split (trade vs shadow vs context) is
            actually being honoured by live `record-outcome` traffic. */}
        <RoleOutcomeCountsCard />
        <MetaModelDiagnosticsCard />
        <MetaAbstainRateCard />
      </Section>

      <Section
        title="Coverage & alignment"
        blurb="Which (coin, timeframe) slices have a tradable model right now, and whether live execution still matches its own backtest."
        id="quant-coverage"
      >
        <QuantCoverageCard />
        <ModelRegistryCard />
        <GatesAlignmentCard />
      </Section>

      <Section
        title="Risk, drift & quarantine"
        blurb="Anything that can take a slice off-line: regime shifts, feature drift, anomaly cancels, quarantine events."
      >
        <RegimeDistributionCard />
        <DriftTrackerCard />
        <AnomalyCancelCard />
        <QuarantineEventsCard />
      </Section>

      <Section
        id="data-integrity"
        title="Reporting & data integrity"
        blurb="PnL attribution, dataset cache footprint, and the journal-write health that the rest of these cards depend on."
      >
        <PnlBreakdownCard />
        <DatasetCacheSizeCard />
        <JournalHealthCard />
      </Section>

      <Section
        title="Experimental"
        blurb="Feature-engineering scratchpad. The LLM sidecar toggle was removed in Task #444 — the deterministic quant brain is now the sole authority."
      >
        <FeatureLabCard />
      </Section>
    </div>
  );
}
