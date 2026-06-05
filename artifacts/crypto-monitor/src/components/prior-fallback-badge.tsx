import { Info } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";

interface PriorFallbackBadgeProps {
  excludedCount: number;
  unit: "predictions" | "trades";
  included: boolean;
  onToggle: () => void;
  className?: string;
  testId?: string;
}

export function PriorFallbackBadge({
  excludedCount,
  unit,
  included,
  onToggle,
  className,
  testId,
}: PriorFallbackBadgeProps) {
  if (excludedCount <= 0) return null;
  const noun = unit === "trades" ? "trade" : "prediction";
  return (
    <div
      className={cn(
        "rounded-md border border-amber-500/40 bg-amber-500/5 p-2 text-[11px] text-amber-200 flex items-start gap-2",
        className,
      )}
      data-testid={testId ?? "prior-fallback-badge"}
    >
      <Info className="h-3.5 w-3.5 flex-shrink-0 mt-0.5" />
      <div className="flex-1 space-y-1">
        <div className="flex flex-wrap items-center gap-2">
          <Badge
            variant="outline"
            className="border-amber-500/50 text-amber-200 text-[10px] uppercase tracking-wider"
          >
            Prior fallback
          </Badge>
          <span>
            {excludedCount.toLocaleString()} {noun}
            {excludedCount === 1 ? "" : "s"}{" "}
            {included ? "folded into" : "excluded from"} the headline number.
          </span>
        </div>
        <div className="text-amber-200/80">
          These came from the Laplace-smoothed pooled prior (a fallback used
          when a timeframe has no trained LightGBM model yet), so every {noun}{" "}
          shares the same marginal probability and would drag the headline
          toward structural noise rather than skill.
        </div>
        <Button
          size="sm"
          variant="outline"
          onClick={onToggle}
          className="h-6 px-2 text-[10px] border-amber-500/40 text-amber-200 hover:bg-amber-500/10 hover:text-amber-100"
          data-testid={`${testId ?? "prior-fallback-badge"}-toggle`}
        >
          {included ? "Hide fallback rows" : "Show fallback rows"}
        </Button>
      </div>
    </div>
  );
}
