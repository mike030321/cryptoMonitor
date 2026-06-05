import { useState } from "react";
import { Shield, Flame, TrendingUp, Zap, Star } from "lucide-react";
import { cn } from "@/lib/utils";
import { ContactFormDialog } from "@/components/contact-form-dialog";

const TIERS = [
  {
    id: "low",
    label: "Low Risk",
    icon: Shield,
    badge: null,
    color: {
      accent: "from-emerald-500/20 to-emerald-600/10",
      border: "border-emerald-500/25",
      iconBg: "bg-emerald-500/15",
      iconColor: "text-emerald-400",
      badgeBg: "",
      glow: "shadow-emerald-500/10",
      bullet: "bg-emerald-400",
      cta: "bg-emerald-500/20 hover:bg-emerald-500/30 text-emerald-300 border border-emerald-500/30",
    },
    headline: "Safe & Steady",
    subheadline: "Preserve capital. Earn consistently.",
    description:
      "The investment will be small, but the chance of profit is low. Designed for those who value security above all else, this tier places measured, well-protected positions in the most stable market conditions. Your capital remains largely shielded from volatility — what you gain comes slowly, but it comes reliably. Ideal for first-time investors or those who simply cannot afford to absorb losses.",
    bullets: [
      "Minimal exposure to market swings",
      "Capital preservation as the first priority",
      "Steady, low-volatility entry points",
      "Best suited for cautious, patient investors",
    ],
    cta: "Start Safely",
  },
  {
    id: "medium",
    label: "Medium Risk",
    icon: TrendingUp,
    badge: "Popular",
    color: {
      accent: "from-blue-500/20 to-violet-600/10",
      border: "border-blue-500/30",
      iconBg: "bg-blue-500/15",
      iconColor: "text-blue-400",
      badgeBg: "bg-blue-500/20 text-blue-300 border border-blue-500/30",
      glow: "shadow-blue-500/10",
      bullet: "bg-blue-400",
      cta: "bg-blue-500/20 hover:bg-blue-500/30 text-blue-300 border border-blue-500/30",
    },
    headline: "Balanced Growth",
    subheadline: "Smart exposure. Meaningful returns.",
    description:
      "The investment will be average, but the chance of winning is average. This tier strikes a calculated balance between protection and opportunity — deploying moderate capital into positions that have been validated by multiple signals. You participate meaningfully in the market's upswings while maintaining enough of a buffer to weather corrections. A rational, proven approach for traders who want real growth without gambling everything on a single outcome.",
    bullets: [
      "Diversified entries across multiple timeframes",
      "Balanced drawdown management with upside exposure",
      "Consistent wins across varying market conditions",
      "Trusted by the majority of active traders",
    ],
    cta: "Grow Steadily",
  },
  {
    id: "high",
    label: "High Risk",
    icon: Flame,
    badge: "Maximum Returns",
    color: {
      accent: "from-orange-500/30 via-red-500/20 to-pink-600/15",
      border: "border-orange-500/50",
      iconBg: "bg-gradient-to-br from-orange-500/30 to-red-500/20",
      iconColor: "text-orange-400",
      badgeBg: "bg-gradient-to-r from-orange-500/30 to-red-500/20 text-orange-300 border border-orange-500/40",
      glow: "shadow-orange-500/20",
      bullet: "bg-gradient-to-r from-orange-400 to-red-400",
      cta: "bg-gradient-to-r from-orange-500 to-red-500 hover:from-orange-400 hover:to-red-400 text-white shadow-lg shadow-orange-500/30",
    },
    headline: "Unleash Maximum Potential",
    subheadline: "One correct call. Everything changes.",
    description:
      "The investment will be high, but the chance of making a large amount of money is high. This is where fortunes are forged. High-risk positions are engineered for those who understand one undeniable truth: the greatest rewards in the history of financial markets have always belonged to those who dared to act decisively while everyone else hesitated. This tier deploys aggressive capital at the sharpest, highest-conviction moments the algorithm identifies — moments where the asymmetry between risk and reward is at its most extreme. A single cycle. A single precise entry. A move that compounds into something the conservative tiers could never touch. This is not reckless gambling — it is calculated aggression. It is the strategy of the elite, the methodology of those who have studied the market long enough to know that playing it safe has never made anyone wealthy. Your capital works at peak intensity, riding explosive price action with the precision of a machine that never sleeps, never panics, never second-guesses. The upside is real. The window is finite. The question is whether you have the conviction to step through it.",
    bullets: [
      "Highest-conviction entries at peak signal strength",
      "Aggressive positioning at optimal market asymmetry",
      "Engineered for life-changing return potential",
      "Reserved for traders who think in multiples, not percentages",
      "Maximum capital deployment at the most explosive moments",
      "Where the algorithm operates at full, unrestricted power",
    ],
    cta: "Claim Maximum Returns",
    featured: true,
  },
];

const CAPTION =
  "Our representative will contact you for confirmation once you submit the form. " +
  "Please provide your details to proceed.";

export default function PriceList() {
  const [formOpen, setFormOpen] = useState(false);

  return (
    <div className="space-y-8">
      {/* Header */}
      <div className="space-y-2">
        <div className="flex items-center gap-2">
          <Star className="w-5 h-5 text-primary" />
          <h1 className="text-2xl font-display font-bold gradient-text">
            Investment Tiers
          </h1>
        </div>
        <p className="text-muted-foreground text-sm max-w-xl">
          Choose the tier that matches your appetite for risk and reward. Each level is
          calibrated by the AI engine to deploy capital with precision at different
          thresholds of market conviction.
        </p>
      </div>

      {/* Cards */}
      <div className="grid gap-6 md:grid-cols-3">
        {TIERS.map((tier) => {
          const Icon = tier.icon;
          return (
            <div
              key={tier.id}
              className={cn(
                "relative rounded-2xl border bg-gradient-to-br p-6 flex flex-col gap-5 transition-all duration-300",
                tier.color.accent,
                tier.color.border,
                tier.featured
                  ? `shadow-2xl ${tier.color.glow} ring-1 ring-orange-500/30 scale-[1.02]`
                  : `shadow-lg ${tier.color.glow}`,
              )}
            >
              {/* Featured glow overlay */}
              {tier.featured && (
                <div className="absolute inset-0 rounded-2xl bg-gradient-to-br from-orange-500/5 to-red-500/5 pointer-events-none" />
              )}

              {/* Badge */}
              {tier.badge && (
                <span
                  className={cn(
                    "absolute -top-3 left-1/2 -translate-x-1/2 px-4 py-1 rounded-full text-xs font-semibold tracking-wide whitespace-nowrap",
                    tier.color.badgeBg,
                  )}
                >
                  {tier.featured && <Zap className="inline w-3 h-3 mr-1 -mt-0.5" />}
                  {tier.badge}
                </span>
              )}

              {/* Icon + Label */}
              <div className="flex items-center gap-3 pt-2">
                <span
                  className={cn(
                    "w-11 h-11 rounded-xl flex items-center justify-center",
                    tier.color.iconBg,
                  )}
                >
                  <Icon className={cn("w-5 h-5", tier.color.iconColor)} />
                </span>
                <div>
                  <p className="text-xs font-semibold uppercase tracking-widest text-muted-foreground">
                    {tier.label}
                  </p>
                  <h2 className="font-display font-bold text-lg leading-tight text-foreground">
                    {tier.headline}
                  </h2>
                </div>
              </div>

              {/* Subheadline */}
              <p className={cn("text-sm font-semibold", tier.color.iconColor)}>
                {tier.subheadline}
              </p>

              {/* Description */}
              <p className="text-sm text-muted-foreground leading-relaxed flex-1">
                {tier.description}
              </p>

              {/* Divider */}
              <div className={cn("h-px w-full opacity-30", `bg-gradient-to-r ${tier.color.accent}`)} />

              {/* Bullets */}
              <ul className="space-y-2">
                {tier.bullets.map((b) => (
                  <li key={b} className="flex items-start gap-2.5 text-sm text-foreground/80">
                    <span
                      className={cn(
                        "mt-1.5 w-1.5 h-1.5 rounded-full flex-shrink-0",
                        tier.color.bullet,
                      )}
                    />
                    {b}
                  </li>
                ))}
              </ul>

              {/* CTA — opens contact form */}
              <button
                onClick={() => setFormOpen(true)}
                className={cn(
                  "mt-auto w-full py-3 px-4 rounded-xl text-sm font-semibold transition-all duration-200 cursor-pointer",
                  tier.color.cta,
                )}
              >
                {tier.cta}
              </button>
            </div>
          );
        })}
      </div>

      {/* Disclaimer */}
      <p className="text-xs text-muted-foreground/60 text-center max-w-2xl mx-auto leading-relaxed">
        All investment tiers involve real market risk. Past performance of the AI engine
        does not guarantee future results. Only invest capital you can afford to lose.
        High-risk positions may result in significant or total loss of invested capital.
      </p>

      {/* Contact form — auto-opens when Price List page is visited */}
      <ContactFormDialog
        controlledOpen={formOpen}
        onControlledOpenChange={setFormOpen}
        captionText={CAPTION}
      />
    </div>
  );
}
