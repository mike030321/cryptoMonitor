/**
 * Recommended number of coins each agent should analyze per cycle.
 *
 * History: this used to live in `rate-limiter.ts` and was derived from the
 * current LLM "mode" (dual / single-primary / economy), which itself was
 * derived from gpt/gemini daily budget burn. Task #444 removed the LLM
 * plane, so nothing records API calls anymore — the mode was structurally
 * pinned to "dual" forever and the budget tracking was dead code. Task
 * #453 deletes `rate-limiter.ts` entirely; this module preserves the one
 * helper (`getRecommendedAgentCoins`) that `monitor.ts` still calls when
 * sizing per-agent coin baskets.
 *
 * The return value matches the previous "dual" branch (3) so cycle behavior
 * is byte-identical to the post-#444 steady state.
 */
export function getRecommendedAgentCoins(): number {
  return 3;
}
