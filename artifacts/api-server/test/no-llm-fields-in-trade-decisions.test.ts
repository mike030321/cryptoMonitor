/**
 * Task #369 — block any future LLM-derived signal from sneaking back into
 * the live trade-decision path.
 *
 * Sister tests already cover the model boundary
 * (`test_quantonly_enforcement` in ml-engine guards `FEATURE_COLUMNS` +
 * `load_model`) and the friction-literal boundary
 * (`quantonly-enforcement.test.ts` regex-scans the source tree for fee /
 * slippage values). What was NOT yet covered: the live request path that
 * flows from `quant-brain.ts` through every gate in `paper-trader.ts`
 * and finally into the order executor (the rows inserted into
 * `paper_trades` / `paper_positions`).
 *
 * Strategy: instead of mirroring the trader and risking drift, this test
 * statically extracts every object-literal payload built at the live
 * call sites in the production source — `recordSkip(...)` contexts,
 * `getMlDecision({...})` requests, `tx.insert(paperTradesTable).values({...})`
 * / `tx.insert(paperPositionsTable).values({...})` rows, and the
 * "Paper trade opened" `logger.info({...})` payload — and the
 * `QuantPrediction` return object emitted by `quant-brain.ts`. Each
 * key found at any of those sites is asserted against an explicit
 * ALLOWLIST of legitimate quant-only field names. Adding a key like
 * `newsBias`, `llmEdge`, `sentimentScore`, or even an innocent-sounding
 * non-quant field to any of those payloads will fail this test.
 *
 * Spread sources (`...portfolioCheck.breakdown`) are followed: every
 * key the source can write is enumerated against the allowlist as
 * well.
 *
 * Belt-and-suspenders: a forbidden-prefix regex scan over the same
 * source strips comments / string literals first and asserts no
 * identifier (anywhere) matches the LLM-derived patterns
 * (`news_*`, `llm_*`, `gpt_*`, `sentiment_*`, `ai_*`, plus camelCase
 * and embedded-marker variants like `newsTag`, `llmBias`,
 * `sentimentScore`, `geminiVote`, `rawNewsScore`, `chatgptCall`).
 *
 * Negative control: the extractor and the forbidden-prefix scanner are
 * each fed deliberately tainted synthetic source to guarantee they
 * fire when they should. Without that, a buggy extractor / scanner
 * could silently approve every fixture.
 *
 * Wired into the parity workflow set
 * (`decision-engine-parity` validation command) so it runs on every
 * change alongside the existing parity / cadence checks.
 */
import { describe, test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const SRC = path.resolve(__dirname, "..", "src", "lib");
const PAPER_TRADER = readFileSync(path.join(SRC, "paper-trader.ts"), "utf8");
const QUANT_BRAIN = readFileSync(path.join(SRC, "quant-brain.ts"), "utf8");
const PORTFOLIO_CONSTRAINTS = readFileSync(
  path.join(SRC, "portfolio-constraints.ts"),
  "utf8",
);
// Task #371 — close-path source. Read here so the new sweep can scan
// every payload reachable from `closeExpiredPositions` → journal-writer.
const JOURNAL_WRITER = readFileSync(path.join(SRC, "journal-writer.ts"), "utf8");

// =============================================================================
// Strip comments + string-literal contents (preserve quotes so the
// balanced-brace parser still sees structure) so identifiers found by the
// scanners are real code, not log-message fragments or doc-block prose.
// =============================================================================
function stripCommentsAndStrings(src: string): string {
  let out = "";
  let i = 0;
  while (i < src.length) {
    const c = src[i];
    const next = src[i + 1];
    if (c === "/" && next === "/") {
      while (i < src.length && src[i] !== "\n") i++;
      continue;
    }
    if (c === "/" && next === "*") {
      i += 2;
      while (i < src.length && !(src[i] === "*" && src[i + 1] === "/")) i++;
      i += 2;
      continue;
    }
    if (c === '"' || c === "'" || c === "`") {
      const quote = c;
      out += c;
      i++;
      while (i < src.length && src[i] !== quote) {
        if (src[i] === "\\") {
          i += 2;
          continue;
        }
        if (quote === "`" && src[i] === "$" && src[i + 1] === "{") {
          // walk a template expression — keep braces balanced
          out += "${";
          i += 2;
          let depth = 1;
          while (i < src.length && depth > 0) {
            if (src[i] === "{") depth++;
            else if (src[i] === "}") depth--;
            if (depth > 0) {
              out += src[i];
              i++;
            }
          }
          out += "}";
          i++;
          continue;
        }
        i++;
      }
      out += quote;
      i++;
      continue;
    }
    out += c;
    i++;
  }
  return out;
}

// =============================================================================
// Find a call expression by its open-paren position. Returns the substring
// between the matching parens, with comments / string contents already
// stripped by the caller passing cleaned source.
// =============================================================================
function readBalanced(src: string, openIdx: number, open: string, close: string): { body: string; endIdx: number } {
  assert.equal(src[openIdx], open, `expected '${open}' at index ${openIdx}`);
  let depth = 0;
  for (let i = openIdx; i < src.length; i++) {
    const c = src[i];
    if (c === open) depth++;
    else if (c === close) {
      depth--;
      if (depth === 0) {
        return { body: src.slice(openIdx + 1, i), endIdx: i };
      }
    }
  }
  throw new Error(`unbalanced '${open}'..'${close}' starting at ${openIdx}`);
}

// =============================================================================
// Split a comma-separated argument list at top-level commas (depth 0 over
// (), [], {}). String contents already stripped by caller.
// =============================================================================
function splitTopLevelArgs(body: string): string[] {
  const out: string[] = [];
  let depth = 0;
  let start = 0;
  for (let i = 0; i < body.length; i++) {
    const c = body[i];
    if (c === "(" || c === "[" || c === "{") depth++;
    else if (c === ")" || c === "]" || c === "}") depth--;
    else if (c === "," && depth === 0) {
      out.push(body.slice(start, i));
      start = i + 1;
    }
  }
  if (start < body.length) out.push(body.slice(start));
  return out.map((s) => s.trim()).filter((s) => s.length > 0);
}

// =============================================================================
// Extract every property name (top-level + recursive into nested object
// literals and array literals) from an object-literal source body. Handles:
//   - shorthand:    { agentName, coinName }
//   - long form:    { agentName: agent.name }
//   - quoted keys:  { "agentName": ... }
//   - computed:     { [k]: ... }   ← rejected explicitly (must be statically allowlistable)
//   - spread:       { ...x }       ← reported separately so caller can validate the source
// =============================================================================
interface ExtractedKeys {
  keys: string[];
  spreads: string[]; // expressions inside `...expr`
}
function extractObjectKeysRecursive(objBody: string): ExtractedKeys {
  const keys: string[] = [];
  const spreads: string[] = [];
  // Walk top-level entries; for each entry, capture the key and (if value is
  // an object/array) recurse into it.
  const entries = splitTopLevelArgs(objBody);
  for (const entryRaw of entries) {
    const entry = entryRaw.trim();
    if (entry.length === 0) continue;
    if (entry.startsWith("...")) {
      spreads.push(entry.slice(3).trim());
      continue;
    }
    // Computed property keys are not allowlistable — reject loudly.
    if (entry.startsWith("[")) {
      throw new Error(
        `computed property key is not allowlistable: ${entry.slice(0, 80)}`,
      );
    }
    // shorthand `foo` (no colon) vs `foo: value`
    // careful: a default value `foo = bar` doesn't appear in object literals,
    // and method `foo() {}` we treat the same as a key.
    // Find the first ':' at depth 0 within this entry.
    let depth = 0;
    let colonIdx = -1;
    for (let i = 0; i < entry.length; i++) {
      const c = entry[i];
      if (c === "(" || c === "[" || c === "{") depth++;
      else if (c === ")" || c === "]" || c === "}") depth--;
      else if (c === ":" && depth === 0) {
        colonIdx = i;
        break;
      }
    }
    let keyToken: string;
    let valueToken: string | null;
    if (colonIdx === -1) {
      // shorthand or method
      const m = entry.match(/^([A-Za-z_$][\w$]*)\s*(\(.*)?$/);
      if (!m) {
        throw new Error(`could not parse shorthand entry: ${entry.slice(0, 80)}`);
      }
      keyToken = m[1];
      valueToken = null;
    } else {
      const lhs = entry.slice(0, colonIdx).trim();
      valueToken = entry.slice(colonIdx + 1).trim();
      // strip surrounding quotes for "foo" / 'foo'
      if (
        (lhs.startsWith('"') && lhs.endsWith('"')) ||
        (lhs.startsWith("'") && lhs.endsWith("'"))
      ) {
        keyToken = lhs.slice(1, -1);
      } else {
        // bare identifier
        const m = lhs.match(/^([A-Za-z_$][\w$]*)$/);
        if (!m) {
          throw new Error(`unrecognized key form: ${lhs.slice(0, 80)}`);
        }
        keyToken = m[1];
      }
    }
    keys.push(keyToken);
    if (valueToken !== null) {
      // Recurse into nested object / array literals — but ONLY if the value
      // is purely a literal (starts with { or [). For mixed expressions
      // like `regime?.regimeLabel ?? null`, we still want to descend into
      // any object literal that appears anywhere inside, so scan for the
      // first balanced { or [ and descend if found.
      collectNestedLiterals(valueToken, keys, spreads);
    }
  }
  return { keys, spreads };
}

function collectNestedLiterals(expr: string, keys: string[], spreads: string[]): void {
  // Walk the expression; whenever we hit '{' or '[' at depth 0 of the
  // surrounding expression, treat the contents as a nested literal and
  // recurse. This handles: `existingPositions.map((pos) => ({ ... }))`
  // and `[{ ... }, { ... }]`.
  let depth = 0;
  let i = 0;
  while (i < expr.length) {
    const c = expr[i];
    if (c === "(") depth++;
    else if (c === ")") depth--;
    else if (c === "{") {
      // Could be a code block (arrow body) or an object literal. To be
      // safe, descend either way — code blocks contain statements, not
      // shorthand keys, but our parser will fail loudly if it can't
      // recognize an entry, which is the desired behavior.
      const inner = readBalanced(expr, i, "{", "}");
      // Heuristic: try parsing as object literal; if that fails because the
      // body looks like statements, scan its inside for nested literals
      // instead.
      try {
        const sub = extractObjectKeysRecursive(inner.body);
        keys.push(...sub.keys);
        spreads.push(...sub.spreads);
      } catch {
        collectNestedLiterals(inner.body, keys, spreads);
      }
      i = inner.endIdx + 1;
      continue;
    } else if (c === "[") {
      const inner = readBalanced(expr, i, "[", "]");
      collectNestedLiterals(inner.body, keys, spreads);
      i = inner.endIdx + 1;
      continue;
    }
    i++;
  }
}

// =============================================================================
// Locate every callsite of a callee name in cleaned source and return the
// (1-based) argument index requested as the raw argument source (still
// cleaned — strings / comments stripped).
//
// `calleeRegex` matches the callee expression immediately followed by `(`.
// For chained calls like `tx.insert(paperTradesTable).values(`, pass the
// full `\.values\b` pattern as a separate matcher and verify the chain
// prefix yourself.
// =============================================================================
function findCallArgs(cleaned: string, calleeRegex: RegExp): Array<{ args: string[]; pos: number }> {
  const out: Array<{ args: string[]; pos: number }> = [];
  let m: RegExpExecArray | null;
  const re = new RegExp(calleeRegex.source, calleeRegex.flags.includes("g") ? calleeRegex.flags : calleeRegex.flags + "g");
  while ((m = re.exec(cleaned)) !== null) {
    // Find the next '(' after the match end
    let i = m.index + m[0].length;
    while (i < cleaned.length && /\s/.test(cleaned[i])) i++;
    if (cleaned[i] !== "(") continue;
    const { body, endIdx } = readBalanced(cleaned, i, "(", ")");
    out.push({ args: splitTopLevelArgs(body), pos: m.index });
    re.lastIndex = endIdx + 1;
  }
  return out;
}

// =============================================================================
// ALLOWLIST — the canonical set of object-literal keys that may appear in
// any payload reachable from the live trade-decision path. Every key is
// strictly quant / portfolio / execution metadata. Adding ANY new key to
// the production source requires extending this list, which keeps the
// review surface tight: the diff makes it impossible to slip in an
// LLM-derived field unnoticed.
//
// Grouped for review-friendliness; the test merges them into one set.
// =============================================================================
const ALLOWED_KEYS = new Set<string>([
  // identity
  "agentId", "agentName", "coinId", "coinName", "timeframe", "direction",
  "brain", "tradeId", "predictionId",
  // sizing / execution
  "action", "entryPrice", "entryFee", "quantity", "positionSize",
  "stopLoss", "takeProfit", "stopLossPrice", "takeProfitPrice",
  "peakPrice", "expiresAt", "status", "slippage", "balance",
  "liveCash", "want", "closed", "trendBias", "minMomentum",
  "eligibleCount", "totalCoins", "regime", "cash",
  // gate inputs / thresholds
  "confidence", "required", "requiredPct", "threshold",
  "tpDistancePct", "atrPct", "evScore", "evRequired",
  "sameSide", "totalOpen", "sameSideShare",
  "dailyLossLimit", "drawdownHalt", "dailyLossHit", "drawdownHit",
  "avgChange24h", "err",
  // quant model surface
  "probUp", "probDown", "probStable", "expectedReturnPct",
  "directionalReturnPct", "modelVersion", "source", "predictionStdPct",
  "rawConfidence", "featureHash", "rawConfidence",
  // meta-model surface
  "metaAction", "metaKind", "metaVersion", "metaExpectedEdgePct",
  "metaSizeMultiplier", "metaAbstainReason", "metaGate",
  // /ml/decide request shape
  "lastPrice", "atrValue", "portfolio", "equityUsd", "cashUsd",
  "openPositions", "newNotionalUsd", "notionalUsd", "regimeAtEntry",
  "betaToBtc", "gates", "base",
  // PredictionResult / QuantPrediction return surface
  "reasoning", "predictedPrice", "noModelAvailable", "specialists",
  "quant", "coin",
  // checkPortfolioConstraints breakdown spread keys (enumerated explicitly
  // from portfolio-constraints.ts so the source is auditable)
  "enabled", "sector_share_after", "sector_cap", "correlated_cap",
  "book_beta", "beta_cap", "regime_share_after", "regime_budget",
  "sector", "correlatedCoins", "new_sector", "open_notional_usd",
  "new_notional_usd",
  // journal / position carry-through
  "entryRegimeLabel",
  // SkipReason / engine-decision passthrough
  "skipReason", "skipDetail", "gatesApplied", "portfolioCheck",
  // logger payload context
  "horizon", "fleet",
  // Task #371 — close-path payloads (writeTradeJournal args, the
  // tradeJournalTable INSERT row, the close/cancel UPDATE patches on
  // paper_trades, and the structured close/cancel logs). Every key here
  // is strictly execution / pricing / accounting metadata produced from
  // the position row and the live price tape — no LLM-derived field is
  // permitted in any of them.
  "predictionJournalId",
  "entryTime", "exitTime", "closedAt",
  "entryPriceRaw", "entryPriceAdj", "exitPriceRaw", "exitPriceAdj",
  "exitFee", "slippagePct",
  "positionSizeUsd",
  "mfePct", "maePct",
  "exitReason",
  "realizedPnlUsd", "realizedPnlPct",
  "regimeLabel",
  "counterfactualBetter",
  "exitPrice", "pnl", "pnlPercent",
  "rawPnl", "netPnl", "closeReason",
  "priceRatio", "refunded", "entryFeeReversed",
  // Trailing-stop extension log keys (also part of the close-path code:
  // the same loop that decides whether to close vs extend).
  "peakPnlPct", "currentPnlPct", "newStopPrice", "extensionMin",
  // closeExpiredPositions summary log
  "closed",
  // Task #468 — deterministic strategy-profile registry context. These
  // are static, code-resident, non-LLM values (registry profile id /
  // status / regime list, and the registry-derived numeric thresholds)
  // surfaced in skip-context payloads so the operator can see which
  // profile rejected a trade and against which floor.
  "profileId", "profileStatus", "profileFloor", "profileMinEdge",
  "blockedRegimes", "directionalReturnFrac",
  // Task #468 — registry-derived per-field policy values surfaced in
  // gate skip context: the agent's confidence floor before any
  // abstain-bias scaling (`baseFloor`), the bias multiplier itself
  // (`abstainBias`), and the live drawdown vs the per-agent
  // sensitivity-scaled threshold (`drawdownNow`, `drawdownSensitivity`).
  // All four are static profile values × deterministic portfolio math —
  // no LLM input.
  "baseFloor", "abstainBias", "drawdownNow", "drawdownSensitivity",
  // Task #468 — pooled-fallback flag surfaced from /ml/predict
  // (`"pooled" | null`). Triggers the registry's
  // `pooled_fallback_penalty` shrink in sizing; static and
  // model-routing-derived, never LLM-derived.
  "fallback",
]);

// Spread sources that are statically allowlistable: each must point to a
// place whose entire key universe is also covered by ALLOWED_KEYS above.
const ALLOWED_SPREAD_SOURCES = new Set<string>([
  "portfolioCheck.breakdown", // keys enumerated in portfolio-constraints.ts (audited below)
  "fallback",                 // PredictionResult literal built by abstain()
  "base",                     // QuantPrediction.metaGate base struct
]);

// =============================================================================
// FORBIDDEN-PREFIX scanner — belt-and-suspenders. Any identifier in the
// source that matches an LLM-derived pattern fails the test, regardless of
// whether it appears in a payload. Catches helper functions, types, etc.
// =============================================================================
const SNAKE_FORBIDDEN = /\b(news|llm|gpt|sentiment|ai|chatgpt|gemini|openai|anthropic|claude)_[a-z][\w]*/g;
const CAMEL_FORBIDDEN_PREFIX = /\b(news|llm|gpt|sentiment|ai|chatgpt|gemini|openai|anthropic|claude)[A-Z][\w]*/g;
// embedded markers like `rawNewsScore`, `priceLlmEdge`
const EMBEDDED_FORBIDDEN = /\b\w*?(News|Llm|Gpt|Sentiment|Chatgpt|Gemini|OpenAi|Anthropic|Claude)(?:[A-Z]\w*|Score|Bias|Tag|Vote|Edge|Signal|Rating|Call)\b/g;
// Task #390 — benchmark / Strategy Lab governance signal must NEVER
// appear on the trade-decision path. Catches `relativeAlpha14d`,
// `bestBaselineReturn7d`, `aiReturn7d` (the `ai*` part is also caught
// by CAMEL_FORBIDDEN_PREFIX), `benchmarkTrust`, `strategyLab*`, and
// snake / embedded variants.
const SNAKE_FORBIDDEN_BENCHMARK = /\b(benchmark|alpha|baseline|strategy_lab)_[a-z][\w]*/g;
const CAMEL_FORBIDDEN_BENCHMARK = /\b(benchmark|strategyLab)[A-Z][\w]*/g;
const EMBEDDED_FORBIDDEN_BENCHMARK = /\b\w*?(Benchmark|StrategyLab|Alpha|Baseline)(?:[A-Z]\w*|Score|Return|Ratio|Trust)\b/g;

const FORBIDDEN_FALSE_POSITIVE_ALLOWLIST = new Set<string>([
  // "AI bot" identifiers refer to internal agent entities (Hodler Hank
  // baseline, auto-deploy bots) in non-trade-decision code paths, not
  // LLM-derived signals. Pre-date this test; audited to confirm they
  // carry no model output.
  "aiBotAgents", "aiBotIds",
  // Task #468 — registry profile field names. These are *static
  // configuration* attached to deterministic agent profiles (see
  // `lib/agents-registry/`); the scanner's `benchmark_*` regex catches
  // the field name itself when paper-trader.ts reads
  // `profile.benchmark_sensitivity` to scale the family-multiplier
  // shrink, but the value is a code-resident float and never sourced
  // from Strategy Lab telemetry. `baseline_*` field names appearing
  // in strategy-family identifiers (`baseline_reference` profile id
  // and the `baseline_buy_hold` / `baseline_dca_cb` /
  // `baseline_trend_filter` sub-id tags surfaced from
  // `mapLegacyNameToSubId()`) are static code-resident strings, not
  // LLM-sourced telemetry — see registry.ts and compat.ts.
  "benchmark_sensitivity",
  "baseline_buy_hold", "baseline_dca_cb", "baseline_trend_filter",
  "baseline_reference",
]);

function scanForbiddenIdentifiers(cleaned: string): string[] {
  const hits = new Set<string>();
  for (const re of [
    SNAKE_FORBIDDEN, CAMEL_FORBIDDEN_PREFIX, EMBEDDED_FORBIDDEN,
    SNAKE_FORBIDDEN_BENCHMARK, CAMEL_FORBIDDEN_BENCHMARK, EMBEDDED_FORBIDDEN_BENCHMARK,
  ]) {
    const r = new RegExp(re.source, "g");
    let m: RegExpExecArray | null;
    while ((m = r.exec(cleaned)) !== null) {
      const ident = m[0];
      if (!FORBIDDEN_FALSE_POSITIVE_ALLOWLIST.has(ident)) hits.add(ident);
    }
  }
  return Array.from(hits);
}

// =============================================================================
// Audit: enumerate the keys that portfolio-constraints.ts can write into
// `breakdown` so the `...portfolioCheck.breakdown` spread inside
// paper-trader.ts is statically bounded.
// =============================================================================
function extractBreakdownKeys(cleaned: string): string[] {
  // matches: breakdown.foo = ... AND breakdown["foo"] = ... AND object literal
  // initializer { foo: ..., bar: ... }
  const keys = new Set<string>();
  const reDot = /\bbreakdown\.([A-Za-z_][\w]*)\b/g;
  const reBracket = /\bbreakdown\[\s*['"]([A-Za-z_][\w]*)['"]\s*\]/g;
  let m: RegExpExecArray | null;
  while ((m = reDot.exec(cleaned)) !== null) keys.add(m[1]);
  while ((m = reBracket.exec(cleaned)) !== null) keys.add(m[1]);
  // Also the initializer literal in: const breakdown: ... = { ... };
  const initRe = /\bconst\s+breakdown[^=]*=\s*\{/;
  const init = initRe.exec(cleaned);
  if (init !== null) {
    const openIdx = cleaned.indexOf("{", init.index);
    if (openIdx !== -1) {
      const { body } = readBalanced(cleaned, openIdx, "{", "}");
      const sub = extractObjectKeysRecursive(body);
      for (const k of sub.keys) keys.add(k);
    }
  }
  return Array.from(keys);
}

// =============================================================================
// Find the cleaned-source fragment for a chained call like
//   tx.insert(paperTradesTable).values(
// and return the first `.values(...)` arg body.
// =============================================================================
function findChainedInsertValues(
  cleaned: string,
  tableName: string,
  receiverPattern: string = "tx",
): string[] {
  const out: string[] = [];
  const re = new RegExp(
    String.raw`\b${receiverPattern}\s*\.\s*insert\s*\(\s*${tableName}\s*\)\s*\.\s*values\s*\(`,
    "g",
  );
  let m: RegExpExecArray | null;
  while ((m = re.exec(cleaned)) !== null) {
    const openIdx = m.index + m[0].length - 1;
    const { body } = readBalanced(cleaned, openIdx, "(", ")");
    const args = splitTopLevelArgs(body);
    assert.equal(args.length, 1, `${tableName}.values(...) must take exactly one object arg`);
    out.push(args[0].replace(/^\{/, "").replace(/\}$/, ""));
  }
  return out;
}

// =============================================================================
// Locate the "Paper trade opened" logger.info call and return its first arg
// (the structured-log object).
// =============================================================================
function findPaperTradeOpenedLog(rawSrc: string): string {
  // The cleaned source has string contents emptied, so use the ORIGINAL
  // source to locate the landmark string, then walk back to the opening
  // `logger.info(` and re-clean just that call site for structural parsing.
  const idx = rawSrc.indexOf('"Paper trade opened"');
  assert.notEqual(idx, -1, "expected to find 'Paper trade opened' log message");
  const head = rawSrc.slice(0, idx);
  const callIdx = head.lastIndexOf("logger.info(");
  assert.notEqual(callIdx, -1, "expected logger.info( before 'Paper trade opened'");
  // Clean the slice from callIdx onwards so balanced parsing is safe.
  const fragment = stripCommentsAndStrings(rawSrc.slice(callIdx));
  const openIdx = fragment.indexOf("(");
  const { body } = readBalanced(fragment, openIdx, "(", ")");
  const args = splitTopLevelArgs(body);
  assert.equal(args.length, 2, "Paper trade opened logger.info must take (obj, msg)");
  const obj = args[0].trim();
  assert.ok(obj.startsWith("{") && obj.endsWith("}"), "first arg must be an object literal");
  return obj.slice(1, -1);
}

// =============================================================================
// Tests
// =============================================================================
describe("Task #369 — no LLM-derived field can reach the live trade path", () => {
  const cleanedTrader = stripCommentsAndStrings(PAPER_TRADER);
  const cleanedBrain = stripCommentsAndStrings(QUANT_BRAIN);
  const cleanedConstraints = stripCommentsAndStrings(PORTFOLIO_CONSTRAINTS);

  test("portfolio-constraints.breakdown writes only allowlisted keys (audit for the spread)", () => {
    const keys = extractBreakdownKeys(cleanedConstraints);
    assert.ok(keys.length > 0, "expected breakdown to define at least one key");
    const forbidden: string[] = [];
    for (const k of keys) {
      if (!ALLOWED_KEYS.has(k)) forbidden.push(k);
    }
    assert.deepEqual(
      forbidden, [],
      `portfolio-constraints.ts breakdown writes keys not in ALLOWED_KEYS: ${forbidden.join(", ")}.\n` +
      `If new fields were added, audit them and append to ALLOWED_KEYS.`,
    );
  });

  test("recordSkip context objects in paper-trader.ts only contain allowlisted keys", () => {
    const calls = findCallArgs(cleanedTrader, /\brecordSkip\b/);
    assert.ok(calls.length >= 8, `expected ≥8 recordSkip calls, found ${calls.length}`);
    const offenders: Array<{ key: string; pos: number }> = [];
    for (const call of calls) {
      // recordSkip(reason, agentName, message, context, { agentId, coinId })
      // context is the 4th arg.
      assert.ok(call.args.length === 5, `recordSkip must take 5 args, got ${call.args.length} at pos ${call.pos}`);
      const ctx = call.args[3].trim();
      assert.ok(ctx.startsWith("{") && ctx.endsWith("}"), "recordSkip context must be an object literal");
      const { keys, spreads } = extractObjectKeysRecursive(ctx.slice(1, -1));
      for (const k of keys) {
        if (!ALLOWED_KEYS.has(k)) offenders.push({ key: k, pos: call.pos });
      }
      for (const s of spreads) {
        if (!ALLOWED_SPREAD_SOURCES.has(s)) {
          offenders.push({ key: `...${s}`, pos: call.pos });
        }
      }
    }
    assert.deepEqual(offenders, [],
      `recordSkip context contains non-allowlisted keys:\n${offenders.map((o) => `  - ${o.key} (pos ${o.pos})`).join("\n")}`,
    );
  });

  test("getMlDecision request payload only contains allowlisted keys (incl. nested portfolio + openPositions)", () => {
    const calls = findCallArgs(cleanedTrader, /\bgetMlDecision\b/);
    assert.ok(calls.length >= 1, `expected ≥1 getMlDecision call, got ${calls.length}`);
    const offenders: string[] = [];
    for (const call of calls) {
      assert.ok(call.args.length === 1, `getMlDecision takes 1 arg, got ${call.args.length}`);
      const obj = call.args[0].trim();
      assert.ok(obj.startsWith("{") && obj.endsWith("}"), "getMlDecision arg must be an object literal");
      const { keys, spreads } = extractObjectKeysRecursive(obj.slice(1, -1));
      for (const k of keys) {
        if (!ALLOWED_KEYS.has(k)) offenders.push(k);
      }
      for (const s of spreads) {
        if (!ALLOWED_SPREAD_SOURCES.has(s)) offenders.push(`...${s}`);
      }
    }
    assert.deepEqual(offenders, [],
      `getMlDecision request contains non-allowlisted keys: ${offenders.join(", ")}`,
    );
  });

  test("paper_trades INSERT row only contains allowlisted columns", () => {
    const bodies = findChainedInsertValues(cleanedTrader, "paperTradesTable");
    assert.ok(bodies.length >= 1, "expected at least one tx.insert(paperTradesTable).values(...) call");
    const offenders: string[] = [];
    for (const body of bodies) {
      const { keys, spreads } = extractObjectKeysRecursive(body);
      for (const k of keys) {
        if (!ALLOWED_KEYS.has(k)) offenders.push(k);
      }
      for (const s of spreads) {
        if (!ALLOWED_SPREAD_SOURCES.has(s)) offenders.push(`...${s}`);
      }
    }
    assert.deepEqual(offenders, [],
      `paper_trades INSERT contains non-allowlisted columns: ${offenders.join(", ")}`,
    );
  });

  test("paper_positions INSERT row only contains allowlisted columns", () => {
    const bodies = findChainedInsertValues(cleanedTrader, "paperPositionsTable");
    assert.ok(bodies.length >= 1, "expected at least one tx.insert(paperPositionsTable).values(...) call");
    const offenders: string[] = [];
    for (const body of bodies) {
      const { keys, spreads } = extractObjectKeysRecursive(body);
      for (const k of keys) {
        if (!ALLOWED_KEYS.has(k)) offenders.push(k);
      }
      for (const s of spreads) {
        if (!ALLOWED_SPREAD_SOURCES.has(s)) offenders.push(`...${s}`);
      }
    }
    assert.deepEqual(offenders, [],
      `paper_positions INSERT contains non-allowlisted columns: ${offenders.join(", ")}`,
    );
  });

  test("'Paper trade opened' structured log payload only contains allowlisted keys", () => {
    const body = findPaperTradeOpenedLog(PAPER_TRADER);
    const { keys, spreads } = extractObjectKeysRecursive(body);
    const offenders = keys.filter((k) => !ALLOWED_KEYS.has(k));
    for (const s of spreads) {
      if (!ALLOWED_SPREAD_SOURCES.has(s)) offenders.push(`...${s}`);
    }
    assert.deepEqual(offenders, [],
      `'Paper trade opened' log contains non-allowlisted keys: ${offenders.join(", ")}`,
    );
  });

  test("forbidden LLM-derived identifier pattern scan over paper-trader.ts", () => {
    const hits = scanForbiddenIdentifiers(cleanedTrader);
    assert.deepEqual(hits, [],
      `paper-trader.ts contains LLM-derived identifiers: ${hits.join(", ")}`,
    );
  });

  test("forbidden LLM-derived identifier pattern scan over quant-brain.ts", () => {
    const hits = scanForbiddenIdentifiers(cleanedBrain);
    assert.deepEqual(hits, [],
      `quant-brain.ts contains LLM-derived identifiers: ${hits.join(", ")}`,
    );
  });

  test("negative control: forbidden-identifier scanner fires on tainted source", () => {
    const tainted = `
      const x = { newsTag: 1, llmBias: 2, sentimentScore: 3, ai_signal: 4, geminiVote: 5, rawNewsScore: 6, chatgptCall: 7 };
      function llm_helper() {}
      const news_feed = [];
    `;
    const cleaned = stripCommentsAndStrings(tainted);
    const hits = scanForbiddenIdentifiers(cleaned);
    for (const expected of [
      "newsTag", "llmBias", "sentimentScore", "ai_signal",
      "geminiVote", "rawNewsScore", "chatgptCall",
      "llm_helper", "news_feed",
    ]) {
      assert.ok(hits.includes(expected), `scanner missed '${expected}' in tainted source (got: ${hits.join(", ")})`);
    }
  });

  test("negative control: scanner does NOT false-positive on safe identifiers", () => {
    const safe = `
      const obj = { agentName: "x", atrPct: 0.5, applicable: true, modelVersion: "v1", aprilFool: false };
      function aiCheck_isNotAReservedToken() {}  // not LLM, but contains 'ai' — must not fire
    `;
    const cleaned = stripCommentsAndStrings(safe);
    const hits = scanForbiddenIdentifiers(cleaned);
    // aiCheck_isNotAReservedToken matches `ai_*` only if rendered ai_check; we
    // wrote camelCase deliberately. None of the safe identifiers should fire.
    for (const safeIdent of ["agentName", "atrPct", "applicable", "modelVersion", "aprilFool"]) {
      assert.ok(!hits.includes(safeIdent), `scanner false-positived on safe identifier '${safeIdent}'`);
    }
  });

  test("negative control: object-key extractor catches a non-allowlisted key", () => {
    const tainted = `
      const cfg = {
        agentName: "x",
        coinName: "y",
        newsTag: 1,
        nested: { llmBias: 2, deeper: { sentimentScore: 3 } },
        list: [{ aiSignal: 4 }],
        ...portfolioCheck.breakdown,
        ...someOtherSource,
      };
    `;
    const cleaned = stripCommentsAndStrings(tainted);
    // Find the literal and extract its keys
    const openIdx = cleaned.indexOf("{");
    const { body } = readBalanced(cleaned, openIdx, "{", "}");
    const { keys, spreads } = extractObjectKeysRecursive(body);
    for (const expected of ["newsTag", "llmBias", "sentimentScore", "aiSignal"]) {
      assert.ok(keys.includes(expected), `extractor missed '${expected}' (got keys: ${keys.join(", ")})`);
    }
    assert.ok(spreads.includes("portfolioCheck.breakdown"), "extractor must report spread sources");
    assert.ok(spreads.includes("someOtherSource"), "extractor must report all spread sources");
    // And the safe keys must come through too
    for (const safeIdent of ["agentName", "coinName"]) {
      assert.ok(keys.includes(safeIdent), `extractor lost safe key '${safeIdent}'`);
    }
  });

  test("QuantPrediction return object in quant-brain.ts only emits allowlisted fields", () => {
    // Find the final `return { ... };` in getQuantPrediction — the largest
    // object-literal return, which carries probUp/probDown/...
    // Strategy: extract every top-level `return {` ... `};` in the cleaned
    // source and pick the one with the most keys (heuristic — the
    // QuantPrediction return is the biggest in the file).
    const re = /\breturn\s*\{/g;
    let m: RegExpExecArray | null;
    let bestKeys: string[] = [];
    let bestSpreads: string[] = [];
    while ((m = re.exec(cleanedBrain)) !== null) {
      const openIdx = m.index + m[0].length - 1;
      try {
        const { body } = readBalanced(cleanedBrain, openIdx, "{", "}");
        const sub = extractObjectKeysRecursive(body);
        if (sub.keys.length > bestKeys.length) {
          bestKeys = sub.keys;
          bestSpreads = sub.spreads;
        }
      } catch {
        // skip — could be a return of an arrow body etc.
      }
    }
    assert.ok(bestKeys.length >= 10,
      `expected QuantPrediction return to have ≥10 keys, got ${bestKeys.length}: ${bestKeys.join(", ")}`,
    );
    const offenders = bestKeys.filter((k) => !ALLOWED_KEYS.has(k));
    for (const s of bestSpreads) {
      if (!ALLOWED_SPREAD_SOURCES.has(s)) offenders.push(`...${s}`);
    }
    assert.deepEqual(offenders, [],
      `QuantPrediction return contains non-allowlisted keys: ${offenders.join(", ")}`,
    );
  });

  test("smoke: the production source files were actually loaded (guard against silent empty reads)", () => {
    assert.ok(PAPER_TRADER.length > 10_000, "paper-trader.ts looks suspiciously small");
    assert.ok(QUANT_BRAIN.length > 5_000, "quant-brain.ts looks suspiciously small");
    assert.ok(PORTFOLIO_CONSTRAINTS.length > 1_000, "portfolio-constraints.ts looks suspiciously small");
    assert.ok(PAPER_TRADER.includes("executePaperTrade"), "paper-trader.ts missing executePaperTrade");
    assert.ok(QUANT_BRAIN.includes("getQuantPrediction"), "quant-brain.ts missing getQuantPrediction");
  });
});

// =============================================================================
// Task #371 — extend the OPEN-path forbidden-field sweep to the CLOSE path.
//
// Task #369 covered every payload that flows from quant-brain through the
// trade-OPEN gates into paper_trades / paper_positions. The CLOSE path
// (`closeExpiredPositions` in paper-trader.ts + `writeTradeJournal` in
// journal-writer.ts) writes its own rows: the close-time UPDATE on
// paper_trades, the structured "Paper trade closed" /
// "Paper trade anomaly-cancelled" / trailing-stop logs, the
// `writeTradeJournal({...})` arg, and the underlying
// `db.insert(tradeJournalTable).values({...})` row. A regression that, e.g.,
// wires a sentiment score into the early-exit decision or stamps an LLM
// tag onto the journal row would slip past the open-path scan.
//
// Strategy mirrors Task #369: statically extract the object-literal keys
// at every close-side call site and assert each one against the same
// ALLOWED_KEYS set (extended above with execution-only fields like
// exitPriceRaw / exitFee / closeReason). The forbidden-prefix scanner is
// also run over journal-writer.ts.
//
// Wired into the parity workflow set automatically — this file is
// already invoked by the `decision-engine-parity` validation command, so
// every change runs the close-path sweep alongside the open-path one.
// =============================================================================
describe("Task #371 — no LLM-derived field can reach the CLOSE-path payloads", () => {
  const cleanedTrader = stripCommentsAndStrings(PAPER_TRADER);
  const cleanedJournal = stripCommentsAndStrings(JOURNAL_WRITER);

  // Helper: locate every `<receiver>.update(<table>).set({...})` invocation
  // and return the object-literal body. Mirrors findChainedInsertValues for
  // the UPDATE side of the close path.
  function findChainedUpdateSets(
    cleaned: string,
    tableName: string,
    receiverPattern: string,
  ): string[] {
    const out: string[] = [];
    const re = new RegExp(
      String.raw`\b${receiverPattern}\s*\.\s*update\s*\(\s*${tableName}\s*\)\s*\.\s*set\s*\(`,
      "g",
    );
    let m: RegExpExecArray | null;
    while ((m = re.exec(cleaned)) !== null) {
      const openIdx = m.index + m[0].length - 1;
      const { body } = readBalanced(cleaned, openIdx, "(", ")");
      const args = splitTopLevelArgs(body);
      assert.equal(args.length, 1, `${tableName}.set(...) must take exactly one object arg`);
      const a = args[0].trim();
      assert.ok(a.startsWith("{") && a.endsWith("}"), `${tableName}.set arg must be an object literal`);
      out.push(a.slice(1, -1));
    }
    return out;
  }

  // Helper: scan every structured-log payload anchored on a known message
  // string (e.g. "Paper trade closed"). The original source carries the
  // landmark text; we walk back to the nearest `logger.<level>(` and
  // re-clean the slice for balanced parsing — same shape as
  // `findPaperTradeOpenedLog` but reusable for the close-side messages.
  function findStructuredLogObject(rawSrc: string, anchorMsg: string, level: "info" | "warn"): string {
    const idx = rawSrc.indexOf(`"${anchorMsg}"`);
    assert.notEqual(idx, -1, `expected to find '${anchorMsg}' log message`);
    const head = rawSrc.slice(0, idx);
    const callIdx = head.lastIndexOf(`logger.${level}(`);
    assert.notEqual(callIdx, -1, `expected logger.${level}( before '${anchorMsg}'`);
    const fragment = stripCommentsAndStrings(rawSrc.slice(callIdx));
    const openIdx = fragment.indexOf("(");
    const { body } = readBalanced(fragment, openIdx, "(", ")");
    const args = splitTopLevelArgs(body);
    assert.equal(args.length, 2, `'${anchorMsg}' logger.${level} must take (obj, msg)`);
    const obj = args[0].trim();
    assert.ok(obj.startsWith("{") && obj.endsWith("}"), "first arg must be an object literal");
    return obj.slice(1, -1);
  }

  test("writeTradeJournal({...}) callsites in paper-trader.ts only contain allowlisted keys", () => {
    const calls = findCallArgs(cleanedTrader, /\bwriteTradeJournal\b/);
    // Two callsites in closeExpiredPositions today: one in the
    // anomaly-cancel branch, one in the standard close branch.
    assert.ok(calls.length >= 2, `expected ≥2 writeTradeJournal calls, found ${calls.length}`);
    const offenders: Array<{ key: string; pos: number }> = [];
    for (const call of calls) {
      assert.equal(call.args.length, 1, `writeTradeJournal takes 1 arg, got ${call.args.length} at pos ${call.pos}`);
      const obj = call.args[0].trim();
      assert.ok(obj.startsWith("{") && obj.endsWith("}"), "writeTradeJournal arg must be an object literal");
      const { keys, spreads } = extractObjectKeysRecursive(obj.slice(1, -1));
      for (const k of keys) {
        if (!ALLOWED_KEYS.has(k)) offenders.push({ key: k, pos: call.pos });
      }
      for (const s of spreads) {
        if (!ALLOWED_SPREAD_SOURCES.has(s)) offenders.push({ key: `...${s}`, pos: call.pos });
      }
    }
    assert.deepEqual(offenders, [],
      `writeTradeJournal arg contains non-allowlisted keys:\n${offenders.map((o) => `  - ${o.key} (pos ${o.pos})`).join("\n")}`,
    );
  });

  test("trade_journal INSERT row in journal-writer.ts only contains allowlisted columns", () => {
    // journal-writer.ts uses `db.insert(...).values(...)` (live writer +
    // backfill helper) — both must obey the same allowlist.
    const bodies = findChainedInsertValues(cleanedJournal, "tradeJournalTable", "db");
    assert.ok(bodies.length >= 1, "expected ≥1 db.insert(tradeJournalTable).values(...) call in journal-writer.ts");
    const offenders: string[] = [];
    for (const body of bodies) {
      const { keys, spreads } = extractObjectKeysRecursive(body);
      for (const k of keys) {
        if (!ALLOWED_KEYS.has(k)) offenders.push(k);
      }
      for (const s of spreads) {
        if (!ALLOWED_SPREAD_SOURCES.has(s)) offenders.push(`...${s}`);
      }
    }
    assert.deepEqual(offenders, [],
      `trade_journal INSERT contains non-allowlisted columns: ${offenders.join(", ")}`,
    );
  });

  test("close-path UPDATE patches on paper_trades only contain allowlisted columns (closePrice selection)", () => {
    // Both branches of closeExpiredPositions patch paper_trades inside a
    // `tx.update(paperTradesTable).set({...})` — the standard close
    // (exitPrice / pnl / status / closedAt) and the anomaly-cancel
    // (status / closedAt / pnl=0). Either is the place where a
    // regression could stamp an LLM-derived column onto the row.
    const bodies = findChainedUpdateSets(cleanedTrader, "paperTradesTable", "tx");
    assert.ok(bodies.length >= 2, `expected ≥2 tx.update(paperTradesTable).set(...) calls, found ${bodies.length}`);
    const offenders: string[] = [];
    for (const body of bodies) {
      const { keys, spreads } = extractObjectKeysRecursive(body);
      for (const k of keys) {
        if (!ALLOWED_KEYS.has(k)) offenders.push(k);
      }
      for (const s of spreads) {
        if (!ALLOWED_SPREAD_SOURCES.has(s)) offenders.push(`...${s}`);
      }
    }
    assert.deepEqual(offenders, [],
      `paper_trades close-path UPDATE contains non-allowlisted columns: ${offenders.join(", ")}`,
    );
  });

  test("'Paper trade closed' structured log payload only contains allowlisted keys", () => {
    const body = findStructuredLogObject(PAPER_TRADER, "Paper trade closed", "info");
    const { keys, spreads } = extractObjectKeysRecursive(body);
    const offenders = keys.filter((k) => !ALLOWED_KEYS.has(k));
    for (const s of spreads) {
      if (!ALLOWED_SPREAD_SOURCES.has(s)) offenders.push(`...${s}`);
    }
    assert.deepEqual(offenders, [],
      `'Paper trade closed' log contains non-allowlisted keys: ${offenders.join(", ")}`,
    );
  });

  test("'Paper trade anomaly-cancelled' structured log payload only contains allowlisted keys", () => {
    // The full message starts with "Paper trade anomaly-cancelled" — match
    // on that prefix substring so trivial wording tweaks don't break the
    // test, but the landmark is still unique in the file.
    const idx = PAPER_TRADER.indexOf("Paper trade anomaly-cancelled");
    assert.notEqual(idx, -1, "expected to find 'Paper trade anomaly-cancelled' log message");
    // Walk back to the closest logger.warn(.
    const head = PAPER_TRADER.slice(0, idx);
    const callIdx = head.lastIndexOf("logger.warn(");
    assert.notEqual(callIdx, -1, "expected logger.warn( before 'Paper trade anomaly-cancelled'");
    const fragment = stripCommentsAndStrings(PAPER_TRADER.slice(callIdx));
    const openIdx = fragment.indexOf("(");
    const { body: callBody } = readBalanced(fragment, openIdx, "(", ")");
    const args = splitTopLevelArgs(callBody);
    assert.equal(args.length, 2, "anomaly-cancelled logger.warn must take (obj, msg)");
    const obj = args[0].trim();
    assert.ok(obj.startsWith("{") && obj.endsWith("}"), "first arg must be an object literal");
    const { keys, spreads } = extractObjectKeysRecursive(obj.slice(1, -1));
    const offenders = keys.filter((k) => !ALLOWED_KEYS.has(k));
    for (const s of spreads) {
      if (!ALLOWED_SPREAD_SOURCES.has(s)) offenders.push(`...${s}`);
    }
    assert.deepEqual(offenders, [],
      `'Paper trade anomaly-cancelled' log contains non-allowlisted keys: ${offenders.join(", ")}`,
    );
  });

  test("trailing-stop 'Profitable position extended' log payload (close-path branch) only contains allowlisted keys", () => {
    const body = findStructuredLogObject(
      PAPER_TRADER,
      "Profitable position extended with trailing stop (peak-based)",
      "info",
    );
    const { keys, spreads } = extractObjectKeysRecursive(body);
    const offenders = keys.filter((k) => !ALLOWED_KEYS.has(k));
    for (const s of spreads) {
      if (!ALLOWED_SPREAD_SOURCES.has(s)) offenders.push(`...${s}`);
    }
    assert.deepEqual(offenders, [],
      `'Profitable position extended' log contains non-allowlisted keys: ${offenders.join(", ")}`,
    );
  });

  test("forbidden LLM-derived identifier pattern scan over journal-writer.ts", () => {
    const hits = scanForbiddenIdentifiers(cleanedJournal);
    assert.deepEqual(hits, [],
      `journal-writer.ts contains LLM-derived identifiers: ${hits.join(", ")}`,
    );
  });

  test("smoke: journal-writer.ts was actually loaded and exports the close-path writer", () => {
    assert.ok(JOURNAL_WRITER.length > 2_000, "journal-writer.ts looks suspiciously small");
    assert.ok(JOURNAL_WRITER.includes("writeTradeJournal"), "journal-writer.ts missing writeTradeJournal");
    assert.ok(JOURNAL_WRITER.includes("tradeJournalTable"), "journal-writer.ts missing tradeJournalTable insert");
    assert.ok(PAPER_TRADER.includes("closeExpiredPositions"), "paper-trader.ts missing closeExpiredPositions");
  });

  test("negative control: close-path extractor catches a non-allowlisted key planted in a writeTradeJournal-shaped literal", () => {
    // Synthetic source that mimics a tainted writeTradeJournal call. Without
    // this control, a buggy extractor or allowlist could silently approve
    // every real fixture.
    const tainted = `
      void writeTradeJournal({
        tradeId: 1,
        exitReason: "stop-loss",
        sentimentScore: 0.7,
        nested: { llmBias: -1 },
      });
    `;
    const cleaned = stripCommentsAndStrings(tainted);
    const calls = findCallArgs(cleaned, /\bwriteTradeJournal\b/);
    assert.equal(calls.length, 1, "extractor must find the synthetic writeTradeJournal call");
    const obj = calls[0].args[0].trim();
    const { keys } = extractObjectKeysRecursive(obj.slice(1, -1));
    const offenders = keys.filter((k) => !ALLOWED_KEYS.has(k));
    for (const expected of ["sentimentScore", "llmBias"]) {
      assert.ok(offenders.includes(expected), `extractor missed planted offender '${expected}' (got: ${offenders.join(", ")})`);
    }
    // And the safe keys must NOT be flagged.
    for (const safe of ["tradeId", "exitReason"]) {
      assert.ok(!offenders.includes(safe), `extractor false-positived on safe key '${safe}'`);
    }
  });
});
