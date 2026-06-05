# Crypto AI Monitor - Full System Specification

## Executive Summary

A real-time cryptocurrency monitoring platform powered by 10 autonomous AI agents that analyze, predict, and paper-trade 10 high-risk altcoins. The system uses multi-model AI consensus (GPT-4o-mini + Gemini 2.5 Flash), multi-timeframe analysis, persistent per-agent pattern learning, and realistic fee simulation to train agents for eventual real-exchange trading.

**Total Paper Capital:** $10,000 ($1,000 per agent)
**Tracked Coins:** PEPE, FLOKI, BONK, WIF, RNDR, INJ, SEI, TIA, JUP, WLD
**Analysis Timeframes:** 5-minute, 1-hour, 2-hour, 6-hour, 1-day

---

## Architecture Overview

```
+------------------+     +------------------+     +------------------+
|   React Frontend |<--->|  Express API     |<--->|  PostgreSQL DB   |
|  (Dashboard UI)  |     |  (Brain + Engine)|     |  (All State)     |
+------------------+     +------------------+     +------------------+
                               |       |
                     +---------+---------+
                     |                   |
              +------+------+    +-------+------+
              | GPT-4o-mini |    | Gemini 2.5   |
              | (OpenAI)    |    | Flash (Google)|
              +-------------+    +--------------+
                     |                   |
              +------+-------------------+------+
              |  Multi-Model Consensus Engine   |
              +---------------------------------+
```

**Stack:** React + Vite (frontend), Express + Node.js (backend), PostgreSQL (database), Drizzle ORM, TanStack Query, Recharts, Shadcn UI, Tailwind CSS

---

## The 10 AI Agents

Each agent is a distinct "personality" with a specialized trading philosophy. They see the same market data but interpret it through different lenses.

| # | Agent Name | Specialty | Focus Indicators |
|---|-----------|-----------|-----------------|
| 1 | **Momentum Max** | Trend-following | MACD histograms, EMA21 alignment |
| 2 | **Contrarian Clara** | Mean-reversion | Bollinger Bands, RSI extremes |
| 3 | **Volume Victor** | Volume analysis | Price-volume divergence, volume spikes |
| 4 | **Pattern Pete** | Chart patterns | Support/resistance, classic formations |
| 5 | **Sentiment Sarah** | Market psychology | Cross-asset correlations, fear/greed |
| 6 | **Trend Tracker Tom** | Multi-EMA alignment | EMA9/EMA21 crossovers |
| 7 | **Breakout Bella** | Volatility breakouts | Bollinger Band squeezes |
| 8 | **Scalper Steve** | Short-term momentum | 5m/1h high-probability small wins |
| 9 | **Whale Watcher Wendy** | Large player detection | Volume/market-cap ratios |
| 10 | **Divergence Dan** | Hidden reversals | RSI/MACD divergences |

**Agent States:**
- **Active** - Normal operation
- **Degraded** - Score dropped below threshold, reduced trading
- **Resting** - Temporarily benched due to poor performance

---

## Full Pipeline: One Analysis Cycle

Every **60 seconds**, the system runs a complete analysis cycle. Here is the exact sequence:

### Stage 1: Data Ingestion (seconds 0-2)

```
1. Fetch live prices for all 10 coins (CoinGecko/CoinMarketCap APIs)
2. Stale data check (skip if data > 2 minutes old)
3. Save prices to price_history table (time-series)
4. Immediately resolve any pending predictions before new analysis
```

### Stage 2: Market Context (seconds 2-5)

```
5. Compute global market sentiment across all coins
6. Fetch Fear & Greed Index (real-time crypto sentiment gauge)
7. Fetch BTC Dominance (Bitcoin's market share %)
8. Generate real market news/events from CoinGecko trending data
9. Classify market regime: bull / bear / sideways / volatile
10. Run contagion detection (is one coin's crash spreading to others?)
```

### Stage 3: Coin & Timeframe Selection (seconds 5-8)

```
11. Select timeframes for this cycle based on rotation schedule:
    - 2h: every cycle (primary timeframe, highest win rate)
    - 6h: every 2nd cycle
    - 1h: every 3rd cycle
    - 5m: every 4th cycle
    - 1d: every 6th cycle

12. Select top 6 coins by signal strength:
    - RSI at extremes (oversold < 30, overbought > 70)
    - MACD crossover strength
    - Bollinger Band position
    - Volume anomalies
    - Recent price momentum
```

### Stage 4: Per-Agent AI Analysis (seconds 8-50)

For EACH of the 10 agents, for EACH assigned coin/timeframe:

```
13. Get agent specialization (which coins is this agent best at?)
14. Allocate coins to agent (prioritize high-affinity coins + exploration slots)

For each coin assigned to this agent:

15. Calculate technical indicators:
    - RSI (14-period)
    - MACD (12/26/9)
    - Bollinger Bands (20-period, 2 std dev)
    - EMA 9 and EMA 21
    - ATR (Average True Range for volatility)

16. Generate composite fingerprint (unique hash of current market state)
    Example: "rsi:oversold|macd:bull-cross|bb:below-lower|vol:high"

17. Search for historical fingerprint matches:
    - Exact matches (identical pattern seen before)
    - Partial matches (1-2 bins different)
    - Return: historical win rate, avg price move, sample count

18. Fetch coin correlations (how does this coin move relative to others?)
19. Check contagion alerts (is a correlated coin crashing/pumping?)
20. Retrieve agent's past accuracy on this specific coin/timeframe
21. Get confidence calibration data (is this agent over/under-confident?)
22. Retrieve learned patterns from coin_insights database

23. BUILD THE AI PROMPT containing:
    - Agent's personality and trading philosophy
    - Current price, 24h change, volume, market cap
    - All technical indicators
    - Historical pattern matches + win rates
    - Market regime and sentiment
    - Agent's own track record
    - Calibration warnings (e.g., "You tend to be overconfident by 15%")
    - Strict confidence guidelines

24. CALL GPT-4o-mini with the prompt
25. CALL Gemini 2.5 Flash with the same prompt
26. Run multi-model consensus (see below)
27. Apply confidence calibration adjustments
28. Apply auto-correction (flip direction if historically wrong)
29. Save prediction to database
30. Execute paper trade if confidence >= 35%
```

### Stage 5: Post-Cycle (seconds 50-60)

```
31. Generate "Best Pick" — the single highest-conviction trade across all agents
32. Update monitoring state (cycle count, timestamps)
33. Compute coin correlations (updated periodically)
```

---

## Multi-Model Consensus Engine

Both GPT-4o-mini and Gemini 2.5 Flash analyze the same data independently. Their outputs are then merged:

### Agreement (Both models predict same direction)
```
- Weighted average of confidence scores
- Consensus boost: +5% to +10% confidence
- Combined reasoning from both models
- Use more aggressive model's take-profit target
```

### Disagreement (Models predict opposite directions)
```
- Winner chosen by dynamic model weights (per-coin performance history)
- Conflict penalty: -15% to -20% confidence
- Risk tightened: use the tighter stop-loss, narrower take-profit
- Both reasonings included with disagreement note
```

### Fallback (If both models fail / rate-limited)
```
- Pure technical rule-based prediction
- RSI extremes + MACD signal + Bollinger position
- Conservative confidence (never above 40%)
- 10-minute cooldown on 429 rate limit errors
```

**Rate Limits:** GPT = 100,000 tokens/day, Gemini = 14,000 tokens/day

---

## Paper Trading System

### Position Sizing

**Phase 1 (First 30 trades per agent — discovery phase):**
```
Position Size = Portfolio Value x 20% x (0.75 + confidence x 0.25)
```

**Phase 2 (After 30 trades — Kelly Criterion activates):**
```
Kelly % = (Win Rate x R - (1 - Win Rate)) / R
  where R = Average Win / Average Loss ratio

Final Size = Kelly% x 35% (third-Kelly) x Confidence Multiplier x Timeframe Multiplier
```

**Timeframe Multipliers (position sizing):**
| Timeframe | Kelly Multiplier |
|-----------|-----------------|
| 5 min | 0.7x (smallest) |
| 1 hour | 0.8x |
| 2 hour | 1.2x |
| 6 hour | 1.5x |
| 1 day | 1.7x (largest) |

**Hard Limits:**
- Maximum position: 30% of total portfolio
- Maximum cash usage: 80% of available cash
- Maximum open positions: 2 per agent
- Maximum portfolio at risk: 60%
- Minimum confidence to trade: 35%

**Design Rationale:** Position sizing is deliberately conservative during the discovery phase. The system must first prove edge exists before sizing up. These parameters can be gradually increased as statistical significance builds (target: 100+ decided predictions per agent with demonstrated positive expectancy).

### Fee Simulation (Realistic Exchange Costs)

| Fee Type | Rate |
|----------|------|
| Taker Fee (entry) | 0.10% |
| Maker Fee (exit) | 0.10% |
| Slippage | 0.05% |
| **Total round-trip cost** | **~0.25%** |

This means every trade starts approximately $2.50 in the hole per $1,000 position — agents must overcome this drag to be profitable.

### Stop-Loss & Take-Profit (Volatility-Based)

Levels are set using ATR (Average True Range) with per-timeframe multipliers:

| Timeframe | SL Multiplier | TP Multiplier | ATR Floor |
|-----------|--------------|---------------|-----------|
| 5 min | 2.5x ATR | 2.0x ATR | 0.4% |
| 1 hour | 3.0x ATR | 2.5x ATR | 0.8% |
| 2 hour | 2.8x ATR | 2.5x ATR | 1.0% |
| 6 hour | 2.5x ATR | 2.5x ATR | 1.5% |
| 1 day | 2.2x ATR | 2.5x ATR | 2.0% |

### Trade Resolution

Positions close when one of three events occurs:
1. **Stop-Loss Hit** - Price moves against position by SL distance
2. **Take-Profit Hit** - Price reaches target profit level
3. **Expiration** - Timeframe duration passes (e.g., 2 hours for 2h trades)

**Abnormal Price Protection:** If price moves 3x or drops 67%+ (data error / extreme crash), trade is cancelled and capital returned.

### Circuit Breakers

| Protection | Rule |
|-----------|------|
| 3-Loss Streak | If an agent loses 3 consecutive trades on a specific coin, that coin is blocked until a win |
| Max Positions | Cannot open more than 3 positions simultaneously |
| Portfolio Risk Cap | Cannot have more than 95% of portfolio in active trades |
| Confidence Floor | No trades below 35% confidence |
| Stale Data Guard | Won't close positions if price data is stale |

---

## 7 Learning Systems (Feedback Loops)

The system has 7 interconnected learning mechanisms that improve predictions over time:

### 1. Pattern Analyzer (Feature Extraction)

Calculates RSI, MACD, Bollinger Bands, EMA, ATR for each coin. Generates a **composite fingerprint** — a unique hash of the current market state:

```
Example: "rsi:oversold|macd:bull-cross|bb:below-lower|vol:high|trend:down"
```

This fingerprint is the key used to index and retrieve historical outcomes.

### 2. Coin Insights (Experience Database)

Stores every resolved prediction with its technical context. When a similar market pattern appears again, the system retrieves:
- Historical win rate for this exact pattern
- Average price move after this pattern
- Sample count (how many times seen)
- Direction bias (which direction worked better)
- RSI zone accuracy (oversold/overbought success rates)
- MACD signal accuracy (bullish/bearish crossover hit rates)

This data is injected directly into the AI prompt: *"In the past, this exact pattern has a 65% win rate over 20 trades."*

### 3. Fingerprint Matching (Pattern Recognition)

Uses "binning" to group continuous indicator values into discrete categories:
- RSI 28.5 -> "oversold"
- MACD histogram 0.003 -> "weak-bull"

Then performs **distance matching** to find not just identical patterns, but "1-2 bins different" patterns — learning from similar (not just exact) conditions.

### 4. Agent Specialization (Dynamic Allocation)

Tracks which agents perform best on which coins. Calculates an **affinity score** based on:
- Win rate per coin
- PnL per coin
- Number of trades (sample size)

High-affinity coins are prioritized in the agent's allocation, while "exploration slots" let agents discover new opportunities.

### 5. Confidence Calibrator (Truth Filter)

Compares "Stated Confidence" vs. "Actual Win Rate" over time:
- Agent says 80% confident but only wins 50% -> **overconfidence bias detected**
- System warns the AI: "You tend to be overconfident by 20%. Reduce stated confidence."
- Post-prediction: mathematically adjusts raw confidence score based on historical reliability buckets
- AI confidence is capped at 75% maximum

### 6. Regime Detector (Macro Context)

Classifies the overall market into regimes:

| Regime | Characteristics | Trading Impact |
|--------|----------------|----------------|
| **Bull** | Sustained upward momentum | Wider TP, tighter SL |
| **Bear** | Sustained downward pressure | Tighter TP, wider SL |
| **Sideways** | Low volatility, range-bound | Lower confidence, tighter bands |
| **Volatile** | High swings, uncertain | Wider stops, lower conviction |

Uses BTC dominance, Fear & Greed index, and cross-coin volatility to classify.

Also tracks **temporal pattern chains** — sequences of fingerprints over time (e.g., "Pattern A followed by Pattern B often leads to outcome C").

### 7. Contagion Detector (Lead-Lag Analysis)

Monitors whether a crash or pump in one coin spreads to correlated coins:
- Tracks "significant movers" (coins with large % moves)
- Uses historical correlation data to predict if/when the move will "infect" related coins
- Generates alerts: *"PEPE moved +5%, 80% probability FLOKI follows within 2 hours"*
- These alerts are passed to agents as additional trading context

---

## Prediction Resolution & Scoring

### How Predictions Are Graded

When a prediction's timeframe expires:

| Outcome | Condition | Score Change |
|---------|-----------|-------------|
| **Correct** | Price moved in predicted direction beyond threshold | +1 to +3 points |
| **Wrong** | Price moved opposite to prediction | -1 to -3 points |
| **Neutral** | Price moved slightly in right direction but below threshold | 0 points |

**Neutral outcomes are excluded from accuracy calculations** — they don't dilute an agent's win rate.

### Agent Scoring System

- Starting score: 100
- Score changes based on prediction accuracy, confidence level, and streak multipliers
- **Hot Streak Bonus**: Consecutive correct predictions amplify score gains
- **Cold Streak Penalty**: Consecutive wrong predictions amplify score losses
- Below threshold: agent status changes to "degraded" or "resting"

---

## Database Schema (12 Tables)

| Table | Purpose | Key Fields |
|-------|---------|------------|
| `agents` | AI agent profiles & stats | name, personality, score, accuracy, streak, status |
| `predictions` | All predictions with outcomes | agent_id, coin, direction, confidence, outcome, fingerprint |
| `paper_portfolios` | Agent trading balances | cash_balance, total_value, pnl, win/loss counts |
| `paper_trades` | Individual trade executions | entry/exit price, pnl, fees, status |
| `paper_positions` | Active open positions | direction, entry_price, stop_loss, take_profit, expires_at |
| `coin_insights` | Learned patterns per coin | pattern_type, direction, outcome, rsi, macd, fingerprint, regime |
| `coin_correlations` | Inter-coin relationships | coin_a, coin_b, correlation coefficient, timeframe |
| `price_history` | Time-series price data | coin_id, price, timestamp |
| `fingerprint_buffers` | Buffered market state hashes | coin_id, timeframe, fingerprints (JSON array) |
| `monitoring_state` | Engine state tracking | is_running, cycle_count, last_cycle_at |
| `conversations` | Chat history | title, created_at |
| `messages` | Chat messages | role, content, conversation_id |

**Key Design:** `coin_insights` and `fingerprint_buffers` survive system resets — they represent accumulated learning that must persist. Only trade/portfolio data is cleared on reset.

---

## Dashboard UI (5 Pages)

### 1. Command Center (Main Dashboard)
- System status header with cycle count and monitoring toggle
- **Top Consensus Pick** — single best trade recommendation across all agents
- Key metrics: total predictions, overall accuracy, top agent, active monitoring
- **AI Budget Tracker** — real-time token usage for GPT and Gemini (color-coded bars)
- Coin signal cards with 24h price changes
- Paper trading summary with net P&L

### 2. AI Agents
- Cards for each of the 10 agents showing score, accuracy, streak, status
- Mini win/loss record with hot streak indicators
- Click into **Agent Detail** for performance charts (accuracy + score over time)

### 3. Live Feed (Predictions)
- Chronological log of all AI predictions
- Outcome badges (Correct/Wrong/Pending)
- Agent reasoning displayed for each prediction
- Filterable by agent or coin

### 4. Analytics
- Time range selector (1h, 6h, 24h, 72h)
- Agent score comparison (bar chart)
- Prediction outcome distribution (pie chart)
- Agent accuracy ranking (horizontal bar chart)
- Detailed leaderboard table

### 5. Markets
- Grid of all 10 coin cards
- Price sparkline charts (1-hour history)
- Current price, 24h change %, volume, market cap

**Visual Style:** Dark "cyberpunk/terminal" aesthetic with emerald accents for gains, red for losses, amber for warnings. Scanline effects and ambient glow for the high-tech monitor feel.

---

## Data Flow Summary

```
Market Data (CoinGecko/CMC)
        |
        v
+---[Price Fetch & Storage]---+
        |
        v
+---[Technical Analysis]------+  --> Fingerprint Generated
        |                              |
        v                              v
+---[Regime Detection]--------+  +---[Fingerprint Match]---+
+---[Contagion Detection]-----+  +---[Coin Insights]-------+
+---[Sentiment Analysis]------+  +---[Agent Specialization]+
        |                              |
        v                              v
+---[Build AI Prompt with ALL context]-+
        |
        +------+-------+
        |              |
        v              v
   [GPT-4o-mini]  [Gemini 2.5]
        |              |
        v              v
+---[Multi-Model Consensus]---+
        |
        v
+---[Confidence Calibration]--+
        |
        v
+---[Auto-Correction]---------+
        |
        v
+---[Paper Trade Execution]---+
        |
        v
+---[Position Monitoring]-----+
   (every 15 seconds)
        |
        v
+---[Trade Resolution]--------+
        |
        v
+---[Update coin_insights]----+  --> Learning loop closes
+---[Update agent scores]-----+
+---[Update specializations]--+
+---[Update calibration]------+
```

---

## Key Metrics & Current Parameters

| Parameter | Value |
|-----------|-------|
| Analysis cycle interval | 60 seconds |
| Position check interval | 15 seconds |
| Prediction resolution interval | 15 seconds |
| Minimum trade confidence | 35% |
| Maximum AI confidence cap | 75% |
| Default position size | 20% of portfolio |
| Maximum position size | 30% of portfolio |
| Maximum open positions per agent | 2 |
| Maximum portfolio at risk | 60% |
| Kelly fraction (after 30 trades) | 35% (third-Kelly) |
| Round-trip fee drag | ~0.25% ($2.50 per $1,000) |
| GPT daily token limit | 100,000 |
| Gemini daily token limit | 14,000 |
| GPT 429 cooldown | 10 minutes |
| Circuit breaker threshold | 3 consecutive losses per coin |
| Agent starting capital | $1,000 each |
| Total system capital | $10,000 |

---

## Validation Framework

The system includes a built-in validation framework to rigorously evaluate whether agents have genuine trading edge vs. simple baselines.

### Baseline Comparisons

Every validation report automatically computes and compares the system against 4 baseline strategies:

| Baseline | Description |
|----------|------------|
| **Buy & Hold** | Equal-weight portfolio of all 10 coins, no trading |
| **RSI-Only** | Simple RSI 30/70 threshold strategy with same fee structure |
| **Random** | Coin-flip direction with same position sizing and fees |
| **Simple Momentum** | 10-period lookback momentum following |

The system must consistently beat ALL baselines to claim edge. The validation report shows `systemEdge` (accuracy difference vs. each baseline) and flags when the system is underperforming any baseline.

### Metrics Tracked

**Per-system:**
- Accuracy, calibration error (stated confidence vs. actual win rate)
- Profit factor (gross profit / gross loss)
- Best/worst trade, average return per trade
- Total P&L with fee drag included

**Per-agent (differentiation analysis):**
- Unique behavior score (are agents making different decisions?)
- Individual accuracy, win rate, P&L
- Warning if agents have <20% behavioral differentiation (acting as themed wrappers)

**Per-timeframe:**
- Accuracy and win rate by 5m, 1h, 2h, 6h, 1d
- Average return per timeframe to identify which horizons have edge

### Statistical Warnings

The report automatically flags:
- Insufficient sample size (<30 decided predictions or <20 closed trades)
- High calibration error (>15% gap between confidence and accuracy)
- High neutral rate (>50% of predictions expiring without clear outcome)
- Profit factor below 1.0 (net losing money)
- Low agent differentiation (agents behaving identically)

**API Endpoint:** `GET /api/crypto/validation-report?hours=24`

---

## Ablation Testing Framework

To prove which features actually contribute to performance, every major system component can be independently toggled on/off at runtime:

| Feature | Toggle | What Gets Disabled |
|---------|--------|-------------------|
| **Dual Consensus** | `dualConsensus` | Only uses single model (GPT or Gemini), no consensus merging |
| **Fingerprint Matching** | `fingerprintMatching` | No historical pattern lookup, no fingerprint distance matching |
| **Contagion Detection** | `contagionDetection` | No cross-coin contagion alerts passed to agents |
| **Confidence Calibration** | `confidenceCalibration` | Raw AI confidence used without historical adjustment |
| **Regime Detection** | `regimeDetection` | No market regime classification, no SL/TP/confidence regime adjustments |
| **Agent Specialization** | `agentSpecialization` | All agents get same coins (no affinity-based allocation) |

### How to Run Ablation Tests

1. Disable one feature at a time via API
2. Let the system run for 24-72 hours
3. Pull validation report comparing performance with vs. without the feature
4. Re-enable and move to next feature

```
# Disable consensus (test single-model vs. dual-model)
POST /api/crypto/ablation-config  { "dualConsensus": false }

# Check current config
GET /api/crypto/ablation-config

# Reset all features to enabled
POST /api/crypto/ablation-reset
```

### Expected Ablation Outcomes

Each feature should demonstrate measurable improvement over the "off" baseline. If a feature doesn't improve accuracy or P&L, it should be removed to reduce complexity, latency, and token cost.

**Priority ablation order:**
1. Dual consensus (most expensive in token cost)
2. Fingerprint matching (most complex learning mechanism)
3. Confidence calibration (directly affects position sizing)
4. Regime detection (affects SL/TP levels)
5. Contagion detection (least proven feature)
6. Agent specialization (may or may not add value)

---

## Long-Term Vision

### Phase 1: Discovery (Current)
Conservative paper trading with small position sizes (20% default). Primary goals:
- Accumulate 100+ decided predictions per agent for statistical significance
- Run full ablation tests to identify which features provide real value
- Compare system performance against all baselines over 7+ day windows
- Identify best-performing timeframes, coins, and agent personalities
- **Success Criteria:** System must beat Buy & Hold and Random baselines consistently over a 14-day rolling window

### Phase 2: Validation
If Phase 1 shows genuine edge:
- Remove features that don't improve performance (reduce complexity and token cost)
- Gradually increase position sizing for proven agents only (evidence-gated)
- Run walk-forward tests: train on week 1-2 data, validate on week 3
- Split-regime evaluation: confirm edge exists in bull, bear, AND sideways markets
- **Success Criteria:** Profit factor > 1.3, positive P&L after fees for 3+ consecutive weeks, edge survives across market regimes

### Phase 3: Scaled Paper Trading
If Phase 2 validates:
- Increase position sizes toward realistic exchange levels
- Add exchange-specific order book simulation (limit orders, partial fills)
- Test with higher capital ($10k+ per agent)
- **Success Criteria:** Sharpe ratio > 1.0, maximum drawdown < 15%, consistent positive weekly P&L for 4+ weeks

### Phase 4: Live Trading (Conditional)
Only if Phase 3 passes all criteria:
- Start with smallest possible real positions on a single exchange
- Run parallel paper/real tracking to confirm no execution gap
- Gradual capital increase with strict drawdown limits
- **Note:** This phase is conditional on demonstrating real, reproducible edge. The system is designed as an experimental research lab first, trading system second.

---

*Generated from live codebase analysis — April 2026*
