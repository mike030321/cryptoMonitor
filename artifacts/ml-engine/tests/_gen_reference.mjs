// One-shot script that runs the TypeScript pattern-analyzer's RSI/MACD/ATR
// (transliterated into JS) on a deterministic series and prints reference
// numbers used by tests/test_features.py to lock Python<->TS parity.
//
// Run: node artifacts/ml-engine/tests/_gen_reference.mjs
//
// The math here is copy-pasted line-for-line from
// artifacts/api-server/src/lib/pattern-analyzer.ts so that any future drift in
// the TS code is caught by re-running this script and updating the locked
// numbers in test_features.py.

function calculateEMAValues(prices, period) {
  if (prices.length === 0) return [];
  const k = 2 / (period + 1);
  const emaValues = [prices[0]];
  for (let i = 1; i < prices.length; i++) {
    emaValues.push(prices[i] * k + emaValues[i - 1] * (1 - k));
  }
  return emaValues;
}

function calculateRSI(prices, period = 14) {
  if (prices.length < period + 1) return 50;
  const changes = [];
  for (let i = 1; i < prices.length; i++) changes.push(prices[i] - prices[i - 1]);
  let avgGain = 0, avgLoss = 0;
  for (let i = 0; i < period; i++) {
    if (changes[i] > 0) avgGain += changes[i];
    else avgLoss += Math.abs(changes[i]);
  }
  avgGain /= period; avgLoss /= period;
  for (let i = period; i < changes.length; i++) {
    const gain = changes[i] > 0 ? changes[i] : 0;
    const loss = changes[i] < 0 ? Math.abs(changes[i]) : 0;
    avgGain = (avgGain * (period - 1) + gain) / period;
    avgLoss = (avgLoss * (period - 1) + loss) / period;
  }
  if (avgLoss === 0) return 100;
  const rs = avgGain / avgLoss;
  return 100 - 100 / (1 + rs);
}

function calculateMACD(prices) {
  if (prices.length < 26) return { macdLine: 0, signalLine: 0, histogram: 0 };
  const ema12 = calculateEMAValues(prices, 12);
  const ema26 = calculateEMAValues(prices, 26);
  const macdLine = ema12.map((v, i) => v - ema26[i]);
  const signalEma = calculateEMAValues(macdLine, 9);
  const cm = macdLine[macdLine.length - 1];
  const cs = signalEma[signalEma.length - 1];
  return { macdLine: cm, signalLine: cs, histogram: cm - cs };
}

function calculateATR(prices, period = 14) {
  if (prices.length < period + 1) return 0;
  const trs = [];
  for (let i = 1; i < prices.length; i++) {
    const prevClose = prices[i - 1];
    const currClose = prices[i];
    const estimatedHigh = Math.max(currClose, prevClose) * (1 + 0.001);
    const estimatedLow = Math.min(currClose, prevClose) * (1 - 0.001);
    const tr = Math.max(
      estimatedHigh - estimatedLow,
      Math.abs(estimatedHigh - prevClose),
      Math.abs(estimatedLow - prevClose),
    );
    trs.push(tr);
  }
  let atr = 0;
  for (let i = 0; i < period; i++) atr += trs[i];
  atr /= period;
  for (let i = period; i < trs.length; i++) {
    atr = (atr * (period - 1) + trs[i]) / period;
  }
  return atr;
}

// Same series the Python tests use.
const series = [];
for (let i = 0; i < 60; i++) {
  series.push(100 + i * 0.5 + Math.sin(i / 3) * 2);
}

const out = {
  rsi14: calculateRSI(series, 14),
  macd: calculateMACD(series),
  atr14: calculateATR(series, 14),
};
console.log(JSON.stringify(out, null, 2));
