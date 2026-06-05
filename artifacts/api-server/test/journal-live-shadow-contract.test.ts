import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const REPO = path.resolve(HERE, "..", "..", "..");

const journalWriter = readFileSync(
  path.join(REPO, "artifacts/api-server/src/lib/journal-writer.ts"),
  "utf8",
);

function functionBody(source: string, name: string): string {
  const start = source.indexOf(`export async function ${name}`);
  assert.ok(start >= 0, `${name} not found`);
  const bodyAnchor = name === "backfillJournals"
    ? source.indexOf("\n  let predictionsInserted = 0;", start)
    : -1;
  const open = bodyAnchor >= 0
    ? source.lastIndexOf("{", bodyAnchor)
    : source.indexOf("{", start);
  let depth = 0;
  for (let i = open; i < source.length; i++) {
    if (source[i] === "{") depth++;
    if (source[i] === "}") depth--;
    if (depth === 0) return source.slice(open, i + 1);
  }
  throw new Error(`${name} body was not balanced`);
}

test("trade journal attribution links only the live prediction_journal row", () => {
  const body = functionBody(journalWriter, "writeTradeJournal");
  assert.match(
    body,
    /eq\s*\(\s*predictionJournalTable\.predictionId\s*,\s*args\.predictionId\s*\)[\s\S]*eq\s*\(\s*predictionJournalTable\.shadow\s*,\s*false\s*\)/,
    "closed trades must not attach to shadow/challenger prediction rows",
  );
});

test("legacy trade backfill marks only live prediction_journal rows", () => {
  const body = functionBody(journalWriter, "backfillJournals");
  assert.match(
    body,
    /eq\s*\(\s*predictionJournalTable\.predictionId\s*,\s*t\.predictionId\s*\)[\s\S]*eq\s*\(\s*predictionJournalTable\.shadow\s*,\s*false\s*\)/,
    "backfilled trades must resolve predictionJournalId from the live row",
  );
  assert.match(
    body,
    /\.set\s*\(\s*\{\s*becameTrade:\s*true,\s*tradeId:\s*t\.id\s*\}\s*\)[\s\S]*eq\s*\(\s*predictionJournalTable\.shadow\s*,\s*false\s*\)/,
    "backfill must not mark shadow/challenger rows as live trades",
  );
});
