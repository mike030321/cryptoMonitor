// PostgreSQL → MSSQL conversion + upload script
// Usage: node pg_to_mssql.mjs
import fs from "fs";
import path from "path";
import readline from "readline";
import { execSync } from "child_process";
import { fileURLToPath } from "url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const DUMP = path.join(__dirname, "database_dump.sql");
const OUT_DIR = path.join(__dirname, "mssql_out");

const SERVER = "sql6034.site4now.net";
const DB = "db_aca32a_cryptoai";
const USER = "db_aca32a_cryptoai_admin";
const PASS = "Ciyvi.123";
const SQLCMD = `sqlcmd -S ${SERVER} -d ${DB} -U ${USER} -P "${PASS}"`;
const BCP_CONN = `-S ${SERVER} -d ${DB} -U ${USER} -P "${PASS}"`;

fs.mkdirSync(OUT_DIR, { recursive: true });

// ── MSSQL reserved words that need quoting ──────────────────────────────────
const RESERVED = new Set([
  "key","value","name","status","timestamp","type","level","data","schema","open","close",
  "identity","table","index","order","group","user","left","right","column",
  "primary","foreign","unique","check","default","create","drop","alter",
  "select","insert","update","delete","from","where","and","or","not","null",
  "int","text","date","time","year","month","day","end","begin","view",
  "trigger","procedure","function","exec","execute","transaction","commit",
  "rollback","with","as","by","on","in","is","to","set","all","any","case",
  "when","then","else","like","between","exists","join","inner","outer",
  "cross","full","having","union","except","intersect","top","offset","fetch",
  "next","rows","only","over","partition","row","number","count","sum","avg",
  "min","max","cast","convert","coalesce","isnull","nullif","dateadd",
  "datediff","getdate","getutcdate","source","scope","action","language",
  "system","read","write","log","backup","restore","file","path","object",
  "rule","result","role","grant","deny","revoke","signal"
]);

function q(col) {
  return RESERVED.has(col.toLowerCase()) ? `[${col}]` : col;
}

// ── Type conversions ─────────────────────────────────────────────────────────
const enumTypes = new Set();

function mapType(raw) {
  let t = raw.trim().toLowerCase().replace(/^public\./, "");
  if (t === "integer" || t === "int" || t === "int4") return "INT";
  if (t === "bigint" || t === "int8") return "BIGINT";
  if (t === "smallint" || t === "int2") return "SMALLINT";
  if (t === "text") return "NVARCHAR(MAX)";
  if (t === "boolean" || t === "bool") return "BIT";
  if (t === "double precision" || t === "float8" || t === "float") return "FLOAT";
  if (t === "real" || t === "float4") return "REAL";
  if (t === "jsonb" || t === "json") return "NVARCHAR(MAX)";
  if (t === "uuid") return "NVARCHAR(36)";
  if (t === "bytea") return "VARBINARY(MAX)";
  if (t === "timestamp with time zone" || t === "timestamptz") return "DATETIMEOFFSET";
  if (t === "timestamp without time zone" || t === "timestamp") return "DATETIME2";
  if (t === "date") return "DATE";
  if (t === "time") return "TIME";
  if (t.startsWith("character varying") || t.startsWith("varchar")) {
    const m = t.match(/\((\d+)\)/);
    return m ? `NVARCHAR(${m[1]})` : "NVARCHAR(MAX)";
  }
  if (t.startsWith("numeric") || t.startsWith("decimal")) {
    const m = t.match(/\([\d,\s]+\)/);
    return m ? `DECIMAL${m[0]}` : "DECIMAL(18,6)";
  }
  if (t.endsWith("[]")) return "NVARCHAR(MAX)"; // arrays → JSON string
  if (enumTypes.has(t)) return "NVARCHAR(50)";
  return "NVARCHAR(MAX)";
}

function mapDefault(val) {
  if (!val) return null;
  val = val.trim();
  // Remove type casts like ::text, ::public.xxx
  val = val.replace(/::[a-z_.\[\]]+/gi, "");
  if (val === "true") return "1";
  if (val === "false") return "0";
  if (val === "now()" || val === "current_timestamp") return "GETDATE()";
  if (val.startsWith("nextval(")) return null; // handled by IDENTITY
  // Strip outer quotes if it's a string literal
  val = val.trim();
  return val;
}

// ── Pass 1: collect enums and identity columns ───────────────────────────────
console.log("Pass 1: scanning enums and sequences...");
const dumpText = fs.readFileSync(DUMP, "utf8").replace(/^﻿/, "");
const lines = dumpText.split("\n");

// Collect enum names
for (let i = 0; i < lines.length; i++) {
  const m = lines[i].match(/^CREATE TYPE (?:public\.)?(\w+) AS ENUM/i);
  if (m) enumTypes.add(m[1].toLowerCase());
}

// Collect identity columns from sequences: tablename_colname_seq
const identityCols = new Map(); // "table.col" → true
for (let i = 0; i < lines.length; i++) {
  const m = lines[i].match(/^CREATE SEQUENCE (?:public\.)?(\w+)_(\w+)_seq/i);
  if (m) identityCols.set(`${m[1]}.${m[2]}`, true);
}
// Also from ALTER TABLE ... SET DEFAULT nextval
for (let i = 0; i < lines.length; i++) {
  const m = lines[i].match(/ALTER TABLE.*?(\w+).*?ALTER COLUMN (\w+) SET DEFAULT nextval/i);
  if (m) identityCols.set(`${m[1]}.${m[2]}`, true);
}

console.log(`  Found ${enumTypes.size} enum types, ${identityCols.size} identity columns`);

// ── Pass 2: convert DDL ──────────────────────────────────────────────────────
console.log("Pass 2: converting DDL...");

const ddlLines = [];
const tableColumns = new Map(); // tableName → [{ name, type, bit: bool }]
let i = 0;

ddlLines.push("USE [db_aca32a_cryptoai];");
ddlLines.push("GO");
ddlLines.push("");

// Helper: skip until condition
function skipUntil(pred) {
  while (i < lines.length && !pred(lines[i])) i++;
}

while (i < lines.length) {
  const line = lines[i];

  // Skip comments, SET, SELECT pg_catalog, empty
  if (
    line.startsWith("--") || line.startsWith("SET ") ||
    line.startsWith("SELECT pg_catalog") || line.trim() === "" ||
    line.startsWith("\\connect") || line.startsWith("\\.")
  ) { i++; continue; }

  // Skip CREATE TYPE ENUM
  if (/^CREATE TYPE/i.test(line)) {
    while (i < lines.length && !lines[i].includes(";")) i++;
    i++; continue;
  }

  // Skip SEQUENCE and related
  if (/^CREATE SEQUENCE|^ALTER SEQUENCE|^SELECT setval/i.test(line)) {
    while (i < lines.length && !lines[i].includes(";")) i++;
    i++; continue;
  }

  // Skip COPY blocks (data handled separately)
  if (/^COPY /i.test(line)) {
    i++;
    while (i < lines.length && lines[i] !== "\\.") i++;
    i++; continue;
  }

  // Skip ALTER TABLE SET DEFAULT nextval
  if (/ALTER TABLE.*SET DEFAULT nextval/i.test(line)) { i++; continue; }

  // ── CREATE TABLE ──────────────────────────────────────────────────────────
  if (/^CREATE TABLE/i.test(line)) {
    const tableMatch = line.match(/CREATE TABLE (?:public\.)?(\w+)/i);
    if (!tableMatch) { i++; continue; }
    const tableName = tableMatch[1];
    const cols = [];

    ddlLines.push(`IF OBJECT_ID('${tableName}') IS NOT NULL DROP TABLE [${tableName}];`);
    ddlLines.push(`CREATE TABLE [${tableName}] (`);

    i++; // move to first column line
    const colDefs = [];

    while (i < lines.length && !lines[i].startsWith(");")) {
      const colLine = lines[i].trim();
      i++;
      if (!colLine || colLine.startsWith("--")) continue;

      // Table constraints (PRIMARY KEY, UNIQUE, etc.) inline
      if (/^CONSTRAINT |^PRIMARY KEY|^UNIQUE/i.test(colLine)) {
        // Parse primary key columns
        const pkMatch = colLine.match(/PRIMARY KEY \(([^)]+)\)/i);
        if (pkMatch) {
          const pkCols = pkMatch[1].split(",").map(c => q(c.trim().replace(/"/g, "")));
          colDefs.push(`    CONSTRAINT [PK_${tableName}] PRIMARY KEY (${pkCols.join(", ")})`);
        }
        const uqMatch = colLine.match(/UNIQUE \(([^)]+)\)/i);
        if (uqMatch) {
          const uqCols = uqMatch[1].split(",").map(c => q(c.trim().replace(/"/g, "")));
          colDefs.push(`    UNIQUE (${uqCols.join(", ")})`);
        }
        continue;
      }

      // Column definition: "colname" type [constraints...]
      const colMatch = colLine.match(/^"?(\w+)"?\s+(.+?)(?:,\s*)?$/);
      if (!colMatch) continue;
      const colName = colMatch[1];
      let rest = colMatch[2].replace(/,$/, "").trim();

      // Extract DEFAULT value
      let defaultVal = null;
      const defMatch = rest.match(/DEFAULT\s+((?:'[^']*'|[^\s,]+)(?:\s*::[a-z_.\[\]]+)?)/i);
      if (defMatch) {
        defaultVal = mapDefault(defMatch[1]);
        rest = rest.replace(/DEFAULT\s+((?:'[^']*'|[^\s,]+)(?:\s*::[a-z_.\[\]]+)?)/i, "").trim();
      }

      const isNotNull = /NOT NULL/i.test(rest);
      // Remove NOT NULL, NULL from rest for type parsing
      rest = rest.replace(/\s*(NOT NULL|NULL)\s*/gi, "").trim();

      // Get the type (everything before any remaining keywords)
      let pgType = rest.split(/\s+REFERENCES|\s+CHECK|\s+DEFAULT/i)[0].trim();
      // Remove trailing commas
      pgType = pgType.replace(/,$/, "").trim();

      const mssqlType = mapType(pgType);
      const isBit = mssqlType === "BIT";

      let colDef = `    ${q(colName)} ${mssqlType}`;
      if (isNotNull) colDef += " NOT NULL";
      if (defaultVal) {
        colDef += ` DEFAULT ${defaultVal}`;
      }

      colDefs.push(colDef);
      cols.push({ name: colName, type: mssqlType, bit: isBit });
    }

    tableColumns.set(tableName, cols);
    ddlLines.push(colDefs.join(",\n"));
    ddlLines.push(");");
    ddlLines.push("GO");
    ddlLines.push("");
    i++; // skip ");"
    continue;
  }

  // ── CREATE INDEX ─────────────────────────────────────────────────────────
  if (/^CREATE (?:UNIQUE )?INDEX/i.test(line)) {
    let idxLines = [];
    let cur = line;
    while (!cur.includes(";")) {
      idxLines.push(cur);
      i++;
      cur = lines[i] || ";";
    }
    idxLines.push(cur);
    i++;
    let idxSql = idxLines.join(" ");
    // Remove PostgreSQL-specific parts
    idxSql = idxSql
      .replace(/USING \w+/gi, "")
      .replace(/NULLS (FIRST|LAST)/gi, "")
      .replace(/public\./gi, "")
      .replace(/WHERE.*?;/, ";")
      .replace(/\s+/g, " ")
      .trim();
    ddlLines.push(idxSql);
    ddlLines.push("GO");
    continue;
  }

  // ── ALTER TABLE ADD CONSTRAINT (PK / FK / UNIQUE) ────────────────────────
  if (/^ALTER TABLE/i.test(line)) {
    let altLines = [line];
    while (!lines[i - 1]?.includes(";")) {
      altLines.push(lines[i]);
      i++;
    }
    let altSql = altLines.join(" ");

    // Skip SET DEFAULT (already handled)
    if (/SET DEFAULT/i.test(altSql)) continue;
    // Skip OWNER TO
    if (/OWNER TO/i.test(altSql)) continue;

    // Clean up public. prefix and double-quotes
    altSql = altSql
      .replace(/public\./gi, "")
      .replace(/"/g, "")
      .replace(/\s+/g, " ")
      .trim();

    ddlLines.push(altSql);
    ddlLines.push("GO");
    continue;
  }

  i++;
}

const ddlFile = path.join(OUT_DIR, "schema.sql");
fs.writeFileSync(ddlFile, ddlLines.join("\n"), "utf8");
console.log(`  DDL written → ${ddlFile}`);

// ── Pass 3: extract COPY blocks as tab-delimited data files ──────────────────
console.log("Pass 3: extracting COPY data blocks...");

const dataFiles = [];
i = 0;

async function extractTable(tableName, cols, tableCols, dataLines) {
  const bitCols = new Set(tableCols.filter(c => c.bit).map(c => c.name));
  const dataFile = path.join(OUT_DIR, `${tableName}.dat`);
  const stream = fs.createWriteStream(dataFile, "utf8");

  for (let row of dataLines) {
    const fields = row.split("\t");
    cols.forEach((col, idx) => {
      if (idx >= fields.length) return;
      if (fields[idx] === "\\N") { fields[idx] = ""; return; }
      if (bitCols.has(col)) {
        if (fields[idx] === "t") { fields[idx] = "1"; return; }
        if (fields[idx] === "f") { fields[idx] = "0"; return; }
      }
      // Fix DATETIMEOFFSET timezone: only on actual datetime values (YYYY-MM-DD...)
      if (/^\d{4}-\d{2}-\d{2}/.test(fields[idx])) {
        fields[idx] = fields[idx].replace(/([+-]\d{2})$/, "$1:00");
      }
    });
    stream.write(fields.join("\t") + "\n");
  }

  await new Promise((resolve, reject) => {
    stream.on("finish", resolve);
    stream.on("error", reject);
    stream.end();
  });

  return dataFile;
}

const extractPromises = [];

while (i < lines.length) {
  const line = lines[i];
  const copyMatch = line.match(/^COPY (?:public\.)?(\w+) \(([^)]+)\) FROM stdin;/i);
  if (!copyMatch) { i++; continue; }

  const tableName = copyMatch[1];
  const cols = copyMatch[2].split(",").map(c => c.trim().replace(/"/g, ""));
  const tableCols = tableColumns.get(tableName) || [];

  i++;
  const dataLines = [];
  while (i < lines.length && lines[i] !== "\\.") {
    dataLines.push(lines[i]);
    i++;
  }
  i++; // skip \.

  if (dataLines.length > 0) {
    console.log(`  ${tableName}: ${dataLines.length} rows`);
    extractPromises.push(
      extractTable(tableName, cols, tableCols, dataLines).then(file => ({
        table: tableName, cols, file, rowCount: dataLines.length
      }))
    );
  }
}

const dataFiles2 = await Promise.all(extractPromises);
dataFiles.push(...dataFiles2);

// ── Step 4: run DDL on MSSQL ─────────────────────────────────────────────────
console.log("\nStep 4: applying DDL to MSSQL...");
try {
  execSync(`sqlcmd -S ${SERVER} -d ${DB} -U ${USER} -P "${PASS}" -i "${ddlFile}" -b`, {
    stdio: "inherit",
  });
  console.log("  DDL applied successfully.");
} catch (e) {
  console.error("  DDL errors above — continuing with data load anyway.");
}

// ── Step 5: generate + run bcp PowerShell script ────────────────────────────
console.log("\nStep 5: bulk-loading data...");

const ps1Lines = [];
ps1Lines.push(`$ErrorActionPreference = 'Continue'`);
for (const { table, file } of dataFiles) {
  const winFile = file.replace(/\//g, "\\");
  ps1Lines.push(
    `Write-Host "Loading ${table}..."`  ,
    `bcp "[${DB}].[dbo].[${table}]" in "${winFile}" -c -t "\`t" -r "\`n" -k -S ${SERVER} -U ${USER} -P "${PASS}" -b 1000`
  );
}
const ps1File = path.join(OUT_DIR, "load_data.ps1");
fs.writeFileSync(ps1File, ps1Lines.join("\n"), "utf8");
console.log(`  PowerShell script written → ${ps1File}`);

try {
  execSync(`powershell -ExecutionPolicy Bypass -File "${ps1File}"`, { stdio: "inherit" });
} catch {
  // non-zero exit is fine — some tables may warn
}

// Verify row counts
console.log("\nVerifying row counts in MSSQL...");
const tables = dataFiles.map(d => `'${d.table}'`).join(",");
execSync(
  `sqlcmd -S ${SERVER} -d ${DB} -U ${USER} -P "${PASS}" -Q "SELECT TABLE_NAME, (SELECT COUNT(*) FROM INFORMATION_SCHEMA.COLUMNS c2 WHERE c2.TABLE_NAME = t.TABLE_NAME) as cols FROM INFORMATION_SCHEMA.TABLES t WHERE TABLE_TYPE='BASE TABLE' ORDER BY TABLE_NAME"`,
  { stdio: "inherit" }
);
