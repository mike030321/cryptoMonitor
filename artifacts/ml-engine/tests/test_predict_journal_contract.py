"""Task #471 — predict ↔ journal contract parity test.

Pin the field-level contract between three places that MUST stay in sync:

  1. `PredictResponse` — the pydantic schema returned by `POST /ml/predict`
     (artifacts/ml-engine/app/main.py)
  2. `MlPredictResponse` — the TypeScript interface the api-server's
     ml-client uses to read that response
     (artifacts/api-server/src/lib/ml-client.ts)
  3. `InsertPredictionJournalArgs` — the journal-writer's expected row
     shape; many of its fields are forwarded straight from the predict
     response (artifacts/api-server/src/lib/journal-writer.ts)

Task #460's root cause was a silent contract drift: `featureHash` was added
to `InsertPredictionJournalArgs` and required by the QUANT-row guard in
journal-writer, but `PredictResponse` did not carry one, so every
freshly-retrained 1d/6h LightGBM prediction was refused at the journal
boundary for months. Nothing in CI noticed because there was no
automated parity check between the Python schema, the TS interface, and
the journal-writer's expectations.

This test fails fast — with a clear, actionable message naming the
missing field on each side — when:

  * a field on `PredictResponse` is missing from `MlPredictResponse`
    (Node client can't read it);
  * a field on `MlPredictResponse` is missing from `PredictResponse`
    (Python service doesn't supply it, so journal would write nulls
    forever);
  * any (journal-arg, predict-field) pair in the canonical mapping is
    missing on either side;
  * `featureHash` specifically disappears from any of the three sides.

If you intentionally renamed or removed a field, update the canonical
mapping below in the SAME commit as the schema change so this test
keeps blocking accidental drift.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from app.main import PredictResponse


# --- locations of the TS source files we parse -------------------------
# tests/test_*.py → tests → ml-engine → artifacts → workspace root
_WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
_API_SERVER_LIB = _WORKSPACE_ROOT / "artifacts" / "api-server" / "src" / "lib"
_ML_CLIENT_TS = _API_SERVER_LIB / "ml-client.ts"
_JOURNAL_WRITER_TS = _API_SERVER_LIB / "journal-writer.ts"


# --- canonical mapping: journal-writer arg → /ml/predict field name ----
# Every entry must resolve to a real field on BOTH `PredictResponse`
# (Python) and `MlPredictResponse` (TypeScript). When a field name is
# the same on both sides — the common case — the key and value match.
# When the journal-writer renames a predict field (e.g. `regime` →
# `regimeLabel`), the value is the predict-side name and the key is the
# journal-side name.
JOURNAL_FROM_PREDICT_MAPPING: dict[str, str] = {
    "modelVersion": "modelVersion",
    "source": "source",
    # Task #460 — pinned explicitly below as well.
    "featureHash": "featureHash",
    "probUp": "probUp",
    "probDown": "probDown",
    "probStable": "probStable",
    "expectedReturnPct": "expectedReturnPct",
    "predictionStdPct": "predictionStdPct",
    # journal-writer renames `regime` → `regimeLabel` on the row.
    "regimeLabel": "regime",
    # journal-writer renames `specialists` → `specialistScores` on the row.
    "specialistScores": "specialists",
}


# Fields that legitimately exist on `PredictResponse` but are NOT
# exposed on `MlPredictResponse` because the Node client doesn't need
# them (purely for the /ml/report HTML view / model debugging). If you
# add a new "Python-only" field, list it here with a comment so the
# parity check still surfaces accidental omissions on the TS side.
_PYTHON_ONLY_PREDICT_FIELDS: set[str] = {
    # Top-5 LightGBM gain-importance pairs. Rendered by the Python
    # `/ml/report` endpoint; the api-server has no consumer for it.
    "featureImportanceTop5",
}

# Fields on `MlPredictResponse` (TS) that legitimately don't exist on
# `PredictResponse` (Python). Currently empty — keep it that way unless
# there's a documented reason the TS interface advertises a field the
# Python service can never supply.
_TS_ONLY_PREDICT_FIELDS: set[str] = set()


# --- TS interface field-name extractor ---------------------------------
def _strip_ts_comments(text: str) -> str:
    """Remove /* ... */ block comments and // line comments so the field
    regex doesn't misfire on commented-out names."""
    text = re.sub(r"/\*[\s\S]*?\*/", "", text)
    text = re.sub(r"//[^\n]*", "", text)
    return text


def _extract_interface_body(source: str, name: str) -> str:
    """Return the text inside `export interface <name> { ... }`, with
    nested braces (e.g. inline object types) handled correctly."""
    m = re.search(rf"export\s+interface\s+{re.escape(name)}\s*\{{", source)
    if not m:
        raise AssertionError(f"interface `{name}` not found in source")
    start = m.end()
    depth = 1
    i = start
    while i < len(source) and depth > 0:
        c = source[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
        i += 1
    if depth != 0:
        raise AssertionError(f"unbalanced braces parsing interface `{name}`")
    return source[start : i - 1]


def _extract_interface_field_names(source: str, name: str) -> set[str]:
    """Top-level field names (ignores nested object literal keys).

    We only count names that appear at brace depth 1 inside the
    interface so a nested type literal — e.g. `base: { probUp: number }`
    in `MlMetaPredictRequest` — doesn't smuggle `probUp` into the
    parent's field set.
    """
    body = _strip_ts_comments(_extract_interface_body(source, name))
    fields: set[str] = set()
    depth = 0
    line_start = 0
    pending_field: str | None = None
    # Walk char-by-char so we can track brace depth while still
    # extracting the leading identifier of each top-level member line.
    for i, ch in enumerate(body):
        if ch == "{":
            depth += 1
            continue
        if ch == "}":
            depth -= 1
            continue
        if depth == 0 and ch == "\n":
            line = body[line_start:i]
            line_start = i + 1
            mm = re.match(r"\s*([A-Za-z_$][\w$]*)\s*\??\s*:", line)
            if mm:
                fields.add(mm.group(1))
        elif depth == 0 and ch == ";":
            # Same line, multiple statements — rare but cheap to handle.
            line = body[line_start : i + 1]
            mm = re.match(r"\s*([A-Za-z_$][\w$]*)\s*\??\s*:", line)
            if mm:
                fields.add(mm.group(1))
            line_start = i + 1
    # Tail (interface body without trailing newline)
    line = body[line_start:]
    mm = re.match(r"\s*([A-Za-z_$][\w$]*)\s*\??\s*:", line)
    if mm:
        fields.add(mm.group(1))
    return fields


# --- module-level fixtures (cheap, parsed once) ------------------------
_ML_CLIENT_SRC = _ML_CLIENT_TS.read_text()
_JOURNAL_WRITER_SRC = _JOURNAL_WRITER_TS.read_text()

_PY_PREDICT_FIELDS: set[str] = set(PredictResponse.model_fields.keys())
_TS_PREDICT_FIELDS: set[str] = _extract_interface_field_names(
    _ML_CLIENT_SRC, "MlPredictResponse"
)
_TS_JOURNAL_FIELDS: set[str] = _extract_interface_field_names(
    _JOURNAL_WRITER_SRC, "InsertPredictionJournalArgs"
)


def _fmt(missing: set[str]) -> str:
    return ", ".join(sorted(missing)) or "<none>"


# --- tests -------------------------------------------------------------
def test_ts_interfaces_parsed_nontrivially():
    """Sanity: if the regex parser silently returns an empty set, every
    other test below would pass for the wrong reason. Pin a few names
    that have been on each interface since the file was written."""
    assert {"coinId", "timeframe", "probUp", "probDown", "source"}.issubset(
        _TS_PREDICT_FIELDS
    ), f"MlPredictResponse parse failed: got {_fmt(_TS_PREDICT_FIELDS)}"
    assert {"coinId", "timeframe", "brain", "predictionId"}.issubset(
        _TS_JOURNAL_FIELDS
    ), f"InsertPredictionJournalArgs parse failed: got {_fmt(_TS_JOURNAL_FIELDS)}"


def test_every_python_predict_field_is_exposed_by_typescript_interface():
    """Every field on `PredictResponse` (Python) must also appear on
    `MlPredictResponse` (TS), otherwise the Node side cannot read a
    value the Python service is actively returning. Allow-listed
    Python-only fields are documented in `_PYTHON_ONLY_PREDICT_FIELDS`."""
    expected = _PY_PREDICT_FIELDS - _PYTHON_ONLY_PREDICT_FIELDS
    missing = expected - _TS_PREDICT_FIELDS
    assert not missing, (
        f"PredictResponse fields missing from MlPredictResponse "
        f"(api-server can't read them): {_fmt(missing)}. "
        f"Either add them to artifacts/api-server/src/lib/ml-client.ts "
        f"`MlPredictResponse`, or — if the field is intentionally "
        f"Python-only — add it to `_PYTHON_ONLY_PREDICT_FIELDS` with a "
        f"comment explaining why."
    )


def test_every_typescript_predict_field_is_emitted_by_python_schema():
    """Every field on `MlPredictResponse` (TS) must exist on
    `PredictResponse` (Python). Otherwise the TS interface advertises
    a field that's never populated and downstream code (e.g. the
    journal-writer) silently writes nulls forever — exactly the Task
    #460 failure mode."""
    expected = _TS_PREDICT_FIELDS - _TS_ONLY_PREDICT_FIELDS
    missing = expected - _PY_PREDICT_FIELDS
    assert not missing, (
        f"MlPredictResponse fields missing from PredictResponse "
        f"(Python service never emits them, journal would store nulls): "
        f"{_fmt(missing)}. Either add them to "
        f"artifacts/ml-engine/app/main.py `PredictResponse`, or — if "
        f"the field is intentionally TS-only — add it to "
        f"`_TS_ONLY_PREDICT_FIELDS` with a comment explaining why."
    )


def test_feature_hash_pinned_on_all_three_sides():
    """Task #460 regression pin. `featureHash` MUST be present on
    `PredictResponse`, `MlPredictResponse`, AND
    `InsertPredictionJournalArgs`. Removing it from any of the three
    breaks the QUANT prediction-journal write path silently."""
    assert "featureHash" in _PY_PREDICT_FIELDS, (
        "PredictResponse (artifacts/ml-engine/app/main.py) is missing "
        "`featureHash` — this is the exact regression Task #460 fixed. "
        "Without it, every QUANT row's hash falls back to the "
        "`missing:...` synthesized placeholder and downstream replay "
        "loses the link to the real feature vector."
    )
    assert "featureHash" in _TS_PREDICT_FIELDS, (
        "MlPredictResponse (artifacts/api-server/src/lib/ml-client.ts) "
        "is missing `featureHash` — the Node client cannot forward what "
        "it cannot read."
    )
    assert "featureHash" in _TS_JOURNAL_FIELDS, (
        "InsertPredictionJournalArgs (artifacts/api-server/src/lib/"
        "journal-writer.ts) is missing `featureHash` — the journal "
        "writer's QUANT-row guard depends on it."
    )


def test_journal_predict_mapping_resolves_on_both_sides():
    """Every (journal-arg, predict-field) pair in the canonical mapping
    must resolve on both the Python and TS sides. This is the
    fail-fast trigger for the Task #460 class of bug: a field that
    journal-writer reads from the predict response but no longer
    exists (or was renamed) on either side."""
    journal_only_missing: list[str] = []
    py_predict_missing: list[str] = []
    ts_predict_missing: list[str] = []

    for journal_arg, predict_field in JOURNAL_FROM_PREDICT_MAPPING.items():
        if journal_arg not in _TS_JOURNAL_FIELDS:
            journal_only_missing.append(journal_arg)
        if predict_field not in _PY_PREDICT_FIELDS:
            py_predict_missing.append(predict_field)
        if predict_field not in _TS_PREDICT_FIELDS:
            ts_predict_missing.append(predict_field)

    problems: list[str] = []
    if journal_only_missing:
        problems.append(
            f"InsertPredictionJournalArgs missing journal-side field(s): "
            f"{_fmt(set(journal_only_missing))}"
        )
    if py_predict_missing:
        problems.append(
            f"PredictResponse (pydantic) missing predict-side field(s): "
            f"{_fmt(set(py_predict_missing))}"
        )
    if ts_predict_missing:
        problems.append(
            f"MlPredictResponse (ts) missing predict-side field(s): "
            f"{_fmt(set(ts_predict_missing))}"
        )
    assert not problems, (
        "predict↔journal contract drift detected:\n  - "
        + "\n  - ".join(problems)
        + "\nIf this is an intentional rename/removal, update "
        "JOURNAL_FROM_PREDICT_MAPPING in this file to match."
    )


@pytest.mark.parametrize(
    "field_name",
    sorted(JOURNAL_FROM_PREDICT_MAPPING.values()),
)
def test_each_mapped_predict_field_is_present_in_python_schema(field_name: str):
    """Per-field parametrization so a CI failure names the EXACT field
    that drifted, not a wall-of-text combined diff."""
    assert field_name in _PY_PREDICT_FIELDS, (
        f"PredictResponse is missing `{field_name}` — journal-writer "
        f"reads this field via the canonical mapping. Add it back to "
        f"artifacts/ml-engine/app/main.py `PredictResponse`."
    )


@pytest.mark.parametrize(
    "field_name",
    sorted(JOURNAL_FROM_PREDICT_MAPPING.values()),
)
def test_each_mapped_predict_field_is_present_in_ts_interface(field_name: str):
    assert field_name in _TS_PREDICT_FIELDS, (
        f"MlPredictResponse is missing `{field_name}` — journal-writer "
        f"reads this field via the canonical mapping. Add it back to "
        f"artifacts/api-server/src/lib/ml-client.ts `MlPredictResponse`."
    )
