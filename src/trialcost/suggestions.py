"""Reviewer suggestions: proposed changes to catalogue values, kept apart from data/ (D40).

In the app, a reviewer edits a value (with a rationale); the change becomes a
*suggestion* that applies to their session only. They download the suggestions
CSV and send it to Vincent, who reviews and applies the accepted ones:

    uv run python -m trialcost.suggestions review suggestions/inbox/x.csv [--design DESIGN.yaml]
    uv run python -m trialcost.suggestions apply  suggestions/inbox/x.csv --accept 1,3 [--grade C]

``apply`` is the only code path that writes to ``data/``. It edits just the
accepted cells (csv module, so the rest of each file is byte-identical), sets
the row's evidence grade, appends "Expert review YYYY-MM-DD" to its source and
the rationale to its notes, re-validates the whole catalogue and restores the
original files if validation fails.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import shutil
import sys
import tempfile
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from trialcost import DATA_DIR
from trialcost.catalogue import COUNTRY_RANGES, Catalogue, CatalogueError, load_catalogue

COLUMNS = [
    "suggestion_id", "table", "row_id", "field", "current_value", "suggested_value",
    "rationale", "reviewer", "created",
]  # fmt: skip

_TRIPLE = ("low", "mode", "high")
#: Editable tables -> (catalogue attribute, CSV file, id columns, editable fields).
TABLES: dict[str, tuple[str, str, tuple[str, ...], tuple[str, ...]]] = {
    "procedures": ("procedures", "procedures.csv", ("procedure_id",), _TRIPLE),
    "study_items": ("study_items", "study_items.csv", ("item_id",), _TRIPLE),
    "constants": ("constants", "constants.csv", ("constant_id",), _TRIPLE),
    "procedure_overrides": (
        "overrides", "procedure_overrides.csv", ("procedure_id", "country"), _TRIPLE,
    ),
    "sponsor_model_factors": (
        "sponsor_factors", "sponsor_model_factors.csv", ("sponsor_model",), _TRIPLE,
    ),
    "countries": (
        "countries", "countries.csv", ("country",),
        tuple(f"{r}_{b}" for r in COUNTRY_RANGES for b in _TRIPLE),
    ),
}  # fmt: skip
#: Separator for composite row ids, e.g. "mri_brain|US" in procedure_overrides.
KEY_SEP = "|"


def empty() -> pd.DataFrame:
    return pd.DataFrame(columns=COLUMNS)


def row_key(table: str, index_value) -> str:
    """Row id as stored in a suggestion (tuple index -> "a|b")."""
    return KEY_SEP.join(index_value) if isinstance(index_value, tuple) else str(index_value)


def _index_value(table: str, row_id: str):
    ids = TABLES[table][2]
    return tuple(row_id.split(KEY_SEP)) if len(ids) > 1 else row_id


def make(
    table: str, row_id: str, field: str, current, suggested, rationale: str, reviewer: str = ""
) -> dict:
    return {
        "table": table,
        "row_id": row_id,
        "field": field,
        "current_value": float(current),
        "suggested_value": float(suggested),
        "rationale": rationale.strip(),
        "reviewer": reviewer.strip(),
        "created": datetime.now().isoformat(timespec="seconds"),
    }


def merge(existing: pd.DataFrame, new: list[dict]) -> pd.DataFrame:
    """Add suggestions; a newer one on the same (table, row, field) replaces the older."""
    df = pd.concat([existing[COLUMNS[1:]], pd.DataFrame(new, columns=COLUMNS[1:])])
    df = df.drop_duplicates(subset=["table", "row_id", "field"], keep="last")
    df.insert(0, "suggestion_id", range(1, len(df) + 1))
    return df.reset_index(drop=True)


def from_edits(
    before: pd.DataFrame, after: pd.DataFrame, table: str, reviewer: str = ""
) -> tuple[list[dict], list[str]]:
    """Suggestions from an edited table view (same rows, ``row_id`` and ``rationale`` columns).

    Returns the new suggestions and the row ids changed without a rationale
    (those are not turned into suggestions).
    """
    fields = TABLES[table][3]
    new, missing = [], []
    for (_, b), (_, a) in zip(before.iterrows(), after.iterrows(), strict=True):
        for f in fields:
            if pd.isna(a[f]) or float(a[f]) == float(b[f]):
                continue
            why = str(a.get("rationale") or "").strip()
            if not why:
                missing.append(str(a["row_id"]))
                continue
            new.append(make(table, str(a["row_id"]), f, b[f], a[f], why, reviewer))
    return new, sorted(set(missing))


def read(path_or_buffer) -> pd.DataFrame:
    text = {"table": str, "row_id": str, "field": str, "rationale": str, "reviewer": str}
    df = pd.read_csv(path_or_buffer, dtype=text, keep_default_na=False)
    missing = set(COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"suggestions file is missing columns {sorted(missing)}")
    return df[COLUMNS]


def validate(df: pd.DataFrame, catalogue: Catalogue) -> list[str]:
    """Problems that prevent applying the suggestions (empty list = OK)."""
    problems = []
    for r in df.itertuples():
        tag = f"#{r.suggestion_id} {r.table}/{r.row_id}/{r.field}"
        if r.table not in TABLES:
            problems.append(f"{tag}: unknown table")
            continue
        attr, _, _, fields = TABLES[r.table]
        if r.field not in fields:
            problems.append(f"{tag}: field not editable (allowed: {', '.join(fields)})")
        if _index_value(r.table, str(r.row_id)) not in getattr(catalogue, attr).index:
            problems.append(f"{tag}: unknown row")
        if not np.isfinite(pd.to_numeric(r.suggested_value, errors="coerce")):
            problems.append(f"{tag}: suggested value is not a number")
        if not str(r.rationale).strip():
            problems.append(f"{tag}: rationale is required")
    if problems:
        return problems
    try:
        apply_to_catalogue(catalogue, df, check=False)
    except ValueError as e:
        problems.append(str(e))
    return problems


def apply_to_catalogue(catalogue: Catalogue, df: pd.DataFrame, check: bool = True) -> Catalogue:
    """A copy of the catalogue with the suggestions applied (session use; data/ untouched)."""
    if check and (problems := validate(df, catalogue)):
        raise ValueError("invalid suggestions:\n" + "\n".join(problems))
    tables: dict[str, pd.DataFrame] = {}
    for r in df.itertuples():
        attr, _, _, _ = TABLES[r.table]
        t = tables.setdefault(attr, getattr(catalogue, attr).copy())
        if t[r.field].dtype.kind in "iu":
            t[r.field] = t[r.field].astype(float)
        t.loc[_index_value(r.table, str(r.row_id)), r.field] = float(r.suggested_value)
    _check_ranges(df, tables)
    return dataclasses.replace(catalogue, **tables)


def _check_ranges(df: pd.DataFrame, tables: dict[str, pd.DataFrame]) -> None:
    problems = []
    for (table, row_id), _ in df.groupby(["table", "row_id"]):
        attr, _, _, fields = TABLES[table]
        row = tables[attr].loc[_index_value(table, str(row_id))]
        prefixes = sorted({f[: -len(b)] for f in fields for b in _TRIPLE if f.endswith(b)})
        for prefix in prefixes:
            lo, mo, hi = (row[f"{prefix}{b}"] for b in _TRIPLE)
            if not lo <= mo <= hi:
                problems.append(
                    f"{table}/{row_id}: after the suggestions {prefix}low <= mode <= high "
                    f"fails ({lo:g}, {mo:g}, {hi:g})"
                )
    if problems:
        raise ValueError("\n".join(problems))


# --------------------------------------------------------------------------- #
# Vincent's side: review and apply to data/
# --------------------------------------------------------------------------- #


def impact(
    df: pd.DataFrame, catalogue: Catalogue, design_path: Path, n_sims: int = 2_000, seed: int = 42
) -> pd.DataFrame:
    """Mean total cost of a design with each suggestion alone, and all together."""
    from trialcost.design import load_design
    from trialcost.simulate import run

    design = load_design(design_path)
    base = run(design, catalogue, n_sims, seed).total.mean()
    out = df[
        ["suggestion_id", "table", "row_id", "field", "current_value", "suggested_value"]
    ].copy()
    deltas = []
    for i in range(len(df)):
        try:
            cat = apply_to_catalogue(catalogue, df.iloc[[i]], check=False)
            deltas.append(run(design, cat, n_sims, seed).total.mean() - base)
        except ValueError:  # e.g. mode moved above high unless another suggestion moves high
            deltas.append(np.nan)
    out["delta_mean_alone_eur"] = deltas
    together = run(design, apply_to_catalogue(catalogue, df), n_sims, seed).total.mean()
    out.attrs.update(base=base, together=together)
    return out


def _fmt_number(x: float) -> str:
    return str(int(x)) if float(x).is_integer() else f"{x:g}"


def apply_to_data(
    df: pd.DataFrame,
    accept: list[int],
    grade: str = "C",
    data_dir: Path = DATA_DIR,
    today: str | None = None,
) -> list[str]:
    """Write accepted suggestions into data/*.csv; return a log of changes.

    Validates the full catalogue afterwards and restores every file on failure.
    """
    if grade not in "ABCD" or len(grade) != 1:
        raise ValueError("grade must be one of A, B, C, D")
    today = today or date.today().isoformat()
    chosen = df[df["suggestion_id"].isin(accept)]
    if missing := sorted(set(accept) - set(chosen["suggestion_id"])):
        raise ValueError(f"unknown suggestion ids: {missing}")
    if problems := validate(chosen, load_catalogue(data_dir)):
        raise ValueError("invalid suggestions:\n" + "\n".join(problems))

    log = []
    backup = Path(tempfile.mkdtemp(prefix="trialcost-data-backup-"))
    files = {TABLES[t][1] for t in chosen["table"]}
    for f in files:
        shutil.copy(data_dir / f, backup / f)
    try:
        for table, rows in chosen.groupby("table"):
            _, filename, id_cols, _ = TABLES[table]
            path = data_dir / filename
            with path.open(encoding="utf-8", newline="") as fh:
                records = list(csv.DictReader(fh))
                fieldnames = list(records[0].keys())
            for r in rows.itertuples():
                key = _index_value(table, str(r.row_id))
                key = key if isinstance(key, tuple) else (key,)
                rec = next(x for x in records if tuple(x[c] for c in id_cols) == key)
                old = rec[r.field]
                rec[r.field] = _fmt_number(float(r.suggested_value))
                rec["evidence"] = grade
                tag = f"Expert review {today}"
                if tag not in rec["source"]:
                    rec["source"] = f"{rec['source']}; {tag}" if rec["source"] else tag
                note = f"[{today}] {r.field} {old} -> {rec[r.field]}: {r.rationale}"
                rec["notes"] = f"{rec['notes']} {note}".strip() if rec.get("notes") else note
                log.append(f"{filename}: {KEY_SEP.join(key)}.{r.field} {old} -> {rec[r.field]}")
            with path.open("w", encoding="utf-8", newline="") as fh:
                writer = csv.DictWriter(fh, fieldnames=fieldnames, lineterminator="\n")
                writer.writeheader()
                writer.writerows(records)
        load_catalogue(data_dir)  # full validation of the result
    except (CatalogueError, ValueError, StopIteration):
        for f in files:
            shutil.copy(backup / f, data_dir / f)
        raise
    finally:
        shutil.rmtree(backup, ignore_errors=True)
    return log


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m trialcost.suggestions")
    sub = p.add_subparsers(dest="command", required=True)
    r = sub.add_parser("review", help="validate and show the cost impact of a suggestions file")
    r.add_argument("file", type=Path)
    r.add_argument(
        "--design",
        type=Path,
        default=DATA_DIR.parent / "designs" / "rare-neuro-ph2-mri-scales.yaml",
    )
    a = sub.add_parser("apply", help="write accepted suggestions into data/")
    a.add_argument("file", type=Path)
    a.add_argument("--accept", required=True, help="comma-separated suggestion ids, or 'all'")
    a.add_argument("--grade", default="C", help="evidence grade for the changed rows (default C)")
    args = p.parse_args(argv)

    df = read(args.file)
    catalogue = load_catalogue()
    if args.command == "review":
        problems = validate(df, catalogue)
        pd.set_option("display.width", 200)
        print(df.drop(columns=["created"]).to_string(index=False))
        if problems:
            print("\nProblems:\n" + "\n".join(f"- {x}" for x in problems))
            return 1
        imp = impact(df, catalogue, args.design)
        print(f"\nImpact on mean total cost of {args.design.name} (2,000 sims, seed 42):")
        print(
            imp[["suggestion_id", "table", "row_id", "field", "delta_mean_alone_eur"]]
            .round(0)
            .to_string(index=False)
        )
        print(f"All together: {imp.attrs['base']:,.0f} -> {imp.attrs['together']:,.0f} EUR")
        return 0
    accept = (
        df["suggestion_id"].tolist()
        if args.accept == "all"
        else [int(x) for x in args.accept.split(",")]
    )
    for line in apply_to_data(df, accept, args.grade):
        print(line)
    print("\nCatalogue re-validated. Next: run tests, update README §6 / work log, commit.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
