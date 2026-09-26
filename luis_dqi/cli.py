"""Portable score generation and integrity checks; no change to DQI mathematics."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys
import tempfile

import pandas as pd

from . import __version__
from .dqi_prometheus_adapter import adapt_score_tape, sha256_file

PACKAGE = Path(__file__).resolve().parent


def provenance() -> dict:
    return json.loads((PACKAGE / "provenance.json").read_text(encoding="utf-8"))


def validate_source(source: Path) -> None:
    frame = pd.read_parquet(source)
    if frame.empty:
        raise ValueError("Source is empty")
    if not {"date", "symbol"}.issubset(frame.columns):
        raise ValueError("Source requires date and symbol columns")
    dates = pd.to_datetime(frame["date"], errors="raise")
    if dates.isna().any() or frame["symbol"].isna().any():
        raise ValueError("Null date/symbol is not allowed")
    symbols = frame["symbol"].astype(str)
    if symbols.str.strip().eq("").any():
        raise ValueError("Empty symbol is not allowed")
    keys = pd.DataFrame({"date": dates, "symbol": symbols})
    if keys.duplicated().any():
        raise ValueError("Duplicate date/symbol would overwrite the original DQI cache")


def score(source: Path, output: Path, overwrite: bool = False) -> dict:
    source = source.expanduser().resolve()
    output = output.expanduser().resolve()
    manifest_path = output.with_suffix(".manifest.json")
    if source == output or source == manifest_path:
        raise ValueError("Output must not replace the input")
    if not overwrite and (output.exists() or manifest_path.exists()):
        raise FileExistsError("Output/manifest already exists; use a new path or --overwrite")
    validate_source(source)
    source_before = sha256_file(source)
    scores, manifest = adapt_score_tape(source)
    if source_before != manifest["source_sha256"]:
        raise ValueError("Source changed during scoring")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=output.parent, suffix=".parquet", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        scores.to_parquet(temporary, index=False)
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    upstream = provenance()
    manifest["scoring_files_sha256"] = {
        name: sha256_file(PACKAGE / name)
        for name in ("dqi_config.py", "dqi_factor_scores.py", "dqi_prometheus_adapter.py", "cli.py")
    }
    manifest.pop("original_files_sha256", None)
    manifest.update({
        "package": "multifactor-luis-dqi", "package_version": __version__,
        "canonical_mode": upstream["canonical_mode"],
        "upstream_sha256": upstream["upstream_sha256"],
        "packaging_changes": upstream["packaging_changes"],
        "output_path": str(output), "output_sha256": sha256_file(output),
        "environment": {"python": platform.python_version(), "platform": platform.platform(),
                        "packages": {name: importlib.metadata.version(name)
                                     for name in ("numpy", "pandas", "pyarrow")}},
    })
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def verify(scores: Path, manifest: Path | None = None, source: Path | None = None,
           base_b_reference: bool = False) -> dict:
    scores = scores.expanduser().resolve()
    manifest = manifest.expanduser().resolve() if manifest else scores.with_suffix(".manifest.json")
    record = json.loads(manifest.read_text(encoding="utf-8"))
    actual = sha256_file(scores)
    if actual != record["output_sha256"]:
        raise ValueError("Scores hash differs from manifest")
    expected_files = {"dqi_config.py", "dqi_factor_scores.py", "dqi_prometheus_adapter.py", "cli.py"}
    if set(record["scoring_files_sha256"]) != expected_files:
        raise ValueError("Manifest must identify every scoring file")
    if record["canonical_mode"] != provenance()["canonical_mode"]:
        raise ValueError("Manifest uses a different scoring mode")
    for name, wanted in record["scoring_files_sha256"].items():
        if sha256_file(PACKAGE / name) != wanted:
            raise ValueError(f"Installed scoring code differs: {name}")
    if source is not None and sha256_file(source.expanduser().resolve()) != record["source_sha256"]:
        raise ValueError("Source hash differs from manifest")
    if base_b_reference:
        expected = provenance()["base_b_reference"]
        if actual != expected["scores_sha256"] or record["source_sha256"] != expected["source_sha256"]:
            raise ValueError("Not the frozen Base B reference (compare numerical values if changing libraries)")
        if record["output_rows"] != expected["rows"] or record["rated_rows"] != expected["rated_rows"]:
            raise ValueError("Base B coverage differs")
    return {"status": "verified", "scores_sha256": actual, "source_sha256": record["source_sha256"],
            "base_b_reference": base_b_reference}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", action="version", version=__version__)
    commands = parser.add_subparsers(dest="command", required=True)
    generate = commands.add_parser("score", help="Generate monthly scores from raw PROMETHEUS parquet")
    generate.add_argument("--source", type=Path, required=True)
    generate.add_argument("--output", type=Path, required=True)
    generate.add_argument("--overwrite", action="store_true")
    check = commands.add_parser("verify", help="Check scores, code and optionally Base B reference")
    check.add_argument("--scores", type=Path, required=True)
    check.add_argument("--manifest", type=Path)
    check.add_argument("--source", type=Path)
    check.add_argument("--base-b-reference", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.command == "score":
            manifest = score(args.source, args.output, args.overwrite)
            result = {key: manifest[key] for key in ("output_path", "output_sha256", "source_sha256", "output_rows", "rated_rows")}
        else:
            result = verify(args.scores, args.manifest, args.source, args.base_b_reference)
    except (OSError, ValueError, KeyError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2))
    return 0
