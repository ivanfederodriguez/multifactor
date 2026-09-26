from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from luis_dqi.cli import PACKAGE, score, validate_source, verify
from luis_dqi.dqi_factor_scores import SUBFACTOR_FIELD_MAP, score_universe, sector_neutral_zscore, winsorize
from luis_dqi.dqi_prometheus_adapter import SOURCE_COLUMN_MAP, adapt_score_tape


def synthetic_frame() -> pd.DataFrame:
    rows = []
    for date in ("2020-01-31", "2020-02-28"):
        for i in range(30):
            row = {"date": pd.Timestamp(date), "symbol": f"SYNTH{i:02d}",
                   "name": f"Synthetic {i}", "sub_industry": "Synthetic sector",
                   "px_last": float(i + 10), "mkt_cap": float((i + 1) * 1000)}
            for field in SOURCE_COLUMN_MAP.values():
                if field:
                    row[field] = float(i + 1)
            rows.append(row)
    return pd.DataFrame(rows)


class ScorerTests(unittest.TestCase):
    def test_upstream_calculations_unchanged(self):
        provenance = json.loads((PACKAGE / "provenance.json").read_text())
        for name, wanted in provenance["upstream_sha256"].items():
            body = (PACKAGE / name).read_bytes()
            if name == "dqi_factor_scores.py":
                body = body.replace(b"from .dqi_config import (", b"from dqi_config import (")
            elif name == "dqi_prometheus_adapter.py":
                body = body.replace(b"from .dqi_factor_scores import", b"from dqi_factor_scores import")
            self.assertEqual(hashlib.sha256(body).hexdigest(), wanted, name)

    def test_original_weighted_composite_and_missing_rd(self):
        cache = {}
        for i in range(30):
            record = {field: float(i + 1) for field in SUBFACTOR_FIELD_MAP.values()}
            record.update(gics_sub_industry_name="Synthetic sector", rd_intensity=None)
            cache[f"S{i:02d}"] = record
        result = score_universe(cache)
        values = winsorize(np.arange(1., 31.))
        z = (values - values.mean()) / values.std(ddof=0)
        # Value=.2z, Quality=.4z, other factors=z => composite=.72z.
        actual = np.array([result[key]["composite_z"] for key in cache])
        np.testing.assert_allclose(actual, .72 * z, rtol=0, atol=1e-14)
        self.assertTrue(all(item["subfactor_z"]["rd_intensity"] is None for item in result.values()))

    def test_ties_and_insufficient_coverage(self):
        full = {field: 1. for field in SUBFACTOR_FIELD_MAP.values()}
        result = score_universe({f"S{i}": dict(full) for i in range(30)})
        self.assertTrue(all(row["percentile"] == .5 and row["rating"] == "NEUTRAL" for row in result.values()))
        partial = score_universe({"S": {"ev_to_t12m_ebitda": 1., "fcf_yield_with_cur_entp_valu": 1.}})
        self.assertIsNone(partial["S"]["composite_z"])
        self.assertEqual(partial["S"]["rating"], "N/A")

    def test_sector_valid_count_fallback(self):
        values = np.array([1., 2., np.nan, np.nan, np.nan, 20., 30., 40., 50., 60.])
        sectors = ["A"] * 5 + ["B"] * 5
        result = sector_neutral_zscore(values, sectors)
        universe = values[np.isfinite(values)]
        self.assertAlmostEqual(result[0], (1. - universe.mean()) / universe.std())
        np.testing.assert_allclose(result[5:], (values[5:] - values[5:].mean()) / values[5:].std())

    def test_base_b_aliases_order_and_coverage(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            frame = synthetic_frame()
            canonical = root / "canonical.parquet"
            aliases = root / "aliases.parquet"
            frame.to_parquet(canonical, index=False)
            frame.rename(columns={"raw_delta_gm": "raw_delta_gross_margin", "raw_delta_em": "raw_delta_oper_margin"}).sample(frac=1, random_state=42).to_parquet(aliases, index=False)
            first, meta = adapt_score_tape(canonical)
            second, alias_meta = adapt_score_tape(aliases)
            assert_frame_equal(first, second, check_exact=True)
            self.assertEqual(meta["output_rows"], 60)
            self.assertEqual(meta["rated_rows"], 60)
            self.assertEqual(len(alias_meta["source_column_aliases_used"]), 2)


class CliTests(unittest.TestCase):
    def test_generate_verify_and_overwrite_protection(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source, output = root / "raw.parquet", root / "scores.parquet"
            synthetic_frame().to_parquet(source, index=False)
            manifest = score(source, output)
            self.assertEqual(manifest["output_rows"], 60)
            self.assertEqual(verify(output, source=source)["status"], "verified")
            with self.assertRaises(FileExistsError):
                score(source, output)
            with self.assertRaises(ValueError):
                verify(output, base_b_reference=True)
            with self.assertRaises(ValueError):
                score(source, source, overwrite=True)
            output.write_bytes(output.read_bytes() + b"changed")
            with self.assertRaises(ValueError):
                verify(output)

    def test_invalid_keys_and_schema(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "raw.parquet"
            frame = synthetic_frame()
            pd.concat([frame, frame.iloc[:1]]).to_parquet(path, index=False)
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                validate_source(path)
            frame.loc[0, "symbol"] = None
            frame.to_parquet(path, index=False)
            with self.assertRaisesRegex(ValueError, "Null"):
                validate_source(path)
            frame.iloc[:0].to_parquet(path, index=False)
            with self.assertRaisesRegex(ValueError, "empty"):
                validate_source(path)
            pd.DataFrame({"date": ["2020-01-01"]}).to_parquet(path, index=False)
            with self.assertRaisesRegex(ValueError, "date and symbol"):
                validate_source(path)

    def test_incomplete_code_manifest_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            source, output = Path(folder) / "raw.parquet", Path(folder) / "scores.parquet"
            synthetic_frame().to_parquet(source, index=False)
            record = score(source, output)
            record["scoring_files_sha256"] = {}
            output.with_suffix(".manifest.json").write_text(json.dumps(record))
            with self.assertRaisesRegex(ValueError, "every scoring file"):
                verify(output)


if __name__ == "__main__":
    unittest.main()
