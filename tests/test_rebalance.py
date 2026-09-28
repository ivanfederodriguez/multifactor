from __future__ import annotations

import argparse
import ast
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from luis_dqi.backtest.policies import POLICIES, drift_trigger, short_gap_trigger, weight_gaps
from luis_dqi.backtest.runner import ENGINE_COMMIT, PACKAGE, check_engine, parse_args, validate_args
from luis_dqi.backtest.runner import REFERENCE_INPUTS
from luis_dqi.backtest.verify import verify_run


def ast_hash(node):
    return hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest()


class FrozenDefinitionTests(unittest.TestCase):
    def test_all_economic_definitions_match_research_sources(self):
        provenance = json.loads((PACKAGE / "provenance.json").read_text())
        for filename, expected in provenance["definition_ast_sha256"].items():
            definitions = {node.name: node for node in ast.parse((PACKAGE / filename).read_text()).body
                           if isinstance(node, (ast.FunctionDef, ast.ClassDef))}
            self.assertTrue(set(expected).issubset(definitions), filename)
            for name, fingerprint in expected.items():
                self.assertEqual(ast_hash(definitions[name]), fingerprint, (filename, name))

    def test_run_case_changes_only_explicit_luis_input_bootstrap(self):
        provenance = json.loads((PACKAGE / "provenance.json").read_text())
        definition = next(node for node in ast.parse((PACKAGE / "_engine_adapter.py").read_text()).body
                          if isinstance(node, ast.FunctionDef) and node.name == "run_case")
        for node in ast.walk(definition):
            if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id in
                                                    ("spec", "score_path") for target in node.targets):
                node.value = ast.Constant("explicit_input")
        self.assertEqual(ast_hash(definition), provenance["normalized_run_case_ast_sha256"])

    def test_frozen_policy_table_and_sizing_constants(self):
        provenance = json.loads((PACKAGE / "provenance.json").read_text())
        table = next(node for node in ast.parse((PACKAGE / "policies.py").read_text()).body
                     if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "POLICIES" for t in node.targets))
        self.assertEqual(ast_hash(table), provenance["policies_assignment_ast_sha256"])
        constants = {node.targets[0].id: ast.literal_eval(node.value)
                     for node in ast.parse((PACKAGE / "_markowitz.py").read_text()).body
                     if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
                     and isinstance(node.value, ast.Constant)}
        self.assertEqual({key: constants[key] for key in (
            "LOOKBACK_PRICES", "MIN_OBS", "VALID_RETURN_FRACTION", "RIDGE", "SHORT_CAP", "P_THRESHOLD",
            "FEE_DRIFT_BUFFER", "SINGLE_NAME_DRIFT_TRIGGER", "PORTFOLIO_ONE_WAY_DRIFT_TRIGGER")}, {
            "LOOKBACK_PRICES":253, "MIN_OBS":60, "VALID_RETURN_FRACTION":.90, "RIDGE":1e-6,
            "SHORT_CAP":.30, "P_THRESHOLD":75., "FEE_DRIFT_BUFFER":1e-8,
            "SINGLE_NAME_DRIFT_TRIGGER":.05, "PORTFOLIO_ONE_WAY_DRIFT_TRIGGER":.10})


class RebalancePolicyTests(unittest.TestCase):
    def test_small_and_large_name_deviation(self):
        small = weight_gaps({"AAPL":.10, "B":.90}, {"AAPL":.09, "B":.91})
        large = weight_gaps({"AAPL":.10, "B":.90}, {"AAPL":.01, "B":.99})
        self.assertAlmostEqual(small[0], .01)
        self.assertAlmostEqual(small[1], .01)
        self.assertFalse(drift_trigger(*small, POLICIES["drift_original"]))
        self.assertTrue(drift_trigger(*large, POLICIES["drift_original"]))

    def test_signed_short_weights_and_union_of_entries_exits(self):
        for current, target in (({"L":.7, "S":-.3}, {"L":.8, "S":-.2}),
                                ({"EXIT":.1}, {"ENTER":.1})):
            name_gap, global_gap = weight_gaps(current, target)
            self.assertAlmostEqual(name_gap, .1)
            self.assertAlmostEqual(global_gap, .1)

    def test_drift_thresholds_are_inclusive_and_or(self):
        for name in ("drift_original", "drift_7p5_15", "drift_10_20", "drift_15_30"):
            cfg = POLICIES[name]
            self.assertTrue(drift_trigger(cfg["name_gap"], 0, cfg))
            self.assertTrue(drift_trigger(0, cfg["global_gap"], cfg))
            self.assertFalse(drift_trigger(cfg["name_gap"] - 1e-6, cfg["global_gap"] - 1e-6, cfg))

    def test_stress_delta_compares_last_rebalance_not_yesterday(self):
        self.assertFalse(short_gap_trigger(.20, None, .20))
        self.assertFalse(short_gap_trigger(.199, 0., .20))
        self.assertTrue(short_gap_trigger(.20, 0., .20))
        self.assertTrue(short_gap_trigger(0., .20, .20))

    def test_only_original_stress_has_a_regime_cross_gate(self):
        self.assertTrue(POLICIES["stress_original"]["regime_cross"])
        for policy in ("stress_gap5", "stress_gap10", "stress_gap15", "stress_gap20"):
            self.assertFalse(POLICIES[policy]["regime_cross"])
        self.assertEqual(len(POLICIES), 10)


class PortableCliTests(unittest.TestCase):
    def arguments(self, root):
        for name in ("scores", "prices", "macro"):
            (root / name).write_bytes(b"synthetic input only")
        return parse_args(["--qbacktest-root", str(root), "--scores-path", str(root / "scores"),
            "--prices-path", str(root / "prices"), "--macro-path", str(root / "macro"),
            "--output-dir", str(root / "new_run")])

    def test_defaults_and_no_expired_cutoff(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self.arguments(Path(directory))
            validate_args(args)
            self.assertIsNone(args.deadline_utc)
            self.assertEqual((args.n_positions, args.scenario, args.short_cap), (30, "p75", .30))

    def test_event_policies_do_not_change_short_mandate(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self.arguments(Path(directory))
            args.policy = "drift_10_20"
            for key, value in (("scenario", "p90"), ("n_positions", 40), ("short_cap", .40)):
                original = getattr(args, key)
                setattr(args, key, value)
                with self.assertRaisesRegex(ValueError, "keep Top 30"):
                    validate_args(args)
                setattr(args, key, original)
            validate_args(args)

    def test_overwrite_reference_costs_and_nonfinite_parameters_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self.arguments(Path(directory))
            args.reference, args.fee_bps = True, 0.
            with self.assertRaisesRegex(ValueError, "reference costs"):
                validate_args(args)
            args.reference, args.fee_bps = False, float("nan")
            with self.assertRaisesRegex(ValueError, "finite"):
                validate_args(args)
            args.fee_bps = 10.
            args.output_dir.mkdir()
            with self.assertRaises(FileExistsError):
                validate_args(args)

    def test_naive_or_expired_deadline_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self.arguments(Path(directory))
            for value in ("2099-01-01T00:00:00", "2000-01-01T00:00:00+00:00"):
                args.deadline_utc = value
                with self.assertRaisesRegex(ValueError, "timezone-aware"):
                    validate_args(args)

    def test_engine_commit_and_source_changes_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "src/qbacktest").mkdir(parents=True)
            (root / "src/qbacktest/__init__.py").write_text("")
            for outputs in (("different", "", ""), (ENGINE_COMMIT, "modified source", ""),
                            (ENGINE_COMMIT, "", "src/qbacktest/new.py")):
                with patch("luis_dqi.backtest.runner.subprocess.check_output", side_effect=outputs):
                    with self.assertRaises(ValueError):
                        check_engine(root)


class VerifierGuardTests(unittest.TestCase):
    def manifest(self):
        return {"status":"completed", "metrics":{}, "contract":{"qbacktest_commit":ENGINE_COMMIT},
                "inputs":{key:{"sha256":value} for key,value in REFERENCE_INPUTS.items()}}

    def check_bad_manifest(self, manifest, message):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "run_manifest.json").write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, message):
                verify_run(path)

    def test_incomplete_run_rejected(self):
        manifest = self.manifest()
        manifest["status"] = "running"
        self.check_bad_manifest(manifest, "not completed")

    def test_different_engine_rejected(self):
        manifest = self.manifest()
        manifest["contract"]["qbacktest_commit"] = "different"
        self.check_bad_manifest(manifest, "Engine commit")

    def test_different_input_identity_rejected(self):
        manifest = self.manifest()
        manifest["inputs"]["scores"]["sha256"] = "different"
        self.check_bad_manifest(manifest, "Input hashes")


@unittest.skipUnless(os.environ.get("LUIS_QBACKTEST_ROOT"), "Needs a separately authorized qbacktest checkout")
class NativeAdapterUnitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = check_engine(Path(os.environ["LUIS_QBACKTEST_ROOT"]))
        sys.path.insert(0, str(root / "src"))
        cls.A = importlib.import_module("luis_dqi.backtest._markowitz")

    def setUp(self):
        self.A.RUN_CONTEXT.clear()
        self.A.MKW_STATS.clear()

    def test_p75_cap30_target_unchanged_across_all_policies(self):
        for name in POLICIES:
            self.A.RUN_CONTEXT.update(threshold=75., short_cap=.30, policy=name)
            for stress, expected in ((None, 0.), (50., 0.), (75., 0.), (87.5, .15), (100., .30)):
                self.assertAlmostEqual(self.A.stress_short_fraction(stress), expected)

    def test_caps_redistribute_and_infeasible_cap_rejected(self):
        import numpy as np
        np.testing.assert_allclose(self.A.capped_proportional_weights(np.array([8., 1., 1.]), .5, 1.), [.5, .25, .25])
        with self.assertRaises(ValueError):
            self.A.capped_proportional_weights(np.array([8., 1., 1.]), .1, 1.)

    def test_macro_requires_batch_strictly_before_execution(self):
        pd = self.A.pd
        date = pd.Timestamp("2020-03-03")
        macro = pd.DataFrame({"BATCH_DATE":pd.to_datetime(["2020-03-02", "2020-03-03"]),
            "FIRST_TRADABLE_DATE":pd.to_datetime(["2020-03-03", "2020-03-03"]), "STRESS_PERCENTILE_0_100":[80., 100.]})
        stress, _ = self.A._causal_macro(macro, date)
        self.assertEqual(stress, 80.)

    def test_fee_reserve_and_missing_entry_no_renormalization(self):
        weights, info = self.A.BASE.fee_reserved_weights({"A":.8, "MISSING":.2}, {}, 1_000_000., {"A":10.}, .001)
        self.assertLess(weights["A"], .8)
        self.assertNotIn("MISSING", weights)
        self.assertEqual(info["missing_entry_weight"], .2)
        self.assertGreater(info["fee_reserve"], 0.)

    def test_frozen_untradable_holding_kept(self):
        weights, info = self.A.BASE.fee_reserved_weights({"A":1.}, {"DELISTED":.1}, 1_000_000., {"A":10.}, .001)
        self.assertEqual(weights["DELISTED"], .1)
        self.assertEqual(info["frozen_gross_weight"], .1)


if __name__ == "__main__":
    unittest.main()
