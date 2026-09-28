"""Strict event gates with the original native order-sizing path."""
from __future__ import annotations

from . import _markowitz as A
from .policies import drift_trigger, short_gap_trigger, weight_gaps

pd = A.pd

class StrictPolicyStrategy(A.Point3Strategy):
    def run(self):
        cfg = A.RUN_CONTEXT["policy_config"]
        policy = A.RUN_CONTEXT["policy"]
        if policy in {"monthly", "stress_original", "drift_original"}:
            return super().run()

        date = pd.Timestamp(self._context.current_date)
        decision = self.decisions.get(date)
        reason = None
        name_gap, global_gap = None, None
        if decision is None and self.mode == "stress":
            stress, _ = A._current_stress_regime(date)
            target_short = A.stress_short_fraction(stress)
            if short_gap_trigger(target_short, self.last_rebalanced_short, cfg["short_delta"]):
                decision = A._daily_decision(date)
                reason = f"stress_short_target_delta_{cfg['short_delta'] * 100:g}pp"
        elif decision is None and self.mode == "drift":
            target = A._daily_decision(date)
            if target is not None:
                current = A._actual_weights_at_previous_close(self, date)
                name_gap, global_gap = weight_gaps(current, target["raw_weights"])
                if drift_trigger(name_gap, global_gap, cfg):
                    decision = target
                    reason = f"drift_gap_{cfg['name_gap'] * 100:g}_{cfg['global_gap'] * 100:g}pp"

        # Invoke the unmodified native sizing path on every bar. An eligible
        # off-cycle target is temporarily supplied as a scheduled target.
        # Remove it before returning, so the original calendar remains frozen.
        injected = reason is not None and decision is not None
        original_mode = self.mode
        if injected:
            self.decisions[date] = decision
        self.mode = "monthly"
        try:
            result = super().run()
            if result is not None:
                info = result[1]
                info.update(rebalance_mode=original_mode, policy=policy)
                if injected:
                    info.update(trigger_reason=reason, max_single_name_gap=name_gap,
                                one_way_target_turnover_gap=global_gap)
            return result
        finally:
            self.mode = original_mode
            if injected:
                del self.decisions[date]
