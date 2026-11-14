"""Tests for Lindley queue replay math, capacity-regime suite, and moving-block bootstrap."""

from __future__ import annotations

import datetime as dt
import unittest

import numpy as np
import pandas as pd

from src.model import StaffingParameters
from src.replay import (
    evaluate_capacity_regimes,
    evaluate_four_policies,
    paired_moving_block_bootstrap,
    replay_weekly_queue,
    run_sensitivity_grid,
)


class TestReplayAndBootstrap(unittest.TestCase):
    def test_hand_worked_queue_replay_no_negative_banking(self) -> None:
        """Excess picking capacity on low-demand days must clip backlog at 0, never bank credit."""
        params = StaffingParameters(core_capacity_c=100, flex_productivity_q=25, friday_penalty=3.0)
        out = replay_weekly_queue([2, 2, 2, 2, 2], [200, 120, 50, 180, 160], params)
        self.assertEqual(out.realized_backlog, (50.0, 20.0, 0.0, 30.0, 40.0))
        self.assertAlmostEqual(out.total_line_days, 140.0)
        self.assertAlmostEqual(out.friday_backlog_lines, 40.0)
        self.assertAlmostEqual(out.weekly_objective_cost, 260.0)

    def test_policy_evaluation_and_capacity_regimes_break_degeneracy(self) -> None:
        """Verify oracle <= all policies, MILP loss tracking, and P90 schedule responsiveness."""
        rows = []
        base_mon = dt.date(2011, 10, 3)
        week_specs = [
            ([220, 160, 100, 90, 90], [210, 155, 105, 95, 90], [218, 158, 102, 92, 90]),
            ([95, 100, 110, 190, 210], [95, 105, 115, 180, 195], [95, 100, 110, 188, 208]),
            ([80, 80, 100, 215, 225], [220, 200, 100, 85, 85], [215, 195, 100, 90, 90]),
        ]
        wd_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
        for w_idx, (act, p_prev, p_ridge) in enumerate(week_specs):
            wm = base_mon + dt.timedelta(days=7 * w_idx)
            for d in range(5):
                rows.append(
                    {
                        "operational_date": wm + dt.timedelta(days=d),
                        "week_monday": wm,
                        "week_idx": 90 + w_idx,
                        "iso_weekday": d + 1,
                        "weekday_name": wd_names[d],
                        "booked_lines": act[d],
                        "pred_prev_week": p_prev[d],
                        "pred_ridge": p_ridge[d],
                    }
                )

        test_df = pd.DataFrame(rows)
        params = StaffingParameters(core_capacity_c=100, flex_productivity_q=25, friday_penalty=3.0)
        summ_df, wk_df, _ = evaluate_four_policies(
            test_df, params, allow_offline_fallback=True, verify_with_enumerator=True
        )

        self.assertEqual(len(wk_df), 3)
        for pol in ("flat", "prev_week_milp", "ridge_milp"):
            self.assertTrue(
                np.all(wk_df["oracle_milp_objective_cost"] <= wk_df[f"{pol}_objective_cost"] + 1e-6)
            )
        self.assertEqual(wk_df.iloc[2]["ridge_outcome_vs_flat"], "loss")
        self.assertGreater(int(summ_df.set_index("policy_key").loc["ridge_milp", "unique_weekly_schedules"]), 1)

        # Build synthetic initial training spine spanning low off-peak (50) and high peak (150)
        train_dates = pd.bdate_range("2010-06-07", periods=100, freq="B")
        train_df = pd.DataFrame(
            {
                "operational_date": train_dates.date,
                "week_idx": np.arange(100) // 5,
                "booked_lines": np.concatenate([np.full(80, 50.0), np.full(20, 150.0)]),
            }
        )
        reg_df = evaluate_capacity_regimes(
            daily_df=train_df,
            initial_train_end_week_idx=19,
            test_predictions_df=test_df,
            allow_offline_fallback=True,
        )
        self.assertEqual(len(reg_df), 4)
        by_reg = reg_df.set_index("regime_key")
        # Low P50 (50 -> C=35, q=8) saturates to 1 schedule (4-4-2-0-0); P90 (150 -> C=105, q=22) adapts!
        self.assertEqual(int(by_reg.loc["train_p50_median", "unique_ridge_schedules"]), 1)
        self.assertGreater(int(by_reg.loc["train_p90_quantile", "unique_ridge_schedules"]), 1)

        sens = run_sensitivity_grid(test_df, params, allow_offline_fallback=True)
        self.assertEqual(len(sens), 9)

    def test_paired_moving_block_bootstrap_reproducibility(self) -> None:
        base_costs = np.array([120, 140, 100, 90, 160, 180, 110, 130, 150, 170, 95, 125], dtype=float)
        cand_costs = base_costs - 25.0
        b1 = paired_moving_block_bootstrap(base_costs, cand_costs, "test", block_length_weeks=3, seed=99)
        b2 = paired_moving_block_bootstrap(base_costs, cand_costs, "test", block_length_weeks=3, seed=99)

        self.assertEqual(b1, b2)
        self.assertAlmostEqual(b1.observed_mean_weekly_diff, 25.0)
        self.assertAlmostEqual(b1.ci_lower_weekly_diff, 25.0)
        self.assertAlmostEqual(b1.ci_upper_weekly_diff, 25.0)


if __name__ == "__main__":
    unittest.main()
