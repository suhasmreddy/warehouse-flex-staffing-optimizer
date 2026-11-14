"""Tests for Lindley queue replay math, four-policy invariants, and moving-block bootstrap."""

from __future__ import annotations

import datetime as dt
import unittest

import numpy as np
import pandas as pd

from src.model import StaffingParameters
from src.replay import (
    evaluate_four_policies,
    paired_moving_block_bootstrap,
    replay_weekly_queue,
    run_sensitivity_grid,
)


class TestReplayAndBootstrap(unittest.TestCase):
    def test_hand_worked_queue_replay_no_negative_banking(self) -> None:
        """Excess picking capacity on low-demand days must clip backlog at 0, never bank credit."""
        params = StaffingParameters(core_capacity_c=100, flex_productivity_q=25, friday_penalty=3.0)
        # Schedule: (2, 2, 2, 2, 2) -> daily total capacity = 100 + 50 = 150 lines/day
        # Mon: 200 -> backlog = 50
        # Tue: 120 -> 50 + 120 - 150 = 20
        # Wed: 50  -> max(0, 20 + 50 - 150) = 0  (80 unused capacity CANNOT carry into Thu!)
        # Thu: 180 -> max(0, 0 + 180 - 150) = 30
        # Fri: 160 -> max(0, 30 + 160 - 150) = 40
        out = replay_weekly_queue([2, 2, 2, 2, 2], [200, 120, 50, 180, 160], params)
        self.assertEqual(out.realized_backlog, (50.0, 20.0, 0.0, 30.0, 40.0))
        self.assertAlmostEqual(out.total_line_days, 140.0)
        self.assertAlmostEqual(out.friday_backlog_lines, 40.0)
        # Objective = 140 + 3 * 40 = 260
        self.assertAlmostEqual(out.weekly_objective_cost, 260.0)

    def test_policy_evaluation_and_oracle_lower_bound(self) -> None:
        """Verify identical test weeks, oracle <= all policies, and ability to record MILP losses."""
        rows = []
        base_mon = dt.date(2011, 10, 3)
        # Construct 3 test weeks: Weeks 1 & 2 accurately forecasted; Week 3 has an inverted shock
        # so forecast-driven MILP deliberately loses to Flat Staffing on Week 3!
        week_specs = [
            # (actual_5d, pred_prev_5d, pred_ridge_5d)
            ([220, 160, 100, 90, 90], [210, 155, 105, 95, 90], [218, 158, 102, 92, 90]),
            ([95, 100, 110, 190, 210], [95, 105, 115, 180, 195], [95, 100, 110, 188, 208]),
            # Week 3 shock: forecast expects Mon/Tue surge, actual surge hits Thu/Fri!
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
        # Oracle must beat or tie every policy on every single week
        for pol in ("flat", "prev_week_milp", "ridge_milp"):
            self.assertTrue(
                np.all(wk_df["oracle_milp_objective_cost"] <= wk_df[f"{pol}_objective_cost"] + 1e-6)
            )
        # Week 3 must explicitly register as a Ridge MILP loss vs Flat Staffing
        self.assertEqual(wk_df.iloc[2]["ridge_outcome_vs_flat"], "loss")

        sens = run_sensitivity_grid(test_df, params, allow_offline_fallback=True)
        self.assertEqual(len(sens), 9)

    def test_paired_moving_block_bootstrap_reproducibility(self) -> None:
        base_costs = np.array([120, 140, 100, 90, 160, 180, 110, 130, 150, 170, 95, 125], dtype=float)
        cand_costs = base_costs - 25.0  # Constant +25 weekly saving
        b1 = paired_moving_block_bootstrap(base_costs, cand_costs, "test", block_length_weeks=3, seed=99)
        b2 = paired_moving_block_bootstrap(base_costs, cand_costs, "test", block_length_weeks=3, seed=99)

        self.assertEqual(b1, b2)
        self.assertAlmostEqual(b1.observed_mean_weekly_diff, 25.0)
        self.assertAlmostEqual(b1.ci_lower_weekly_diff, 25.0)
        self.assertAlmostEqual(b1.ci_upper_weekly_diff, 25.0)


if __name__ == "__main__":
    unittest.main()
