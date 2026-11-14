"""Tests for MILP solver status, integrality, training capacity calibration, and brute-force equivalence."""

from __future__ import annotations

import datetime as dt
import unittest

import numpy as np
import pandas as pd

from src.model import (
    StaffingParameters,
    compute_training_capacity_reference,
    enumerate_optimal_schedule,
    solve_weekly_staffing_milp,
)


class TestModelAndEnumerator(unittest.TestCase):
    def setUp(self) -> None:
        self.params = StaffingParameters(
            core_capacity_c=700,
            flex_productivity_q=150,
            weekly_blocks=10,
            min_daily_blocks=0,
            max_daily_blocks=4,
            initial_backlog=0.0,
            friday_penalty=3.0,
        )

    def test_enumerator_checks_all_381_schedules(self) -> None:
        enum_res = enumerate_optimal_schedule([800, 900, 1000, 1100, 1200], self.params)
        self.assertEqual(enum_res.num_feasible_schedules_checked, 381)

    def test_zero_demand_instance(self) -> None:
        forecast = [0.0, 0.0, 0.0, 0.0, 0.0]
        sol = solve_weekly_staffing_milp(forecast, self.params, allow_offline_fallback=True)
        enum_res = enumerate_optimal_schedule(forecast, self.params)

        self.assertEqual(sol.solver_status, "optimal")
        self.assertEqual(sum(sol.blocks), 10)
        self.assertAlmostEqual(sol.objective_value, 0.0, places=6)
        self.assertEqual(sol.modeled_backlog, (0.0, 0.0, 0.0, 0.0, 0.0))
        self.assertIn(sol.blocks, enum_res.all_optimal_schedules)

    def test_persistent_overload_frontloads_capacity(self) -> None:
        """Under unbroken daily overload, a Monday block saves backlog across all 5 days + 3*Fri."""
        forecast = [2500.0, 2500.0, 2500.0, 2500.0, 2500.0]
        sol = solve_weekly_staffing_milp(forecast, self.params, allow_offline_fallback=True)
        enum_res = enumerate_optimal_schedule(forecast, self.params)

        self.assertEqual(sol.blocks, (4, 4, 2, 0, 0))
        self.assertEqual(enum_res.all_optimal_schedules, [(4, 4, 2, 0, 0)])
        self.assertAlmostEqual(sol.objective_value, enum_res.optimal_objective, places=5)

    def test_training_capacity_reference_ignores_post_training_weeks(self) -> None:
        """Mutating validation/test weeks must have zero effect on training capacity calibration."""
        dates = pd.bdate_range("2010-09-06", periods=100, freq="B")
        df = pd.DataFrame(
            {
                "operational_date": dates.date,
                "week_idx": np.arange(100) // 5,
                "booked_lines": np.linspace(1000.0, 3000.0, 100),
            }
        )
        ref_clean, util_clean = compute_training_capacity_reference(
            df, initial_train_end_week_idx=14, regime="train_p90_quantile"
        )
        df_mut = df.copy()
        df_mut.loc[df_mut["week_idx"] > 14, "booked_lines"] = 9_999_999.0
        ref_mut, util_mut = compute_training_capacity_reference(
            df_mut, initial_train_end_week_idx=14, regime="train_p90_quantile"
        )
        self.assertEqual(ref_clean, ref_mut)
        self.assertEqual(util_clean, util_mut)

    def test_milp_matches_brute_force_across_random_and_penalty_cases(self) -> None:
        rng = np.random.default_rng(12345)
        for pen in (1.0, 3.0, 5.0):
            for mult in (0.75, 1.0, 1.25):
                p = StaffingParameters(
                    core_capacity_c=700,
                    flex_productivity_q=150,
                    friday_penalty=pen,
                    productivity_multiplier=mult,
                )
                for _ in range(5):
                    fc = rng.uniform(350.0, 1650.0, size=5)
                    sol = solve_weekly_staffing_milp(fc, p, allow_offline_fallback=True)
                    enum_res = enumerate_optimal_schedule(fc, p)
                    self.assertAlmostEqual(
                        sol.objective_value, enum_res.optimal_objective, places=4
                    )
                    self.assertIn(sol.blocks, enum_res.all_optimal_schedules)


if __name__ == "__main__":
    unittest.main()
