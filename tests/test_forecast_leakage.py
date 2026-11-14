"""Tests for Friday-origin lag alignment, nonnegative clipping, and strict anti-leakage."""

from __future__ import annotations

import datetime as dt
import unittest

import numpy as np

from src import calendar as ops_calendar
from src.forecast import (
    NonnegativeRidgeForecaster,
    build_supervised_workload_frame,
    compute_wape,
    run_chronological_forecasting,
)


def _make_synthetic_daily_spine(n_weeks: int = 80, seed: int = 7) -> "pd.DataFrame":
    rng = np.random.default_rng(seed)
    first_mon = dt.date(2010, 1, 4)
    last_fri = first_mon + dt.timedelta(days=7 * n_weeks - 3)
    spine = ops_calendar.build_weekday_spine(first_mon, last_fri)

    wd_pattern = np.array([120.0, 95.0, 100.0, 130.0, 85.0])
    vals = []
    for wk in range(n_weeks):
        seasonal = 1.0 + 0.25 * np.sin(2.0 * np.pi * wk / 52.0)
        noise = rng.normal(0.0, 8.0, size=5)
        vals.extend(np.maximum(0.0, np.round(wd_pattern * seasonal + noise)).astype(int))

    spine["booked_lines"] = vals
    spine["weekday_direct_lines"] = vals
    spine["weekend_folded_lines"] = 0
    spine["unique_invoices"] = np.maximum(1, np.asarray(vals) // 5)
    spine["unique_skus"] = vals
    return spine


class TestForecastLeakage(unittest.TestCase):
    def test_future_mutation_causes_zero_leakage(self) -> None:
        """Corrupting future test weeks by 1000x must not alter earlier Friday forecasts."""
        base_df = _make_synthetic_daily_spine(n_weeks=80, seed=11)
        out_clean = run_chronological_forecasting(
            base_df, validation_weeks=8, test_weeks=12, min_lag_weeks=52
        )

        # Mutate the final 6 test weeks (weeks 74..79) with massive synthetic spikes
        mutated_df = base_df.copy()
        mutated_df.loc[mutated_df["week_idx"] >= 74, "booked_lines"] = 999_999

        out_mutated = run_chronological_forecasting(
            mutated_df, validation_weeks=8, test_weeks=12, min_lag_weeks=52
        )

        # Initial-training median and validation-selected alpha must be completely unchanged
        self.assertEqual(
            out_clean.initial_train_median_daily_lines,
            out_mutated.initial_train_median_daily_lines,
        )
        self.assertEqual(out_clean.selected_ridge_alpha, out_mutated.selected_ridge_alpha)

        # Predictions for test weeks 68..74 (including week 74 itself, predicted on Friday of week 73!)
        # must be bit-for-bit identical!
        clean_early = out_clean.test_predictions.loc[out_clean.test_predictions["week_idx"] <= 74]
        mut_early = out_mutated.test_predictions.loc[out_mutated.test_predictions["week_idx"] <= 74]

        np.testing.assert_allclose(
            clean_early["pred_prev_week"].to_numpy(),
            mut_early["pred_prev_week"].to_numpy(),
            rtol=0.0,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            clean_early["pred_ridge"].to_numpy(),
            mut_early["pred_ridge"].to_numpy(),
            rtol=0.0,
            atol=1e-12,
        )

    def test_exact_lag_alignment_and_nonnegative_clipping(self) -> None:
        df = _make_synthetic_daily_spine(n_weeks=60, seed=3)
        sup = build_supervised_workload_frame(df)
        row_260 = sup.iloc[260]
        self.assertEqual(row_260["lag_1w"], sup.iloc[255]["booked_lines"])
        self.assertEqual(row_260["lag_4w"], sup.iloc[240]["booked_lines"])
        self.assertEqual(row_260["lag_52w"], sup.iloc[0]["booked_lines"])

        # Force negative linear extrapolation and verify >= 0 clipping
        X_train = np.array([[0.0], [1.0], [2.0], [3.0]])
        y_train = np.array([30.0, 20.0, 10.0, 0.0])
        reg = NonnegativeRidgeForecaster(alpha=0.01, clip_nonnegative=True).fit(X_train, y_train)
        preds = reg.predict(np.array([[10.0], [20.0]]))
        self.assertTrue(np.all(preds == 0.0))

    def test_wape_computation_and_zero_guard(self) -> None:
        self.assertAlmostEqual(compute_wape([100, 200], [110, 170]), 40.0 / 300.0)
        self.assertTrue(np.isnan(compute_wape([0, 0, 0], [5, 0, 2])))


if __name__ == "__main__":
    unittest.main()
