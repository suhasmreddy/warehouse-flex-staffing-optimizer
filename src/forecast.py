"""Friday-origin daily workload forecasting (Previous-Week baseline & Nonnegative Ridge)."""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
from scipy import linalg

from src import calendar as ops_calendar

CALENDAR_FEATURE_COLS: tuple[str, ...] = (
    "wd_mon",
    "wd_tue",
    "wd_wed",
    "wd_thu",
    "wd_fri",
    "sin_annual_1",
    "cos_annual_1",
    "sin_annual_2",
    "cos_annual_2",
    "is_q4_peak",
    "is_xmas_shutdown_window",
    "is_uk_bank_holiday",
    "is_post_holiday_weekday",
)

LAG_FEATURE_COLS: tuple[str, ...] = (
    "lag_1w",
    "lag_4w",
    "lag_52w",
    "prev_week_mean",
    "prev_4w_mean",
)

RIDGE_FEATURE_COLS: tuple[str, ...] = LAG_FEATURE_COLS + CALENDAR_FEATURE_COLS


def compute_wape(y_true: Sequence[float] | np.ndarray, y_pred: Sequence[float] | np.ndarray) -> float:
    """Compute Weighted Absolute Percentage Error: sum(|y - y_hat|) / sum(y).

    Returns ``np.nan`` if ``sum(y_true) == 0`` (e.g., synthetic zero-demand unit test).
    """
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    if yt.shape != yp.shape:
        raise ValueError(f"Shape mismatch in WAPE: {yt.shape} vs {yp.shape}.")
    denom = float(np.sum(np.abs(yt)))
    if denom == 0.0:
        return float("nan")
    return float(np.sum(np.abs(yt - yp)) / denom)


def build_supervised_workload_frame(daily_df: pd.DataFrame) -> pd.DataFrame:
    """Build strictly Friday-available lag and calendar features for each Mon-Fri day.

    Anti-leakage invariants enforced:
    1. Daily spine is contiguous Mon-Fri (5 rows per complete week), verified by checking
       that ``shift(5 * k)`` equals an exact ``7 * k`` calendar-day offset for k in {1, 4, 52}.
    2. Every feature for week ``w`` (Mon..Fri) uses workload strictly on or before Friday
       of week ``w - 1`` plus deterministic calendar constants of the target date.
    """
    if len(daily_df) % 5 != 0:
        raise ValueError("daily_df must contain complete 5-day Monday-Friday weeks.")

    df = ops_calendar.add_deterministic_calendar_features(daily_df, date_col="operational_date")
    df = df.sort_values("operational_date").reset_index(drop=True)

    # Verify exact calendar-day alignment of 1-, 4-, and 52-week weekday shifts
    dates = pd.Series(df["operational_date"])
    for k_weeks in (1, 4, 52):
        shift_rows = 5 * k_weeks
        if len(df) > shift_rows:
            deltas = dates.iloc[shift_rows:].to_numpy() - dates.iloc[:-shift_rows].to_numpy()
            expected = dt.timedelta(days=7 * k_weeks)
            if not all(d == expected for d in deltas):
                raise AssertionError(
                    f"Calendar spine misaligned for {k_weeks}-week lag; expected {expected}."
                )

    df["lag_1w"] = df["booked_lines"].shift(5).astype(float)
    df["lag_4w"] = df["booked_lines"].shift(20).astype(float)
    df["lag_52w"] = df["booked_lines"].shift(260).astype(float)

    # Weekly aggregates strictly through Friday of week w - 1
    weekly_means = df.groupby("week_idx", sort=True)["booked_lines"].mean()
    prev_wk_mean_map = weekly_means.shift(1)
    prev_4wk_mean_map = weekly_means.shift(1).rolling(window=4, min_periods=4).mean()

    df["prev_week_mean"] = df["week_idx"].map(prev_wk_mean_map).astype(float)
    df["prev_4w_mean"] = df["week_idx"].map(prev_4wk_mean_map).astype(float)

    # Record the exact prediction-origin Friday date (3 days before week_monday)
    df["forecast_origin_friday"] = [
        wm - dt.timedelta(days=3) for wm in df["week_monday"]
    ]
    return df


class NonnegativeRidgeForecaster:
    """Standardized L2-penalized linear forecaster with unpenalized intercept and >=0 clipping.

    Uses ``sklearn.linear_model.Ridge`` when installed, or the mathematically identical
    closed-form normal equations ``(X_s^T X_s + alpha I) beta = X_s^T (y - y_bar)`` via
    ``scipy.linalg.solve`` in environments without ``scikit-learn``.
    """

    def __init__(self, alpha: float = 10.0, clip_nonnegative: bool = True) -> None:
        if alpha <= 0:
            raise ValueError(f"Ridge regularization alpha must be positive (got {alpha}).")
        self.alpha = float(alpha)
        self.clip_nonnegative = bool(clip_nonnegative)
        self.feature_mean_: np.ndarray | None = None
        self.feature_scale_: np.ndarray | None = None
        self.coef_: np.ndarray | None = None
        self.intercept_: float | None = None
        self.backend_used_: str = "unfitted"

    def fit(self, X: np.ndarray, y: np.ndarray) -> "NonnegativeRidgeForecaster":
        X_arr = np.asarray(X, dtype=float)
        y_arr = np.asarray(y, dtype=float)
        if X_arr.ndim != 2 or len(X_arr) != len(y_arr) or len(X_arr) == 0:
            raise ValueError("Invalid training matrices passed to NonnegativeRidgeForecaster.fit.")

        mean = X_arr.mean(axis=0)
        std = X_arr.std(axis=0, ddof=0)
        scale = np.where(std > 1e-12, std, 1.0)
        X_std = (X_arr - mean) / scale

        try:
            from sklearn.linear_model import Ridge  # type: ignore[import-not-found]

            reg = Ridge(alpha=self.alpha, fit_intercept=True, solver="cholesky")
            reg.fit(X_std, y_arr)
            self.coef_ = np.asarray(reg.coef_, dtype=float)
            self.intercept_ = float(reg.intercept_)
            self.backend_used_ = "scikit-learn"
        except ImportError:
            y_mean = float(y_arr.mean())
            y_centered = y_arr - y_mean
            n_features = X_std.shape[1]
            gram = X_std.T @ X_std + self.alpha * np.eye(n_features, dtype=float)
            rhs = X_std.T @ y_centered
            self.coef_ = linalg.solve(gram, rhs, assume_a="pos")
            self.intercept_ = y_mean
            self.backend_used_ = "scipy_closed_form"

        self.feature_mean_ = mean
        self.feature_scale_ = scale
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.coef_ is None or self.feature_mean_ is None or self.feature_scale_ is None:
            raise RuntimeError("Forecaster must be fitted before calling predict().")
        X_arr = np.asarray(X, dtype=float)
        X_std = (X_arr - self.feature_mean_) / self.feature_scale_
        raw = X_std @ self.coef_ + float(self.intercept_)
        if self.clip_nonnegative:
            return np.maximum(0.0, raw)
        return raw


@dataclass(frozen=True)
class ChronologicalForecastOutput:
    """Container for chronological split boundaries, selected alpha, and out-of-sample predictions."""

    total_complete_weeks: int
    initial_train_end_week_idx: int
    validation_week_indices: list[int]
    test_week_indices: list[int]
    initial_train_median_daily_lines: float
    selected_ridge_alpha: float
    validation_alpha_wape: dict[float, float]
    validation_baseline_wape: float
    validation_ridge_wape: float
    test_baseline_wape: float
    test_ridge_wape: float
    ridge_backend_used: str
    validation_predictions: pd.DataFrame
    test_predictions: pd.DataFrame


def run_chronological_forecasting(
    daily_df: pd.DataFrame,
    ridge_alphas: Sequence[float] = (0.01, 0.1, 1.0, 5.0, 10.0, 25.0, 50.0, 100.0, 250.0, 500.0, 1000.0),
    min_lag_weeks: int = 52,
    validation_weeks: int = 8,
    test_weeks: int = 12,
    feature_cols: Sequence[str] = RIDGE_FEATURE_COLS,
) -> ChronologicalForecastOutput:
    """Execute strict Friday-origin rolling validation and sealed 12-week test forecasting.

    Timeline structure over ``W`` complete weeks (indexed ``0 .. W - 1``):
    - ``0 .. W - (validation_weeks + test_weeks) - 1``: Initial training set (used for
      calibrating ``C`` and ``q`` and warm-starting 52-week lags).
    - ``W - (validation_weeks + test_weeks) .. W - test_weeks - 1``: 8-week rolling
      validation set (used solely to select ``alpha*`` minimizing validation WAPE).
    - ``W - test_weeks .. W - 1``: Final 12 complete weeks reserved as the sealed test set.
    """
    sup_df = build_supervised_workload_frame(daily_df)
    total_weeks = int(sup_df["week_idx"].nunique())
    min_required_weeks = min_lag_weeks + 4 + validation_weeks + test_weeks
    if total_weeks < min_required_weeks:
        raise ValueError(
            f"Dataset has {total_weeks} complete weeks, but chronological 52-week lag + "
            f"initial fit + {validation_weeks} val + {test_weeks} test requires >= {min_required_weeks} weeks."
        )

    val_start_wk = total_weeks - (validation_weeks + test_weeks)
    test_start_wk = total_weeks - test_weeks
    init_train_end_wk = val_start_wk - 1

    val_weeks_list = list(range(val_start_wk, test_start_wk))
    test_weeks_list = list(range(test_start_wk, total_weeks))

    # Strictly initial-training daily workload median (zero validation/test leakage)
    init_train_slice = sup_df.loc[sup_df["week_idx"] <= init_train_end_wk, "booked_lines"]
    init_train_median = float(np.median(init_train_slice.to_numpy(dtype=float)))

    feat_list = list(feature_cols)

    def _rolling_predict_weeks(target_weeks: list[int], alpha: float) -> tuple[pd.DataFrame, str]:
        records: list[pd.DataFrame] = []
        backend_name = "unknown"
        for wk in target_weeks:
            target_rows = sup_df.loc[sup_df["week_idx"] == wk].copy()
            origin_friday = target_rows["forecast_origin_friday"].iloc[0]

            # Strict anti-leakage filter: training rows must have operational_date <= origin_friday
            # and complete 52-week lag features.
            train_rows = sup_df.loc[
                (sup_df["week_idx"] >= min_lag_weeks)
                & (sup_df["operational_date"] <= origin_friday)
            ]
            if train_rows.empty:
                raise RuntimeError(f"Empty lag-complete training fold prior to week {wk}.")
            if train_rows["operational_date"].max() > origin_friday:
                raise AssertionError("Future data leakage detected in training fold!")
            if train_rows[feat_list].isna().any().any():
                raise AssertionError(f"NaN lag feature found in training fold before week {wk}.")
            if target_rows[feat_list].isna().any().any():
                raise AssertionError(f"NaN lag feature found in target week {wk}.")

            X_train = train_rows[feat_list].to_numpy(dtype=float)
            y_train = train_rows["booked_lines"].to_numpy(dtype=float)
            X_target = target_rows[feat_list].to_numpy(dtype=float)

            model = NonnegativeRidgeForecaster(alpha=alpha, clip_nonnegative=True)
            model.fit(X_train, y_train)
            backend_name = model.backend_used_

            target_rows["pred_prev_week"] = np.maximum(0.0, target_rows["lag_1w"].to_numpy(dtype=float))
            target_rows["pred_ridge"] = model.predict(X_target)
            records.append(target_rows)

        return pd.concat(records, ignore_index=True), backend_name

    alpha_wape_map: dict[float, float] = {}
    val_frames_by_alpha: dict[float, pd.DataFrame] = {}
    ridge_backend = "unknown"

    for alpha in ridge_alphas:
        val_pred_df, ridge_backend = _rolling_predict_weeks(val_weeks_list, float(alpha))
        wape_val = compute_wape(val_pred_df["booked_lines"], val_pred_df["pred_ridge"])
        alpha_wape_map[float(alpha)] = wape_val
        val_frames_by_alpha[float(alpha)] = val_pred_df

    best_alpha = min(alpha_wape_map, key=lambda a: alpha_wape_map[a])
    best_val_df = val_frames_by_alpha[best_alpha]
    val_base_wape = compute_wape(best_val_df["booked_lines"], best_val_df["pred_prev_week"])
    val_ridge_wape = alpha_wape_map[best_alpha]

    # Sealed 12-week test evaluation with locked alpha*
    test_pred_df, ridge_backend = _rolling_predict_weeks(test_weeks_list, best_alpha)
    test_base_wape = compute_wape(test_pred_df["booked_lines"], test_pred_df["pred_prev_week"])
    test_ridge_wape = compute_wape(test_pred_df["booked_lines"], test_pred_df["pred_ridge"])

    keep_cols = [
        "operational_date",
        "week_monday",
        "week_idx",
        "iso_weekday",
        "weekday_name",
        "forecast_origin_friday",
        "booked_lines",
        "weekday_direct_lines",
        "weekend_folded_lines",
        "pred_prev_week",
        "pred_ridge",
    ]
    return ChronologicalForecastOutput(
        total_complete_weeks=total_weeks,
        initial_train_end_week_idx=init_train_end_wk,
        validation_week_indices=val_weeks_list,
        test_week_indices=test_weeks_list,
        initial_train_median_daily_lines=init_train_median,
        selected_ridge_alpha=best_alpha,
        validation_alpha_wape=alpha_wape_map,
        validation_baseline_wape=val_base_wape,
        validation_ridge_wape=val_ridge_wape,
        test_baseline_wape=test_base_wape,
        test_ridge_wape=test_ridge_wape,
        ridge_backend_used=ridge_backend,
        validation_predictions=best_val_df[keep_cols].reset_index(drop=True),
        test_predictions=test_pred_df[keep_cols].reset_index(drop=True),
    )
