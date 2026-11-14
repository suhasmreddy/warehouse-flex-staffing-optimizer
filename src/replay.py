"""Frozen-schedule queue replay, four-policy comparison, sensitivity grid, and block bootstrap."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence

import numpy as np
import pandas as pd

from src.forecast import compute_wape
from src.model import (
    StaffingParameters,
    enumerate_optimal_schedule,
    solve_weekly_staffing_milp,
)

POLICY_ORDER: tuple[str, ...] = (
    "flat",
    "prev_week_milp",
    "ridge_milp",
    "oracle_milp",
)

POLICY_LABELS: dict[str, str] = {
    "flat": "Flat Staffing (2-2-2-2-2)",
    "prev_week_milp": "Previous-Week Forecast + MILP",
    "ridge_milp": "Ridge Forecast + MILP",
    "oracle_milp": "Perfect-Foresight MILP (Oracle)",
}


@dataclass(frozen=True)
class WeeklyReplayOutcome:
    """Realized ex-post queue metrics for one frozen 5-day staffing schedule."""

    blocks: tuple[int, int, int, int, int]
    realized_backlog: tuple[float, float, float, float, float]
    total_line_days: float
    friday_backlog_lines: float
    weekly_objective_cost: float


@dataclass(frozen=True)
class BlockBootstrapResult:
    """Exploratory paired moving-block bootstrap summary for weekly cost differences."""

    comparison_name: str
    n_weeks: int
    block_length_weeks: int
    n_resamples: int
    confidence_level: float
    observed_mean_weekly_diff: float
    observed_total_diff: float
    bootstrap_se_weekly_diff: float
    ci_lower_weekly_diff: float
    ci_upper_weekly_diff: float
    ci_lower_total_diff: float
    ci_upper_total_diff: float
    prob_positive_improvement: float

    def to_dict(self) -> dict:
        return asdict(self)


def replay_weekly_queue(
    frozen_blocks: Sequence[int],
    realized_workload: Sequence[float],
    params: StaffingParameters,
) -> WeeklyReplayOutcome:
    """Independently replay daily Lindley backlog using frozen blocks and realized demand.

    Queue evolution (resetting weekly to ``params.initial_backlog = 0`` on Monday morning):
        b_d = max(0, b_{d-1} + actual_d - C - q_eff * x_d),   d = 0..4
    """
    if len(frozen_blocks) != 5 or len(realized_workload) != 5:
        raise ValueError("Both frozen_blocks and realized_workload must have length 5.")

    blocks_tup = tuple(int(x) for x in frozen_blocks)
    if sum(blocks_tup) != params.weekly_blocks:
        raise ValueError(
            f"Frozen schedule {blocks_tup} sums to {sum(blocks_tup)}, expected {params.weekly_blocks}."
        )
    for d, x_d in enumerate(blocks_tup):
        if x_d < params.min_daily_blocks or x_d > params.max_daily_blocks:
            raise ValueError(f"Frozen block x[{d}]={x_d} outside allowed daily bounds.")

    q_eff = params.effective_q
    c_cap = float(params.core_capacity_c)
    b_prev = float(params.initial_backlog)
    realized_b: list[float] = []

    for d in range(5):
        arrivals = max(0.0, float(realized_workload[d]))
        service = c_cap + q_eff * blocks_tup[d]
        b_prev = max(0.0, b_prev + arrivals - service)
        realized_b.append(b_prev)

    total_ld = float(sum(realized_b))
    fri_b = float(realized_b[4])
    obj_cost = total_ld + float(params.friday_penalty) * fri_b

    return WeeklyReplayOutcome(
        blocks=blocks_tup,  # type: ignore[arg-type]
        realized_backlog=tuple(realized_b),  # type: ignore[arg-type]
        total_line_days=total_ld,
        friday_backlog_lines=fri_b,
        weekly_objective_cost=obj_cost,
    )


def evaluate_four_policies(
    test_predictions_df: pd.DataFrame,
    params: StaffingParameters,
    solver_backend: str = "pyomo_highs",
    allow_offline_fallback: bool = False,
    verify_with_enumerator: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Evaluate Flat, Prev-Week+MILP, Ridge+MILP, and Oracle MILP on identical test weeks.

    Freezes all forecast-driven staffing decisions BEFORE revealing actual realized
    test workload to the queue replay engine.

    Returns
    -------
    summary_df : pd.DataFrame
        Aggregate 4-policy metrics across all held-out test weeks.
    weekly_df : pd.DataFrame
        Week-by-week outcomes for every held-out test week (including weeks where MILP loses).
    daily_schedule_df : pd.DataFrame
        Day-level frozen block assignments and realized backlogs for all 4 policies.
    """
    df = test_predictions_df.sort_values("operational_date").reset_index(drop=True)
    if len(df) % 5 != 0:
        raise ValueError("test_predictions_df must consist of complete 5-day weeks.")

    week_ids = sorted(df["week_idx"].unique().tolist())
    weekly_rows: list[dict] = []
    daily_rows: list[dict] = []
    solver_used_label = "flat_deterministic"

    for test_ord, wk in enumerate(week_ids, start=1):
        wk_slice = df.loc[df["week_idx"] == wk].sort_values("iso_weekday")
        if list(wk_slice["iso_weekday"]) != [1, 2, 3, 4, 5]:
            raise AssertionError(f"Week {wk} does not have Monday..Friday ordering.")

        week_monday = str(wk_slice["week_monday"].iloc[0])
        pred_prev = wk_slice["pred_prev_week"].to_numpy(dtype=float)
        pred_ridge = wk_slice["pred_ridge"].to_numpy(dtype=float)

        # STEP 1: Freeze non-oracle schedules strictly BEFORE reading realized workload
        flat_sched = (2, 2, 2, 2, 2)
        sol_prev = solve_weekly_staffing_milp(
            pred_prev,
            params,
            solver_backend=solver_backend,
            allow_offline_fallback=allow_offline_fallback,
        )
        sol_ridge = solve_weekly_staffing_milp(
            pred_ridge,
            params,
            solver_backend=solver_backend,
            allow_offline_fallback=allow_offline_fallback,
        )
        solver_used_label = sol_ridge.solver_backend

        if verify_with_enumerator:
            enum_ridge = enumerate_optimal_schedule(pred_ridge, params)
            if abs(sol_ridge.objective_value - enum_ridge.optimal_objective) > 1e-5:
                raise AssertionError(
                    f"Enumerator mismatch on week {wk}: MILP={sol_ridge.objective_value} "
                    f"vs brute_force={enum_ridge.optimal_objective}"
                )

        # STEP 2: Reveal actual booked workload for ex-post replay and oracle lower bound
        actual_y = wk_slice["booked_lines"].to_numpy(dtype=float)
        sol_oracle = solve_weekly_staffing_milp(
            actual_y,
            params,
            solver_backend=solver_backend,
            allow_offline_fallback=allow_offline_fallback,
        )

        schedules = {
            "flat": flat_sched,
            "prev_week_milp": sol_prev.blocks,
            "ridge_milp": sol_ridge.blocks,
            "oracle_milp": sol_oracle.blocks,
        }
        replays = {
            pol: replay_weekly_queue(sched, actual_y, params)
            for pol, sched in schedules.items()
        }

        # Mathematical lower-bound invariant: Oracle cost <= every feasible policy's cost
        oracle_cost = replays["oracle_milp"].weekly_objective_cost
        for pol, rep in replays.items():
            if oracle_cost > rep.weekly_objective_cost + 1e-6:
                raise AssertionError(
                    f"Oracle invariant violated in week {wk} vs {pol}: "
                    f"oracle={oracle_cost} > {rep.weekly_objective_cost}"
                )

        wk_row: dict = {
            "test_week_num": test_ord,
            "week_idx": int(wk),
            "week_monday": week_monday,
            "actual_weekly_lines": int(round(float(actual_y.sum()))),
            "prev_week_wape": compute_wape(actual_y, pred_prev),
            "ridge_wape": compute_wape(actual_y, pred_ridge),
        }
        for pol in POLICY_ORDER:
            rep = replays[pol]
            wk_row[f"{pol}_blocks"] = "-".join(str(b) for b in rep.blocks)
            wk_row[f"{pol}_line_days"] = rep.total_line_days
            wk_row[f"{pol}_friday_backlog"] = rep.friday_backlog_lines
            wk_row[f"{pol}_objective_cost"] = rep.weekly_objective_cost

        wk_row["ridge_vs_flat_cost_saving"] = (
            replays["flat"].weekly_objective_cost - replays["ridge_milp"].weekly_objective_cost
        )
        wk_row["prev_vs_flat_cost_saving"] = (
            replays["flat"].weekly_objective_cost - replays["prev_week_milp"].weekly_objective_cost
        )
        wk_row["ridge_vs_prev_cost_saving"] = (
            replays["prev_week_milp"].weekly_objective_cost
            - replays["ridge_milp"].weekly_objective_cost
        )
        wk_row["ridge_outcome_vs_flat"] = (
            "win"
            if wk_row["ridge_vs_flat_cost_saving"] > 1e-6
            else ("loss" if wk_row["ridge_vs_flat_cost_saving"] < -1e-6 else "tie")
        )
        weekly_rows.append(wk_row)

        for d_idx, (_, day_rec) in enumerate(wk_slice.iterrows()):
            d_entry: dict = {
                "test_week_num": test_ord,
                "week_idx": int(wk),
                "week_monday": week_monday,
                "operational_date": str(day_rec["operational_date"]),
                "iso_weekday": int(day_rec["iso_weekday"]),
                "weekday_name": str(day_rec["weekday_name"]),
                "actual_lines": float(day_rec["booked_lines"]),
                "pred_prev_week": float(day_rec["pred_prev_week"]),
                "pred_ridge": float(day_rec["pred_ridge"]),
            }
            for pol in POLICY_ORDER:
                d_entry[f"{pol}_blocks"] = replays[pol].blocks[d_idx]
                d_entry[f"{pol}_backlog"] = replays[pol].realized_backlog[d_idx]
            daily_rows.append(d_entry)

    weekly_df = pd.DataFrame(weekly_rows)
    daily_schedule_df = pd.DataFrame(daily_rows)

    # Aggregate summary across all held-out weeks
    flat_total_obj = float(weekly_df["flat_objective_cost"].sum())
    oracle_total_obj = float(weekly_df["oracle_milp_objective_cost"].sum())
    oracle_max_gain = flat_total_obj - oracle_total_obj

    wape_by_policy = {
        "flat": float("nan"),
        "prev_week_milp": compute_wape(df["booked_lines"], df["pred_prev_week"]),
        "ridge_milp": compute_wape(df["booked_lines"], df["pred_ridge"]),
        "oracle_milp": 0.0,
    }

    summary_rows: list[dict] = []
    for pol in POLICY_ORDER:
        tot_ld = float(weekly_df[f"{pol}_line_days"].sum())
        tot_fri = float(weekly_df[f"{pol}_friday_backlog"].sum())
        tot_obj = float(weekly_df[f"{pol}_objective_cost"].sum())
        mean_obj = float(weekly_df[f"{pol}_objective_cost"].mean())

        rel_imp_vs_flat = (
            ((flat_total_obj - tot_obj) / flat_total_obj) * 100.0
            if flat_total_obj > 0
            else 0.0
        )
        diff_vs_oracle = tot_obj - oracle_total_obj
        oracle_gain_captured_pct = (
            ((flat_total_obj - tot_obj) / oracle_max_gain) * 100.0
            if oracle_max_gain > 0
            else float("nan")
        )

        diffs_vs_flat = weekly_df["flat_objective_cost"] - weekly_df[f"{pol}_objective_cost"]
        wins = int((diffs_vs_flat > 1e-6).sum())
        ties = int((diffs_vs_flat.abs() <= 1e-6).sum())
        losses = int((diffs_vs_flat < -1e-6).sum())

        summary_rows.append(
            {
                "policy_key": pol,
                "policy_label": POLICY_LABELS[pol],
                "forecast_wape_pct": wape_by_policy[pol] * 100.0
                if not np.isnan(wape_by_policy[pol])
                else float("nan"),
                "total_modeled_backlog_line_days": tot_ld,
                "total_friday_backlog_lines": tot_fri,
                "total_objective_cost": tot_obj,
                "mean_weekly_objective_cost": mean_obj,
                "rel_improvement_vs_flat_pct": rel_imp_vs_flat,
                "objective_gap_vs_oracle": diff_vs_oracle,
                "oracle_gain_captured_pct": oracle_gain_captured_pct,
                "weeks_won_vs_flat": wins,
                "weeks_tied_vs_flat": ties,
                "weeks_lost_vs_flat": losses,
                "solver_backend": "deterministic" if pol == "flat" else solver_used_label,
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    return summary_df, weekly_df, daily_schedule_df


def run_sensitivity_grid(
    test_predictions_df: pd.DataFrame,
    base_params: StaffingParameters,
    productivity_multipliers: Sequence[float] = (0.75, 1.0, 1.25),
    friday_penalties: Sequence[float] = (1.0, 3.0, 5.0),
    solver_backend: str = "pyomo_highs",
    allow_offline_fallback: bool = False,
) -> pd.DataFrame:
    """Run 3x3 sensitivity analysis over flex productivity multipliers and Friday penalties."""
    rows: list[dict] = []
    for mult in productivity_multipliers:
        for pen in friday_penalties:
            scen_params = StaffingParameters(
                core_capacity_c=base_params.core_capacity_c,
                flex_productivity_q=base_params.flex_productivity_q,
                weekly_blocks=base_params.weekly_blocks,
                min_daily_blocks=base_params.min_daily_blocks,
                max_daily_blocks=base_params.max_daily_blocks,
                initial_backlog=base_params.initial_backlog,
                friday_penalty=float(pen),
                productivity_multiplier=float(mult),
            )
            summ_df, wk_df, _ = evaluate_four_policies(
                test_predictions_df=test_predictions_df,
                params=scen_params,
                solver_backend=solver_backend,
                allow_offline_fallback=allow_offline_fallback,
                verify_with_enumerator=False,
            )
            by_pol = summ_df.set_index("policy_key")
            flat_obj = float(by_pol.loc["flat", "total_objective_cost"])
            prev_obj = float(by_pol.loc["prev_week_milp", "total_objective_cost"])
            ridge_obj = float(by_pol.loc["ridge_milp", "total_objective_cost"])
            oracle_obj = float(by_pol.loc["oracle_milp", "total_objective_cost"])

            rows.append(
                {
                    "productivity_multiplier": float(mult),
                    "effective_q_lines_per_block": scen_params.effective_q,
                    "friday_penalty": float(pen),
                    "flat_total_cost": flat_obj,
                    "prev_week_milp_total_cost": prev_obj,
                    "ridge_milp_total_cost": ridge_obj,
                    "oracle_milp_total_cost": oracle_obj,
                    "prev_vs_flat_improvement_pct": float(
                        by_pol.loc["prev_week_milp", "rel_improvement_vs_flat_pct"]
                    ),
                    "ridge_vs_flat_improvement_pct": float(
                        by_pol.loc["ridge_milp", "rel_improvement_vs_flat_pct"]
                    ),
                    "ridge_vs_prev_improvement_pct": (
                        ((prev_obj - ridge_obj) / prev_obj) * 100.0 if prev_obj > 0 else 0.0
                    ),
                    "ridge_weeks_lost_vs_flat": int(
                        (wk_df["ridge_vs_flat_cost_saving"] < -1e-6).sum()
                    ),
                }
            )
    return pd.DataFrame(rows)


def paired_moving_block_bootstrap(
    cost_baseline: Sequence[float] | np.ndarray,
    cost_candidate: Sequence[float] | np.ndarray,
    comparison_name: str,
    block_length_weeks: int = 3,
    n_resamples: int = 10000,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> BlockBootstrapResult:
    """Compute an exploratory circular moving-block bootstrap CI on paired weekly savings.

    Positive ``diff_w = cost_baseline[w] - cost_candidate[w]`` means ``candidate``
    reduced weekly objective cost relative to ``baseline``. Circular wrapping ensures
    every held-out week has identical inclusion probability across overlapping blocks.
    """
    base_arr = np.asarray(cost_baseline, dtype=float)
    cand_arr = np.asarray(cost_candidate, dtype=float)
    if base_arr.shape != cand_arr.shape or base_arr.ndim != 1:
        raise ValueError("Paired weekly cost arrays must be 1-D with matching length.")

    n_weeks = len(base_arr)
    if n_weeks < 2:
        raise ValueError("Need at least 2 weeks for moving-block bootstrap.")
    b_len = max(1, min(int(block_length_weeks), n_weeks))

    diffs = base_arr - cand_arr
    obs_mean = float(np.mean(diffs))
    obs_total = float(np.sum(diffs))

    # Circular moving blocks of length b_len so boundary weeks receive equal weight
    doubled = np.concatenate([diffs, diffs[: b_len - 1]])
    blocks = np.array([doubled[i : i + b_len] for i in range(n_weeks)], dtype=float)

    n_blocks_needed = int(np.ceil(n_weeks / b_len))
    rng = np.random.default_rng(seed)
    sampled_idx = rng.integers(0, n_weeks, size=(n_resamples, n_blocks_needed))
    resampled_series = blocks[sampled_idx].reshape(n_resamples, n_blocks_needed * b_len)[
        :, :n_weeks
    ]

    boot_means = resampled_series.mean(axis=1)
    alpha_tail = (1.0 - confidence_level) / 2.0
    ci_lo, ci_hi = np.quantile(boot_means, [alpha_tail, 1.0 - alpha_tail])

    return BlockBootstrapResult(
        comparison_name=comparison_name,
        n_weeks=n_weeks,
        block_length_weeks=b_len,
        n_resamples=int(n_resamples),
        confidence_level=float(confidence_level),
        observed_mean_weekly_diff=obs_mean,
        observed_total_diff=obs_total,
        bootstrap_se_weekly_diff=float(boot_means.std(ddof=1)),
        ci_lower_weekly_diff=float(ci_lo),
        ci_upper_weekly_diff=float(ci_hi),
        ci_lower_total_diff=float(ci_lo * n_weeks),
        ci_upper_total_diff=float(ci_hi * n_weeks),
        prob_positive_improvement=float(np.mean(boot_means > 0.0)),
    )
