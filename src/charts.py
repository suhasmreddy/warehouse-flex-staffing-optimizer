"""Publication-grade Matplotlib visualizations for forecasting and staffing evaluation."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PALETTE = {
    "actual": "#1B263B",
    "prev_week": "#E07A5F",
    "ridge": "#0077B6",
    "flat": "#6C757D",
    "oracle": "#2A9D8F",
    "capacity": "#8D99AE",
    "loss_bg": "#FFE5E5",
}


def _apply_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 11.5,
            "axes.titleweight": "bold",
            "axes.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.dpi": 200,
        }
    )


def plot_predicted_vs_actual(
    daily_schedule_df: pd.DataFrame,
    core_capacity_c: int,
    base_wape: float,
    ridge_wape: float,
    dataset_label: str,
    out_path: str | Path,
) -> Path:
    """Plot 60-weekday test horizon: Actual Booked Lines vs Previous-Week & Ridge Forecasts."""
    _apply_style()
    df = daily_schedule_df.sort_values("operational_date").reset_index(drop=True)
    x = np.arange(len(df))

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(12.6, 6.8), gridspec_kw={"height_ratios": [2.4, 1.1]}, sharex=True
    )

    ax_top.plot(
        x,
        df["actual_lines"],
        color=PALETTE["actual"],
        linewidth=1.9,
        marker="o",
        markersize=3.5,
        label="Actual Booked Order Lines (Proxy Demand)",
    )
    ax_top.plot(
        x,
        df["pred_prev_week"],
        color=PALETTE["prev_week"],
        linewidth=1.4,
        linestyle="--",
        alpha=0.85,
        label=f"Previous-Week Same-Day Baseline (WAPE = {base_wape * 100:.2f}%)",
    )
    ax_top.plot(
        x,
        df["pred_ridge"],
        color=PALETTE["ridge"],
        linewidth=1.8,
        label=f"Nonnegative Ridge Forecast (WAPE = {ridge_wape * 100:.2f}%)",
    )
    ax_top.axhline(
        core_capacity_c,
        color=PALETTE["capacity"],
        linestyle=":",
        linewidth=1.4,
        label=f"Synthetic Core Daily Capacity C = {core_capacity_c:,} lines/day",
    )

    for wk_start in range(0, len(df), 5):
        ax_top.axvline(wk_start - 0.5, color="#E9ECEF", linewidth=0.8, zorder=0)
        ax_bot.axvline(wk_start - 0.5, color="#E9ECEF", linewidth=0.8, zorder=0)

    ymax = float(max(df["actual_lines"].max(), df["pred_prev_week"].max(), df["pred_ridge"].max()))
    ax_top.set_ylim(0, ymax * 1.25)
    ax_top.set_ylabel("Booked Order Lines / Weekday")
    ax_top.set_title(
        f"Sealed 12-Week Test Horizon: Friday-Origin Forecasts vs Realized Booked Lines\n[{dataset_label}]"
    )
    ax_top.legend(loc="upper left", ncol=2, frameon=True, framealpha=0.94, fontsize=8.8)
    ax_top.grid(axis="y", color="#F1F3F5", linewidth=0.8)

    ridge_err = df["pred_ridge"] - df["actual_lines"]
    bar_colors = np.where(ridge_err >= 0, "#4EA8DE", "#E76F51")
    ax_bot.bar(x, ridge_err, color=bar_colors, width=0.72, alpha=0.85)
    ax_bot.axhline(0.0, color="#343A40", linewidth=0.9)
    ax_bot.set_ylabel("Ridge Error\n(Pred - Actual)")
    ax_bot.set_xlabel("Held-Out Test Week Mondays (5 Operational Weekdays per Block)")
    ax_bot.grid(axis="y", color="#F1F3F5", linewidth=0.8)

    monday_positions = list(range(0, len(df), 5))
    monday_labels = [df["operational_date"].iloc[i][5:] for i in monday_positions]
    ax_bot.set_xticks(monday_positions)
    ax_bot.set_xticklabels(monday_labels, rotation=30, ha="right")

    fig.tight_layout()
    target = Path(out_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target, bbox_inches="tight")
    plt.close(fig)
    return target


def plot_flat_vs_optimized_schedules(
    daily_schedule_df: pd.DataFrame,
    dataset_label: str,
    out_path: str | Path,
) -> Path:
    """Plot weekday block allocation profiles and 12-week Ridge MILP schedule heatmap."""
    _apply_style()
    df = daily_schedule_df.copy()
    weekdays = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]

    means = df.groupby("iso_weekday")[
        ["flat_blocks", "prev_week_milp_blocks", "ridge_milp_blocks", "oracle_milp_blocks"]
    ].mean()

    fig, (ax_left, ax_right) = plt.subplots(
        1, 2, figsize=(13.2, 5.6), gridspec_kw={"width_ratios": [1.1, 1.35], "wspace": 0.28}
    )

    idx = np.arange(5)
    w = 0.19
    ax_left.bar(idx - 1.5 * w, means["flat_blocks"], width=w, color=PALETTE["flat"], label="Flat (2-2-2-2-2)")
    ax_left.bar(
        idx - 0.5 * w,
        means["prev_week_milp_blocks"],
        width=w,
        color=PALETTE["prev_week"],
        label="Prev-Week + MILP",
    )
    ax_left.bar(
        idx + 0.5 * w,
        means["ridge_milp_blocks"],
        width=w,
        color=PALETTE["ridge"],
        label="Ridge + MILP",
    )
    ax_left.bar(
        idx + 1.5 * w,
        means["oracle_milp_blocks"],
        width=w,
        color=PALETTE["oracle"],
        label="Oracle MILP",
    )
    ax_left.set_xticks(idx)
    ax_left.set_xticklabels([d[:3] for d in weekdays])
    ax_left.set_ylim(0, 4.55)
    ax_left.set_ylabel("Mean Assigned 4-Hour Blocks / Day")
    ax_left.set_title("Mean Weekday Block Allocation\n(10 Blocks = 40 Flex Hours / Week)")
    ax_left.legend(loc="upper right", fontsize=8.8, frameon=True)
    ax_left.grid(axis="y", color="#F1F3F5", linewidth=0.8)

    piv = df.pivot(index="test_week_num", columns="iso_weekday", values="ridge_milp_blocks")
    mat = piv.to_numpy(dtype=int)
    im = ax_right.imshow(mat, cmap="Blues", vmin=0, vmax=4, aspect="auto")

    for r in range(mat.shape[0]):
        for c in range(mat.shape[1]):
            val = mat[r, c]
            txt_color = "white" if val >= 3 else "#1B263B"
            ax_right.text(c, r, f"{val}", ha="center", va="center", color=txt_color, fontweight="bold")

    wk_mondays = df.drop_duplicates("test_week_num").sort_values("test_week_num")["week_monday"].tolist()
    ax_right.set_xticks(range(5))
    ax_right.set_xticklabels([d[:3] for d in weekdays])
    ax_right.set_yticks(range(len(wk_mondays)))
    ax_right.set_yticklabels([f"W{i + 1} ({m[5:]})" for i, m in enumerate(wk_mondays)], fontsize=8.8)
    ax_right.set_title("Frozen Ridge + MILP Weekly Schedule Matrix\n(Cell = 4-Hour Blocks; Each Row Sums to 10)")
    cbar = fig.colorbar(im, ax=ax_right, ticks=[0, 1, 2, 3, 4], pad=0.03)
    cbar.set_label("Assigned Blocks")

    fig.suptitle(f"Flex Labor Shift Allocation Comparison — {dataset_label}", fontweight="bold", fontsize=12.5, y=0.99)
    fig.subplots_adjust(top=0.84)
    target = Path(out_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target, bbox_inches="tight")
    plt.close(fig)
    return target


def plot_weekly_backlog_costs(
    weekly_df: pd.DataFrame,
    dataset_label: str,
    out_path: str | Path,
) -> Path:
    """Plot week-by-week realized objective costs for all 4 policies, highlighting MILP losses."""
    _apply_style()
    df = weekly_df.sort_values("test_week_num").reset_index(drop=True)
    n_wks = len(df)
    idx = np.arange(n_wks)
    w = 0.20

    fig, ax = plt.subplots(figsize=(12.6, 5.6))

    for i, row in df.iterrows():
        if row["ridge_outcome_vs_flat"] == "loss":
            ax.axvspan(i - 0.45, i + 0.45, color=PALETTE["loss_bg"], alpha=0.65, zorder=0)

    ax.bar(idx - 1.5 * w, df["flat_objective_cost"], width=w, color=PALETTE["flat"], label="Flat Staffing (2-2-2-2-2)")
    ax.bar(
        idx - 0.5 * w,
        df["prev_week_milp_objective_cost"],
        width=w,
        color=PALETTE["prev_week"],
        label="Prev-Week Forecast + MILP",
    )
    ax.bar(
        idx + 0.5 * w,
        df["ridge_milp_objective_cost"],
        width=w,
        color=PALETTE["ridge"],
        label="Ridge Forecast + MILP",
    )
    ax.bar(
        idx + 1.5 * w,
        df["oracle_milp_objective_cost"],
        width=w,
        color=PALETTE["oracle"],
        label="Perfect-Foresight MILP (Oracle Lower Bound)",
    )

    max_cost = float(
        df[
            [
                "flat_objective_cost",
                "prev_week_milp_objective_cost",
                "ridge_milp_objective_cost",
                "oracle_milp_objective_cost",
            ]
        ]
        .max()
        .max()
    )
    ax.set_ylim(0, max_cost * 1.24)

    labels = [f"W{int(r.test_week_num)}\n{r.week_monday[5:]}" for r in df.itertuples()]
    ax.set_xticks(idx)
    ax.set_xticklabels(labels, fontsize=8.8)
    ax.set_ylabel("Weekly Objective Cost: sum(b[d]) + 3 * b[Fri]  [weighted line-days]")
    ax.set_title(
        f"Held-Out Weekly Modeled Backlog Objective by Policy [{dataset_label}]\n"
        "(Pink-shaded weeks indicate forecast misses where Ridge+MILP lost to Flat Staffing)"
    )
    ax.legend(loc="upper left", ncol=2, frameon=True, framealpha=0.94, fontsize=9)
    ax.grid(axis="y", color="#F1F3F5", linewidth=0.8)

    fig.tight_layout()
    target = Path(out_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target, bbox_inches="tight")
    plt.close(fig)
    return target


def plot_sensitivity_results(
    sensitivity_df: pd.DataFrame,
    dataset_label: str,
    out_path: str | Path,
) -> Path:
    """Plot 3x3 sensitivity grid across productivity multipliers {0.75,1.0,1.25} and Friday penalties {1,3,5}."""
    _apply_style()
    mults = sorted(sensitivity_df["productivity_multiplier"].unique().tolist())
    pens = sorted(sensitivity_df["friday_penalty"].unique().tolist())

    piv_ridge = sensitivity_df.pivot(
        index="productivity_multiplier",
        columns="friday_penalty",
        values="ridge_vs_flat_improvement_pct",
    ).loc[mults, pens]
    piv_prev = sensitivity_df.pivot(
        index="productivity_multiplier",
        columns="friday_penalty",
        values="prev_vs_flat_improvement_pct",
    ).loc[mults, pens]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.2, 4.9), sharey=True)

    vmin = min(float(piv_ridge.min().min()), float(piv_prev.min().min()), 0.0)
    vmax = max(float(piv_ridge.max().max()), float(piv_prev.max().max()), 5.0)

    for ax, piv, title in (
        (ax1, piv_prev, "Prev-Week + MILP\n% Objective Reduction vs Flat"),
        (ax2, piv_ridge, "Ridge + MILP\n% Objective Reduction vs Flat"),
    ):
        arr = piv.to_numpy(dtype=float)
        im = ax.imshow(arr, cmap="YlGnBu", vmin=vmin, vmax=vmax, aspect="auto")
        for r in range(arr.shape[0]):
            for c in range(arr.shape[1]):
                val = arr[r, c]
                txt_col = "white" if val > (vmin + 0.58 * (vmax - vmin)) else "#1B263B"
                ax.text(c, r, f"{val:+.1f}%", ha="center", va="center", color=txt_col, fontweight="bold")
        ax.set_xticks(range(len(pens)))
        ax.set_xticklabels([f"lambda_Fri = {int(p)}" for p in pens])
        ax.set_yticks(range(len(mults)))
        ax.set_yticklabels([f"q x {m:.2f}" for m in mults])
        ax.set_title(title)
        ax.spines["top"].set_visible(True)
        ax.spines["right"].set_visible(True)

    ax1.set_ylabel("Flex Productivity Multiplier (m)")
    fig.suptitle(
        f"3x3 Sensitivity Grid Across Flex Productivity & Friday Penalty\n[{dataset_label}]",
        fontweight="bold",
        y=1.03,
    )
    cbar = fig.colorbar(im, ax=[ax1, ax2], fraction=0.03, pad=0.04)
    cbar.set_label("% Cost Reduction vs Flat Staffing")

    target = Path(out_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target, bbox_inches="tight")
    plt.close(fig)
    return target


def plot_capacity_regimes(
    regimes_df: pd.DataFrame,
    dataset_label: str,
    out_path: str | Path,
) -> Path:
    """Plot cost reduction vs Flat and schedule diversity across training-calibrated capacity regimes."""
    _apply_style()
    df = regimes_df.copy().reset_index(drop=True)
    short_labels = [
        f"Train P50\nC={int(r.core_capacity_C):,}\nq={int(r.flex_productivity_q):,}"
        if r.regime_key == "train_p50_median"
        else (
            f"2010 Q4 Med\nC={int(r.core_capacity_C):,}\nq={int(r.flex_productivity_q):,}"
            if r.regime_key == "train_q4_peak_median"
            else (
                f"Train P90*\nC={int(r.core_capacity_C):,}\nq={int(r.flex_productivity_q):,}"
                if r.regime_key == "train_p90_quantile"
                else f"Train P95\nC={int(r.core_capacity_C):,}\nq={int(r.flex_productivity_q):,}"
            )
        )
        for r in df.itertuples()
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.2, 5.4), gridspec_kw={"wspace": 0.26})
    x = np.arange(len(df))
    w = 0.25

    ax1.bar(
        x - w,
        df["prev_vs_flat_improvement_pct"],
        width=w,
        color=PALETTE["prev_week"],
        label="Prev-Week + MILP",
    )
    ax1.bar(
        x,
        df["ridge_vs_flat_improvement_pct"],
        width=w,
        color=PALETTE["ridge"],
        label="Ridge + MILP",
    )
    ax1.bar(
        x + w,
        df["oracle_vs_flat_improvement_pct"],
        width=w,
        color=PALETTE["oracle"],
        label="Oracle MILP (Ceiling)",
    )
    ax1.axhline(0.0, color="#343A40", linewidth=0.9)
    ax1.set_xticks(x)
    ax1.set_xticklabels(short_labels, fontsize=9)
    ax1.set_ylim(0, float(df["oracle_vs_flat_improvement_pct"].max()) * 1.20)
    ax1.set_ylabel("% Objective Cost Reduction vs Flat Staffing")
    ax1.set_title("Policy Improvement Across Training Capacity Tiers")
    ax1.legend(loc="upper left", fontsize=8.8, frameon=True)
    ax1.grid(axis="y", color="#F1F3F5", linewidth=0.8)

    ax2.bar(
        x - w / 2,
        df["unique_prev_schedules"],
        width=w,
        color=PALETTE["prev_week"],
        label="Prev-Week + MILP",
    )
    ax2.bar(
        x + w / 2,
        df["unique_ridge_schedules"],
        width=w,
        color=PALETTE["ridge"],
        label="Ridge + MILP",
    )
    ax2.axhline(1.0, color="#D62828", linestyle="--", linewidth=1.1, label="Degenerate Corner (1 Schedule)")
    ax2.set_xticks(x)
    ax2.set_xticklabels(short_labels, fontsize=9)
    ax2.set_ylim(0, 12.5)
    ax2.set_ylabel("Distinct Weekly Schedules Chosen (Out of 12 Weeks)")
    ax2.set_title("Schedule Responsiveness to Forecasts\n(Breaking the P50 4-4-2-0-0 Saturation Degeneracy)")
    ax2.legend(loc="upper left", fontsize=8.8, frameon=True)
    ax2.grid(axis="y", color="#F1F3F5", linewidth=0.8)

    fig.suptitle(
        f"Training-Calibrated Capacity Regimes (Zero Test-Set Tuning; *Primary = Train P90) — {dataset_label}",
        fontweight="bold",
        fontsize=12.0,
        y=0.99,
    )
    fig.subplots_adjust(top=0.84, bottom=0.16)
    target = Path(out_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target, bbox_inches="tight")
    plt.close(fig)
    return target
