"""End-to-end command-line driver for Warehouse Flex Staffing & Demand Forecasting Optimizer."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.request
import zipfile
from pathlib import Path

# Guard against running `python src/cli.py` directly, which would put `src/`
# at sys.path[0] and shadow Python's stdlib `calendar` module.
_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent
if str(_THIS_DIR) in sys.path:
    sys.path.remove(str(_THIS_DIR))
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import yaml

from src.charts import (
    plot_flat_vs_optimized_schedules,
    plot_predicted_vs_actual,
    plot_sensitivity_results,
    plot_weekly_backlog_costs,
)
from src.forecast import run_chronological_forecasting
from src.model import StaffingParameters, enumerate_optimal_schedule, solve_weekly_staffing_milp
from src.replay import (
    evaluate_four_policies,
    paired_moving_block_bootstrap,
    run_sensitivity_grid,
)
from src.workload import aggregate_daily_workload, load_excel_worksheets


def load_config(config_path: str | Path) -> dict:
    with Path(config_path).open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_uci_workbook(cfg: dict, target_override: str | None = None) -> Path:
    """Download UCI Online Retail II (ID 502) zip archive and extract `online_retail_II.xlsx`."""
    excel_path = Path(target_override or cfg["dataset"]["raw_excel_path"])
    if excel_path.exists() and excel_path.stat().st_size > 1_000_000:
        print(f"[download] Existing UCI workbook found at {excel_path} ({excel_path.stat().st_size:,} bytes).")
        return excel_path

    excel_path.parent.mkdir(parents=True, exist_ok=True)
    zip_url = cfg["dataset"]["uci_zip_url"]
    zip_tmp = excel_path.parent / "online_retail_ii_502.zip"

    print(f"[download] Fetching UCI Online Retail II (ID 502) from {zip_url} ...")
    urllib.request.urlretrieve(zip_url, zip_tmp)

    with zipfile.ZipFile(zip_tmp, "r") as zf:
        xlsx_members = [n for n in zf.namelist() if n.lower().endswith(".xlsx")]
        if not xlsx_members:
            raise RuntimeError(f"No .xlsx file found in downloaded archive {zip_tmp}: {zf.namelist()}")
        with zf.open(xlsx_members[0]) as src_f, excel_path.open("wb") as dst_f:
            dst_f.write(src_f.read())

    zip_tmp.unlink(missing_ok=True)
    print(f"[download] Extracted {excel_path} ({excel_path.stat().st_size:,} bytes).")
    return excel_path


def run_pipeline(
    config_path: str | Path = "configs/base.yml",
    excel_path_override: str | None = None,
    output_dir: str | Path = "results",
    dataset_mode: str = "uci_online_retail_ii",
    allow_offline_fallback: bool = False,
    download_if_missing: bool = False,
) -> dict:
    """Execute full cleaning, forecasting, MILP staffing, replay, sensitivity, bootstrap, and plotting."""
    cfg = load_config(config_path)
    out_dir = Path(output_dir)
    fig_dir = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    excel_path = Path(excel_path_override or cfg["dataset"]["raw_excel_path"])
    if download_if_missing and not excel_path.exists():
        download_uci_workbook(cfg, str(excel_path))

    print(f"[1/6] Loading Excel worksheets from {excel_path} ...")
    raw_df = load_excel_worksheets(excel_path, worksheets=cfg["dataset"]["worksheets"])

    print("[2/6] Executing SQL workload cleaning & weekend-to-Monday aggregation ...")
    daily_df, cleaning_audit = aggregate_daily_workload(
        raw_df=raw_df,
        sql_path=_REPO_ROOT / "sql" / "workload.sql",
        allow_sqlite_fallback=allow_offline_fallback,
    )
    daily_df.to_csv(out_dir / "daily_workload.csv", index=False)

    print("[3/6] Running chronological Friday-origin forecasting (8-wk val, 12-wk sealed test) ...")
    fc_out = run_chronological_forecasting(
        daily_df=daily_df,
        ridge_alphas=cfg["forecasting"]["ridge_alphas"],
        min_lag_weeks=int(cfg["splits"]["min_lag_weeks"]),
        validation_weeks=int(cfg["splits"]["validation_weeks"]),
        test_weeks=int(cfg["splits"]["test_weeks"]),
    )
    fc_out.validation_predictions.to_csv(out_dir / "validation_predictions.csv", index=False)
    fc_out.test_predictions.to_csv(out_dir / "test_predictions.csv", index=False)

    opt_cfg = cfg["optimization"]
    params = StaffingParameters.from_initial_training_median(
        median_daily_lines=fc_out.initial_train_median_daily_lines,
        core_fraction=float(opt_cfg["core_capacity_fraction"]),
        flex_fraction=float(opt_cfg["flex_productivity_fraction"]),
        weekly_blocks=int(opt_cfg["weekly_blocks"]),
        min_daily_blocks=int(opt_cfg["min_daily_blocks"]),
        max_daily_blocks=int(opt_cfg["max_daily_blocks"]),
        initial_backlog=float(opt_cfg["initial_weekly_backlog"]),
        friday_penalty=float(opt_cfg["friday_penalty"]),
        productivity_multiplier=1.0,
    )

    print(
        f"[4/6] Solving 5-day MILP (C={params.core_capacity_c}, q={params.flex_productivity_q}) "
        "& replaying frozen schedules across 4 policies ..."
    )
    summary_df, weekly_df, daily_sched_df = evaluate_four_policies(
        test_predictions_df=fc_out.test_predictions,
        params=params,
        solver_backend=str(opt_cfg["solver_backend"]),
        allow_offline_fallback=allow_offline_fallback,
        verify_with_enumerator=True,
    )
    summary_df.to_csv(out_dir / "policy_summary.csv", index=False)
    weekly_df.to_csv(out_dir / "weekly_test_results.csv", index=False)
    daily_sched_df.to_csv(out_dir / "daily_test_schedules.csv", index=False)

    # Independent brute-force check explicitly recorded on held-out Test Week 1
    wk1_pred = (
        fc_out.test_predictions.loc[
            fc_out.test_predictions["week_idx"] == fc_out.test_week_indices[0]
        ]
        .sort_values("iso_weekday")["pred_ridge"]
        .to_numpy(dtype=float)
    )
    milp_wk1 = solve_weekly_staffing_milp(
        wk1_pred,
        params,
        solver_backend=str(opt_cfg["solver_backend"]),
        allow_offline_fallback=allow_offline_fallback,
    )
    enum_wk1 = enumerate_optimal_schedule(wk1_pred, params)

    print("[5/6] Running 3x3 sensitivity grid and exploratory paired moving-block bootstrap ...")
    sens_df = run_sensitivity_grid(
        test_predictions_df=fc_out.test_predictions,
        base_params=params,
        productivity_multipliers=cfg["sensitivity"]["productivity_multipliers"],
        friday_penalties=cfg["sensitivity"]["friday_penalties"],
        solver_backend=str(opt_cfg["solver_backend"]),
        allow_offline_fallback=allow_offline_fallback,
    )
    sens_df.to_csv(out_dir / "sensitivity_grid.csv", index=False)

    boot_cfg = cfg["bootstrap"]
    boot_ridge_vs_flat = paired_moving_block_bootstrap(
        cost_baseline=weekly_df["flat_objective_cost"],
        cost_candidate=weekly_df["ridge_milp_objective_cost"],
        comparison_name="flat_minus_ridge_milp",
        block_length_weeks=int(boot_cfg["block_length_weeks"]),
        n_resamples=int(boot_cfg["n_resamples"]),
        confidence_level=float(boot_cfg["confidence_level"]),
        seed=int(boot_cfg["seed"]),
    )
    boot_prev_vs_flat = paired_moving_block_bootstrap(
        cost_baseline=weekly_df["flat_objective_cost"],
        cost_candidate=weekly_df["prev_week_milp_objective_cost"],
        comparison_name="flat_minus_prev_week_milp",
        block_length_weeks=int(boot_cfg["block_length_weeks"]),
        n_resamples=int(boot_cfg["n_resamples"]),
        confidence_level=float(boot_cfg["confidence_level"]),
        seed=int(boot_cfg["seed"]),
    )
    boot_ridge_vs_prev = paired_moving_block_bootstrap(
        cost_baseline=weekly_df["prev_week_milp_objective_cost"],
        cost_candidate=weekly_df["ridge_milp_objective_cost"],
        comparison_name="prev_week_milp_minus_ridge_milp",
        block_length_weeks=int(boot_cfg["block_length_weeks"]),
        n_resamples=int(boot_cfg["n_resamples"]),
        confidence_level=float(boot_cfg["confidence_level"]),
        seed=int(boot_cfg["seed"]),
    )

    label_tag = (
        "UCI Online Retail II (ID 502)"
        if dataset_mode == "uci_online_retail_ii"
        else "Synthetic Two-Sheet Verification Fixture (NOT UCI Empirical)"
    )

    print("[6/6] Rendering diagnostic charts ...")
    plot_predicted_vs_actual(
        daily_schedule_df=daily_sched_df,
        core_capacity_c=params.core_capacity_c,
        base_wape=fc_out.test_baseline_wape,
        ridge_wape=fc_out.test_ridge_wape,
        dataset_label=label_tag,
        out_path=fig_dir / "predicted_vs_actual.png",
    )
    plot_flat_vs_optimized_schedules(
        daily_schedule_df=daily_sched_df,
        dataset_label=label_tag,
        out_path=fig_dir / "flat_vs_optimized_schedules.png",
    )
    plot_weekly_backlog_costs(
        weekly_df=weekly_df,
        dataset_label=label_tag,
        out_path=fig_dir / "weekly_backlog_costs.png",
    )
    plot_sensitivity_results(
        sensitivity_df=sens_df,
        dataset_label=label_tag,
        out_path=fig_dir / "sensitivity_heatmap.png",
    )

    manifest = {
        "dataset_mode": dataset_mode,
        "is_uci_empirical_run": bool(dataset_mode == "uci_online_retail_ii"),
        "input_workbook_path": str(excel_path),
        "input_workbook_sha256": sha256_file(excel_path),
        "cleaning_audit": cleaning_audit.to_dict(),
        "capacity_calibration": {
            "initial_train_median_daily_lines": fc_out.initial_train_median_daily_lines,
            "core_daily_capacity_C_lines": params.core_capacity_c,
            "flex_block_productivity_q_lines": params.flex_productivity_q,
            "synthetic_disclaimer": (
                "C and q are synthetic capacity parameters calibrated solely from initial-training "
                "median daily booked order lines. Real warehouse pick rates and staffing are unobserved."
            ),
        },
        "forecasting": {
            "selected_ridge_alpha": fc_out.selected_ridge_alpha,
            "validation_alpha_grid_wape": {
                str(k): v for k, v in fc_out.validation_alpha_wape.items()
            },
            "validation_prev_week_wape": fc_out.validation_baseline_wape,
            "validation_ridge_wape": fc_out.validation_ridge_wape,
            "test_prev_week_wape": fc_out.test_baseline_wape,
            "test_ridge_wape": fc_out.test_ridge_wape,
            "ridge_backend_used": fc_out.ridge_backend_used,
        },
        "brute_force_verification_week1": {
            "feasible_schedules_checked": enum_wk1.num_feasible_schedules_checked,
            "milp_blocks": list(milp_wk1.blocks),
            "milp_objective": milp_wk1.objective_value,
            "enumerator_optimal_objective": enum_wk1.optimal_objective,
            "milp_schedule_in_enumerator_optima": milp_wk1.blocks in enum_wk1.all_optimal_schedules,
            "solver_backend_used": milp_wk1.solver_backend,
        },
        "bootstrap_exploratory_95ci": {
            "flat_minus_ridge_milp": boot_ridge_vs_flat.to_dict(),
            "flat_minus_prev_week_milp": boot_prev_vs_flat.to_dict(),
            "prev_week_milp_minus_ridge_milp": boot_ridge_vs_prev.to_dict(),
        },
    }

    with (out_dir / "cleaning_audit.json").open("w", encoding="utf-8") as fh:
        json.dump(cleaning_audit.to_dict(), fh, indent=2)
    with (out_dir / "bootstrap_ci.json").open("w", encoding="utf-8") as fh:
        json.dump(manifest["bootstrap_exploratory_95ci"], fh, indent=2)
    with (out_dir / "run_manifest.json").open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)

    print(f"[done] Saved verified results and charts under {out_dir.resolve()}")
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Warehouse Flex Staffing & Demand Forecasting Optimizer CLI"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    dl = sub.add_parser("download", help="Download UCI Online Retail II (ID 502) Excel workbook")
    dl.add_argument("--config", default="configs/base.yml")
    dl.add_argument("--output", default=None)

    run_p = sub.add_parser("run", help="Execute end-to-end SQL, forecast, MILP, replay, and charts")
    run_p.add_argument("--config", default="configs/base.yml")
    run_p.add_argument("--excel-path", default=None, help="Override path to two-sheet .xlsx input")
    run_p.add_argument("--output-dir", default="results/uci")
    run_p.add_argument(
        "--dataset-mode",
        choices=["uci_online_retail_ii", "synthetic_verification_fixture"],
        default="uci_online_retail_ii",
    )
    run_p.add_argument(
        "--download-if-missing",
        action="store_true",
        help="Automatically fetch UCI Dataset 502 if workbook is missing",
    )
    run_p.add_argument(
        "--allow-offline-fallback",
        action="store_true",
        help="Permit SQLite3 / SciPy-HiGHS fallbacks in offline CI containers without duckdb/pyomo",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "download":
        cfg = load_config(args.config)
        download_uci_workbook(cfg, args.output)
        return 0
    if args.command == "run":
        run_pipeline(
            config_path=args.config,
            excel_path_override=args.excel_path,
            output_dir=args.output_dir,
            dataset_mode=args.dataset_mode,
            allow_offline_fallback=args.allow_offline_fallback,
            download_if_missing=args.download_if_missing,
        )
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
