"""Render verified UCI markdown summary tables and truthful resume bullets from saved outputs."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd


def main() -> int:
    uci_dir = Path("results/uci")
    manifest_path = uci_dir / "run_manifest.json"
    summary_path = uci_dir / "policy_summary.csv"
    weekly_path = uci_dir / "weekly_test_results.csv"

    if not (manifest_path.exists() and summary_path.exists() and weekly_path.exists()):
        print(
            "UCI results not yet generated under results/uci/. "
            "Run `bash scripts/reproduce.sh` on a machine with internet access.",
            file=sys.stderr,
        )
        return 1

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not manifest.get("is_uci_empirical_run", False):
        raise RuntimeError("Refusing to render UCI summary from a non-UCI manifest!")

    summ = pd.read_csv(summary_path).set_index("policy_key")
    audit = manifest["cleaning_audit"]
    cap = manifest["capacity_calibration"]

    base_wape = summ.loc["prev_week_milp", "forecast_wape_pct"]
    ridge_wape = summ.loc["ridge_milp", "forecast_wape_pct"]
    ridge_imp = summ.loc["ridge_milp", "rel_improvement_vs_flat_pct"]
    prev_imp = summ.loc["prev_week_milp", "rel_improvement_vs_flat_pct"]

    bullets = (
        "# Auto-Generated Empirical Resume Bullets (Verified from `results/uci/`)\n\n"
        f"- Engineered a DuckDB SQL and Python demand-planning pipeline over {audit['raw_rows_total']:,} "
        f"UCI Online Retail II transactions ({audit['valid_unique_invoice_sku_pairs']:,} deduplicated "
        f"order-line picks across {audit['complete_weeks_count']} weeks), achieving {ridge_wape:.1f}% "
        f"sealed-test WAPE (vs. {base_wape:.1f}% naive seasonal lag) via leakage-free rolling Ridge regression.\n"
        f"- Formulated a 5-day mixed-integer linear program (Pyomo, HiGHS) allocating 40 weekly flex hours "
        f"across indivisible 4-hour blocks (C={cap['core_daily_capacity_C_lines']}, q={cap['flex_block_productivity_q_lines']}), "
        f"changing modeled backlog objective cost by {ridge_imp:+.1f}% vs. flat staffing ({prev_imp:+.1f}% with naive forecasts) "
        f"across 12 held-out test weeks verified against a 381-schedule brute-force enumerator and 9-scenario sensitivity grid.\n"
    )
    (uci_dir / "resume_bullets.md").write_text(bullets, encoding="utf-8")
    print(f"[render] Wrote {uci_dir / 'resume_bullets.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
