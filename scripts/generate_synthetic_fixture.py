"""Generate a realistic two-worksheet synthetic retail Excel fixture for offline CI verification.

IMPORTANT:
    This generator is strictly for offline automated testing and pipeline verification
    when `data/raw/online_retail_II.xlsx` (UCI Dataset ID 502) cannot be downloaded
    (e.g., air-gapped CI runners). Output is written to
    `data/fixtures/synthetic_retail_two_sheet_fixture.xlsx` and is NEVER substituted
    for empirical UCI Online Retail II results.
"""

from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import pandas as pd

from src import calendar as ops_calendar


def generate_two_sheet_fixture(
    out_path: str | Path = "data/fixtures/synthetic_retail_two_sheet_fixture.xlsx",
    seed: int = 2026,
) -> Path:
    """Create a 105-week two-sheet `.xlsx` fixture exhibiting UCI Online Retail II quirks.

    Simulates realistic wholesale/retail day-to-day overdispersion (CV ~ 0.32),
    weekend-to-Monday order accumulation, statutory UK bank holidays, Dec 1-9 2010
    cross-sheet overlap, cancellations, negative quantities, and admin stock codes,
    spanning balanced shoulder weeks and peak Q4 congestion regimes.
    """
    rng = np.random.default_rng(seed)
    start_dt = dt.date(2009, 12, 1)
    end_dt = dt.date(2011, 12, 9)

    holidays = (
        ops_calendar.uk_statutory_holidays(2009)
        | ops_calendar.uk_statutory_holidays(2010)
        | ops_calendar.uk_statutory_holidays(2011)
    )

    # Distinct day-of-week profile + Sunday online orders that fold into Monday
    weekday_mult = {1: 0.92, 2: 0.96, 3: 0.90, 4: 1.16, 5: 0.88, 6: 0.0, 7: 0.28}
    rows_sheet1: list[tuple] = []
    rows_sheet2: list[tuple] = []
    inv_counter = 489434

    cur = start_dt
    while cur <= end_dt:
        iso_wd = cur.isoweekday()
        is_xmas_closed = (cur.month == 12 and cur.day >= 23) or (cur.month == 1 and cur.day <= 3)

        if iso_wd != 6 and cur not in holidays and not is_xmas_closed:
            doy = cur.timetuple().tm_yday
            # Moderate Q4 peak in Nov - early Dec alongside balanced Sep-Oct shoulder weeks
            if cur.month == 11 or (cur.month == 12 and cur.day <= 12):
                seasonal_mult = 1.24
            elif cur.month == 10:
                seasonal_mult = 1.04
            else:
                seasonal_mult = 1.0 + 0.12 * np.sin(2.0 * np.pi * (doy - 250) / 365.25)

            mean_lines = 52.0 * weekday_mult[iso_wd] * seasonal_mult
            # Realistic wholesale order overdispersion (Gamma-Poisson mixture + occasional burst)
            gamma_draw = rng.gamma(shape=9.5, scale=mean_lines / 9.5)
            if rng.random() < 0.09:
                gamma_draw *= float(rng.uniform(1.30, 1.60))
            elif rng.random() < 0.08:
                gamma_draw *= float(rng.uniform(0.55, 0.75))

            target_valid_lines = max(3, int(round(gamma_draw)))

            n_invoices = max(2, target_valid_lines // 6)
            lines_allocated = 0
            for i_inv in range(n_invoices):
                inv_counter += 1
                inv_str = str(inv_counter)
                n_items = (
                    target_valid_lines - lines_allocated
                    if i_inv == n_invoices - 1
                    else max(1, min(8, target_valid_lines - lines_allocated - (n_invoices - i_inv - 1)))
                )
                lines_allocated += n_items
                cust_id = None if rng.random() < 0.22 else int(rng.integers(12346, 18287))
                sku_choices = rng.choice(range(20001, 20500), size=max(1, n_items), replace=False)
                ts = dt.datetime.combine(
                    cur,
                    dt.time(hour=int(rng.integers(8, 17)), minute=int(rng.integers(0, 60))),
                )

                for sku_num in sku_choices:
                    rec = (
                        inv_str,
                        f"{sku_num}",
                        f"GIFTWARE ITEM {sku_num}",
                        int(rng.integers(1, 25)),
                        ts,
                        round(float(rng.uniform(0.85, 19.95)), 2),
                        cust_id,
                        "United Kingdom",
                    )
                    if cur <= dt.date(2010, 12, 9):
                        rows_sheet1.append(rec)
                    if cur >= dt.date(2010, 12, 1):
                        # Intentional Dec 1-9, 2010 cross-worksheet overlap matching UCI Dataset 502
                        rows_sheet2.append(rec)

                    if rng.random() < 0.015:
                        target_list = rows_sheet1 if cur < dt.date(2010, 12, 1) else rows_sheet2
                        target_list.append(rec)

                target_list = rows_sheet1 if cur < dt.date(2010, 12, 1) else rows_sheet2
                if rng.random() < 0.12:
                    target_list.append(
                        (f"C{inv_counter}", f"{sku_choices[0]}", "CANCELLED LINE", -3, ts, 4.95, cust_id, "United Kingdom")
                    )
                if rng.random() < 0.08:
                    admin_code = str(rng.choice(["POST", "DOT", "M", "BANK CHARGES", "TEST001", "gift_0001_10"]))
                    target_list.append(
                        (inv_str, admin_code, "ADMIN FEE", 1, ts, 15.00, cust_id, "United Kingdom")
                    )
                if rng.random() < 0.04:
                    target_list.append(
                        (inv_str, "20999", "DAMAGED STOCK CHECK", -5, ts, 0.0, None, "United Kingdom")
                    )

        cur += dt.timedelta(days=1)

    rows_sheet2.append(
        ("A563185", "B", "Adjust bad debt", 1, dt.datetime(2011, 8, 12, 14, 50), -11062.06, None, "United Kingdom")
    )

    cols = [
        "Invoice",
        "StockCode",
        "Description",
        "Quantity",
        "InvoiceDate",
        "Price",
        "Customer ID",
        "Country",
    ]
    df1 = pd.DataFrame(rows_sheet1, columns=cols)
    df2 = pd.DataFrame(rows_sheet2, columns=cols)

    target = Path(out_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(target, engine="openpyxl") as writer:
        df1.to_excel(writer, sheet_name="Year 2009-2010", index=False)
        df2.to_excel(writer, sheet_name="Year 2010-2011", index=False)

    print(
        f"[fixture] Wrote {target} (Sheet1={len(df1):,} rows, Sheet2={len(df2):,} rows, "
        f"Total={len(df1) + len(df2):,} rows)."
    )
    return target


if __name__ == "__main__":
    generate_two_sheet_fixture()
