"""Tests for dual-sheet Excel loading, duplicate/admin/cancellation cleaning, and weekend folding."""

from __future__ import annotations

import calendar as stdlib_calendar
import datetime as dt
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src import calendar as ops_calendar
from src.workload import aggregate_daily_workload, load_excel_worksheets


class TestCleaningAndCalendar(unittest.TestCase):
    def test_stdlib_calendar_not_shadowed(self) -> None:
        """Importing `src.calendar` must never shadow Python's built-in `calendar` module."""
        self.assertTrue(hasattr(stdlib_calendar, "monthrange"))
        self.assertEqual(stdlib_calendar.monthrange(2011, 2), (1, 28))

    def test_weekend_folding_rules(self) -> None:
        sat = dt.date(2010, 12, 4)  # Saturday
        sun = dt.date(2010, 12, 5)  # Sunday
        mon = dt.date(2010, 12, 6)  # Monday
        fri = dt.date(2010, 12, 10)  # Friday

        self.assertEqual(ops_calendar.fold_weekend_date(sat), mon)
        self.assertEqual(ops_calendar.fold_weekend_date(sun), mon)
        self.assertEqual(ops_calendar.fold_weekend_date(mon), mon)
        self.assertEqual(ops_calendar.fold_weekend_date(fri), fri)

    def test_dual_sheet_dedup_cancellations_admin_and_weekend_sql(self) -> None:
        """Verify cross-sheet overlap deduplication, cancellations, admin SKUs, and weekend folding."""
        # Week 1: Mon 2010-12-06 .. Fri 2010-12-10 (plus preceding Sat 12-04 & Sun 12-05)
        sheet1_rows = [
            # Valid Saturday booking -> folds to Mon 2010-12-06
            ("536001", "85123A", "WHITE HANGING HEART", 6, dt.datetime(2010, 12, 4, 10, 0), 2.55, 17850, "UK"),
            # Valid Sunday booking with missing Customer ID -> retained & folds to Mon 2010-12-06
            ("536002", "22423", "REGENCY CAKESTAND", 2, dt.datetime(2010, 12, 5, 14, 0), 12.75, None, "UK"),
            # Overlapping boundary record duplicated identically in Sheet 2 -> counted ONCE on Mon 2010-12-06
            ("536003", "84879", "ASSORTED COLOUR BIRD", 8, dt.datetime(2010, 12, 6, 9, 30), 1.69, 13047, "UK"),
            # Intra-invoice duplicate (Invoice, StockCode) split across two rows -> counted ONCE
            ("536004", "21730", "GLASS STAR FROSTED", 3, dt.datetime(2010, 12, 7, 11, 0), 4.25, 13047, "UK"),
            ("536004", "21730", "GLASS STAR FROSTED", 1, dt.datetime(2010, 12, 7, 11, 5), 4.25, 13047, "UK"),
            # Cancellation ('C' prefix) -> excluded
            ("C536005", "22423", "CANCELLED ITEM", -2, dt.datetime(2010, 12, 7, 15, 0), 12.75, 17850, "UK"),
            # Negative quantity without 'C' -> excluded
            ("536006", "22423", "DAMAGED ADJUST", -5, dt.datetime(2010, 12, 8, 10, 0), 12.75, None, "UK"),
            # Zero price -> excluded
            ("536007", "22423", "FREE SAMPLE ZERO", 4, dt.datetime(2010, 12, 8, 12, 0), 0.0, None, "UK"),
        ]
        sheet2_rows = [
            # Exact cross-worksheet duplicate of 536003 from Sheet 1
            ("536003", "84879", "ASSORTED COLOUR BIRD", 8, dt.datetime(2010, 12, 6, 9, 30), 1.69, 13047, "UK"),
            # Documented administrative StockCodes -> excluded
            ("536008", "POST", "POSTAGE", 1, dt.datetime(2010, 12, 8, 13, 0), 18.00, 12583, "France"),
            ("536008", "DOT", "DOTCOM POSTAGE", 1, dt.datetime(2010, 12, 8, 13, 0), 45.00, None, "UK"),
            ("536008", "M", "MANUAL", 1, dt.datetime(2010, 12, 8, 13, 0), 5.00, None, "UK"),
            ("536008", "BANK CHARGES", "BANK CHARGES", 1, dt.datetime(2010, 12, 8, 13, 0), 15.00, None, "UK"),
            ("A563185", "B", "Adjust bad debt", 1, dt.datetime(2010, 12, 9, 14, 0), 100.0, None, "UK"),
            # Valid Friday line (leaving Wed 12-08 and Thu 12-09 with ZERO valid lines -> zero-filled!)
            ("536009", "47566", "PARTY BUNTING", 10, dt.datetime(2010, 12, 10, 16, 0), 4.65, 14688, "UK"),
        ]
        cols = ["Invoice", "StockCode", "Description", "Quantity", "InvoiceDate", "Price", "Customer ID", "Country"]

        with tempfile.TemporaryDirectory() as tmp:
            xlsx_path = Path(tmp) / "test_retail.xlsx"
            with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
                pd.DataFrame(sheet1_rows, columns=cols).to_excel(writer, sheet_name="Year 2009-2010", index=False)
                pd.DataFrame(sheet2_rows, columns=cols).to_excel(writer, sheet_name="Year 2010-2011", index=False)

            raw_df = load_excel_worksheets(xlsx_path)
            daily_df, audit = aggregate_daily_workload(raw_df, allow_sqlite_fallback=True)

        self.assertEqual(len(daily_df), 5)  # Mon 2010-12-06 .. Fri 2010-12-10
        by_date = daily_df.set_index("operational_date")

        # Mon 12-06 gets Sat(536001) + Sun(536002) + Mon(536003 deduplicated) = 3 lines (2 folded)
        self.assertEqual(int(by_date.loc[dt.date(2010, 12, 6), "booked_lines"]), 3)
        self.assertEqual(int(by_date.loc[dt.date(2010, 12, 6), "weekend_folded_lines"]), 2)
        self.assertEqual(int(by_date.loc[dt.date(2010, 12, 6), "weekday_direct_lines"]), 1)

        # Tue 12-07 gets 536004 deduplicated to 1 unique (invoice, sku) pair
        self.assertEqual(int(by_date.loc[dt.date(2010, 12, 7), "booked_lines"]), 1)

        # Wed 12-08 & Thu 12-09 had only invalid/admin rows -> explicitly zero-filled on spine
        self.assertEqual(int(by_date.loc[dt.date(2010, 12, 8), "booked_lines"]), 0)
        self.assertEqual(int(by_date.loc[dt.date(2010, 12, 9), "booked_lines"]), 0)

        # Fri 12-10 gets 536009 = 1 line
        self.assertEqual(int(by_date.loc[dt.date(2010, 12, 10), "booked_lines"]), 1)

        self.assertEqual(audit.valid_unique_invoice_sku_pairs, 5)
        self.assertEqual(audit.retained_missing_customer_id_pairs, 1)
        self.assertEqual(audit.dropped_exact_duplicate_rows, 1)


if __name__ == "__main__":
    unittest.main()
