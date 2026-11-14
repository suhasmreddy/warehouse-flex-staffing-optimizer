"""UCI Online Retail II dual-sheet Excel loader, cleaning audit, and DuckDB SQL executor."""

from __future__ import annotations

import sqlite3
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from src import calendar as ops_calendar

COLUMN_ALIASES: dict[str, str] = {
    "invoice": "invoice_no",
    "invoiceno": "invoice_no",
    "invoice_no": "invoice_no",
    "stockcode": "stock_code",
    "stock_code": "stock_code",
    "description": "description",
    "quantity": "quantity",
    "invoicedate": "invoice_ts",
    "invoice_date": "invoice_ts",
    "invoice_ts": "invoice_ts",
    "price": "unit_price",
    "unitprice": "unit_price",
    "unit_price": "unit_price",
    "customer id": "customer_id",
    "customerid": "customer_id",
    "customer_id": "customer_id",
    "country": "country",
}

DEFAULT_ADMIN_CODES: tuple[str, ...] = (
    "POST",
    "DOT",
    "M",
    "D",
    "CRUK",
    "C2",
    "BANK CHARGES",
    "AMAZONFEE",
    "B",
    "S",
    "PADS",
    "SP1002",
)
DEFAULT_ADMIN_PREFIXES: tuple[str, ...] = ("TEST", "ADJUST", "GIFT_")


@dataclass(frozen=True)
class CleaningAudit:
    """Step-by-step audit trail of transaction cleaning and deduplication."""

    raw_rows_total: int
    rows_per_sheet: dict[str, int]
    dropped_missing_required: int
    dropped_exact_duplicate_rows: int
    dropped_cancellations_and_adjustments: int
    dropped_nonpositive_quantity: int
    dropped_nonpositive_price: int
    dropped_admin_stock_codes: int
    valid_rows_before_pair_dedup: int
    dropped_duplicate_invoice_sku_lines: int
    valid_unique_invoice_sku_pairs: int
    retained_missing_customer_id_pairs: int
    weekend_folded_pairs: int
    complete_weeks_count: int
    complete_weekdays_count: int
    first_complete_monday: str
    last_complete_friday: str
    sql_engine_used: str

    def to_dict(self) -> dict:
        return asdict(self)


def normalize_retail_columns(df: pd.DataFrame, sheet_name: str = "") -> pd.DataFrame:
    """Standardize UCI Online Retail II worksheet columns across schema variants."""
    rename_map: dict[str, str] = {}
    for col in df.columns:
        key = str(col).strip().lower()
        if key in COLUMN_ALIASES:
            rename_map[col] = COLUMN_ALIASES[key]
    out = df.rename(columns=rename_map).copy()
    required = {"invoice_no", "stock_code", "quantity", "invoice_ts", "unit_price"}
    missing = required - set(out.columns)
    if missing:
        raise ValueError(
            f"Worksheet '{sheet_name}' is missing required columns {sorted(missing)}; "
            f"found {list(df.columns)}."
        )
    for optional_col in ("description", "customer_id", "country"):
        if optional_col not in out.columns:
            out[optional_col] = None
    out["sheet_name"] = sheet_name
    out["invoice_ts"] = pd.to_datetime(out["invoice_ts"], errors="coerce")
    out["quantity"] = pd.to_numeric(out["quantity"], errors="coerce")
    out["unit_price"] = pd.to_numeric(out["unit_price"], errors="coerce")
    return out[
        [
            "sheet_name",
            "invoice_no",
            "stock_code",
            "description",
            "quantity",
            "invoice_ts",
            "unit_price",
            "customer_id",
            "country",
        ]
    ]


def load_excel_worksheets(
    excel_path: str | Path,
    worksheets: Sequence[str] = ("Year 2009-2010", "Year 2010-2011"),
) -> pd.DataFrame:
    """Load both Excel worksheets from ``online_retail_II.xlsx`` and concatenate."""
    path = Path(excel_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Excel workbook not found at '{path}'. Run `python -m src.cli download` "
            "or `bash scripts/reproduce.sh` to fetch UCI Dataset ID 502."
        )

    sheets_dict = pd.read_excel(path, sheet_name=list(worksheets), engine="openpyxl")
    frames = [normalize_retail_columns(df, sheet_name=name) for name, df in sheets_dict.items()]
    return pd.concat(frames, ignore_index=True)


def compute_cleaning_audit(
    raw_df: pd.DataFrame,
    sql_engine_used: str,
    daily_df: pd.DataFrame,
    admin_codes: Sequence[str] = DEFAULT_ADMIN_CODES,
    admin_prefixes: Sequence[str] = DEFAULT_ADMIN_PREFIXES,
    min_unit_price: float = 0.0001,
) -> CleaningAudit:
    """Compute transparent row-level funnel counts mirroring ``sql/workload.sql``."""
    raw_total = int(len(raw_df))
    rows_per_sheet = {
        str(k): int(v) for k, v in raw_df["sheet_name"].value_counts(dropna=False).items()
    }

    df = raw_df.copy()
    inv_str = df["invoice_no"].astype("string").str.strip()
    sku_str = df["stock_code"].astype("string").str.strip().str.upper()

    valid_req_mask = (
        inv_str.notna()
        & (inv_str != "")
        & sku_str.notna()
        & (sku_str != "")
        & df["invoice_ts"].notna()
        & df["quantity"].notna()
        & df["unit_price"].notna()
    )
    dropped_missing = int((~valid_req_mask).sum())
    df = df.loc[valid_req_mask].copy()
    df["invoice_clean"] = inv_str.loc[valid_req_mask]
    df["invoice_upper"] = df["invoice_clean"].str.upper()
    df["sku_clean"] = sku_str.loc[valid_req_mask]

    # Cross-sheet exact row duplicates (e.g., Dec 1-9, 2010 boundary overlap)
    before_exact_dedup = len(df)
    df = df.drop_duplicates(
        subset=["invoice_clean", "sku_clean", "quantity", "invoice_ts", "unit_price"]
    )
    dropped_exact_dups = int(before_exact_dedup - len(df))

    # Cancellations ('C%') and bad-debt accounting adjustments ('A%')
    cancel_mask = df["invoice_upper"].str.startswith("C") | df["invoice_upper"].str.startswith("A")
    dropped_cancel = int(cancel_mask.sum())
    df = df.loc[~cancel_mask]

    # Nonpositive quantities
    bad_qty_mask = df["quantity"] <= 0
    dropped_qty = int(bad_qty_mask.sum())
    df = df.loc[~bad_qty_mask]

    # Nonpositive / sub-threshold prices
    bad_price_mask = df["unit_price"] <= min_unit_price
    dropped_price = int(bad_price_mask.sum())
    df = df.loc[~bad_price_mask]

    # Documented administrative stock codes
    admin_set = {c.upper() for c in admin_codes}
    prefix_tuple = tuple(p.upper() for p in admin_prefixes)
    admin_mask = df["sku_clean"].isin(admin_set) | df["sku_clean"].str.startswith(prefix_tuple)
    dropped_admin = int(admin_mask.sum())
    df = df.loc[~admin_mask].copy()

    valid_rows_before_pair = int(len(df))
    df["booking_date"] = pd.to_datetime(df["invoice_ts"]).dt.date
    df["cust_missing"] = df["customer_id"].isna() | (
        df["customer_id"].astype("string").str.strip() == ""
    )

    # Collapse to unique (invoice_no, stock_code) booked workload units
    df_sorted = df.sort_values("booking_date")
    pairs = df_sorted.drop_duplicates(subset=["invoice_clean", "sku_clean"], keep="first")
    dropped_pair_dups = int(valid_rows_before_pair - len(pairs))

    iso_wd = pd.to_datetime(pairs["booking_date"]).dt.isocalendar().day.astype(int)
    weekend_pairs = int(iso_wd.isin([6, 7]).sum())
    missing_cust_pairs = int(pairs["cust_missing"].sum())

    return CleaningAudit(
        raw_rows_total=raw_total,
        rows_per_sheet=rows_per_sheet,
        dropped_missing_required=dropped_missing,
        dropped_exact_duplicate_rows=dropped_exact_dups,
        dropped_cancellations_and_adjustments=dropped_cancel,
        dropped_nonpositive_quantity=dropped_qty,
        dropped_nonpositive_price=dropped_price,
        dropped_admin_stock_codes=dropped_admin,
        valid_rows_before_pair_dedup=valid_rows_before_pair,
        dropped_duplicate_invoice_sku_lines=dropped_pair_dups,
        valid_unique_invoice_sku_pairs=int(len(pairs)),
        retained_missing_customer_id_pairs=missing_cust_pairs,
        weekend_folded_pairs=weekend_pairs,
        complete_weeks_count=int(len(daily_df) // 5),
        complete_weekdays_count=int(len(daily_df)),
        first_complete_monday=str(daily_df["operational_date"].iloc[0]),
        last_complete_friday=str(daily_df["operational_date"].iloc[-1]),
        sql_engine_used=sql_engine_used,
    )


def _execute_duckdb_workload(raw_df: pd.DataFrame, sql_path: Path) -> pd.DataFrame:
    """Execute ``sql/workload.sql`` natively inside an in-memory DuckDB connection."""
    import duckdb  # type: ignore[import-not-found]

    query = sql_path.read_text(encoding="utf-8")
    con = duckdb.connect(database=":memory:")
    try:
        con.register("raw_transactions_df", raw_df)
        con.execute(
            """
            CREATE TABLE raw_transactions AS
            SELECT
                CAST(sheet_name AS VARCHAR) AS sheet_name,
                CAST(invoice_no AS VARCHAR) AS invoice_no,
                CAST(stock_code AS VARCHAR) AS stock_code,
                CAST(description AS VARCHAR) AS description,
                CAST(quantity AS DOUBLE) AS quantity,
                CAST(invoice_ts AS TIMESTAMP) AS invoice_ts,
                CAST(unit_price AS DOUBLE) AS unit_price,
                CAST(customer_id AS VARCHAR) AS customer_id,
                CAST(country AS VARCHAR) AS country
            FROM raw_transactions_df;
            """
        )
        res = con.execute(query).df()
    finally:
        con.close()

    res["operational_date"] = pd.to_datetime(res["operational_date"]).dt.date
    res["week_monday"] = pd.to_datetime(res["week_monday"]).dt.date
    res["iso_weekday"] = res["iso_weekday"].astype(int)
    res["weekday_name"] = [ops_calendar.WEEKDAY_NAMES[i - 1] for i in res["iso_weekday"]]
    res["week_idx"] = np.arange(len(res)) // 5
    return res


def _execute_sqlite_compat_workload(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Execute equivalent SQL cleaning & weekend folding via stdlib SQLite3 when offline.

    Used strictly when ``allow_sqlite_fallback=True`` in air-gapped CI/VM environments
    where the ``duckdb`` PyPI wheel cannot be installed. Implements the exact same
    filtering, ``(invoice_no, stock_code)`` deduplication, and Saturday/Sunday-to-Monday
    folding rules as ``sql/workload.sql``, then joins onto ``build_weekday_spine``.
    """
    prep = pd.DataFrame(
        {
            "invoice_no": raw_df["invoice_no"].where(raw_df["invoice_no"].notna(), None).astype(object),
            "stock_code": raw_df["stock_code"].where(raw_df["stock_code"].notna(), None).astype(object),
            "quantity": pd.to_numeric(raw_df["quantity"], errors="coerce"),
            "unit_price": pd.to_numeric(raw_df["unit_price"], errors="coerce"),
            "booking_date": pd.to_datetime(raw_df["invoice_ts"], errors="coerce").dt.strftime("%Y-%m-%d"),
        }
    )
    for col in ("invoice_no", "stock_code", "booking_date"):
        prep[col] = [None if pd.isna(v) else str(v) for v in prep[col]]

    con = sqlite3.connect(":memory:")
    try:
        prep.to_sql("raw_transactions", con, index=False, if_exists="replace")
        sqlite_sql = """
        WITH normalized AS (
            SELECT
                TRIM(invoice_no) AS invoice_no_clean,
                UPPER(TRIM(invoice_no)) AS invoice_no_upper,
                UPPER(TRIM(stock_code)) AS stock_code_clean,
                CAST(quantity AS REAL) AS quantity_num,
                CAST(unit_price AS REAL) AS unit_price_num,
                booking_date
            FROM raw_transactions
            WHERE invoice_no IS NOT NULL
              AND stock_code IS NOT NULL
              AND booking_date IS NOT NULL
              AND quantity IS NOT NULL
              AND unit_price IS NOT NULL
              AND TRIM(invoice_no) <> ''
              AND TRIM(stock_code) <> ''
        ),
        valid_lines AS (
            SELECT
                invoice_no_clean AS invoice_no,
                stock_code_clean AS stock_code,
                MIN(booking_date) AS booking_date
            FROM normalized
            WHERE invoice_no_upper NOT LIKE 'C%'
              AND invoice_no_upper NOT LIKE 'A%'
              AND quantity_num > 0
              AND unit_price_num > 0.0001
              AND stock_code_clean NOT IN (
                  'POST', 'DOT', 'M', 'D', 'CRUK', 'C2',
                  'BANK CHARGES', 'AMAZONFEE', 'B', 'S', 'PADS', 'SP1002'
              )
              AND stock_code_clean NOT LIKE 'TEST%'
              AND stock_code_clean NOT LIKE 'ADJUST%'
              AND stock_code_clean NOT LIKE 'GIFT_%'
            GROUP BY invoice_no_clean, stock_code_clean
        ),
        folded_lines AS (
            SELECT
                invoice_no,
                stock_code,
                booking_date,
                -- SQLite strftime('%w'): 0=Sunday, 6=Saturday
                CASE CAST(strftime('%w', booking_date) AS INTEGER)
                    WHEN 6 THEN date(booking_date, '+2 days')
                    WHEN 0 THEN date(booking_date, '+1 day')
                    ELSE booking_date
                END AS operational_date,
                CASE
                    WHEN CAST(strftime('%w', booking_date) AS INTEGER) IN (0, 6) THEN 1
                    ELSE 0
                END AS is_weekend_booking
            FROM valid_lines
        )
        SELECT
            operational_date,
            COUNT(*) AS booked_lines,
            SUM(CASE WHEN is_weekend_booking = 0 THEN 1 ELSE 0 END) AS weekday_direct_lines,
            SUM(is_weekend_booking) AS weekend_folded_lines,
            COUNT(DISTINCT invoice_no) AS unique_invoices,
            COUNT(DISTINCT stock_code) AS unique_skus
        FROM folded_lines
        GROUP BY operational_date
        ORDER BY operational_date ASC;
        """
        counts_df = pd.read_sql_query(sqlite_sql, con)
    finally:
        con.close()

    if counts_df.empty:
        raise ValueError("No valid booked order lines remained after SQL cleaning.")

    counts_df["operational_date"] = pd.to_datetime(counts_df["operational_date"]).dt.date
    first_monday, last_friday = ops_calendar.complete_week_bounds(counts_df["operational_date"])
    spine = ops_calendar.build_weekday_spine(first_monday, last_friday)

    merged = spine.merge(counts_df, on="operational_date", how="left")
    for col in (
        "booked_lines",
        "weekday_direct_lines",
        "weekend_folded_lines",
        "unique_invoices",
        "unique_skus",
    ):
        merged[col] = merged[col].fillna(0).astype(np.int64)
    return merged


def aggregate_daily_workload(
    raw_df: pd.DataFrame,
    sql_path: str | Path = "sql/workload.sql",
    allow_sqlite_fallback: bool = False,
) -> tuple[pd.DataFrame, CleaningAudit]:
    """Clean raw transactions and aggregate daily Mon-Fri workload via DuckDB SQL.

    Parameters
    ----------
    raw_df : pd.DataFrame
        Concatenated two-sheet transactions normalized via ``normalize_retail_columns``.
    sql_path : str | Path
        Path to ``sql/workload.sql``.
    allow_sqlite_fallback : bool, default False
        If False (default), requires ``duckdb`` and executes ``sql/workload.sql`` natively.
        If True and ``duckdb`` is unavailable in an offline environment, executes the
        verified SQLite3 SQL compatibility query and logs ``sql_engine_used="sqlite3_compat"``.
    """
    sql_file = Path(sql_path)
    try:
        import duckdb as _duckdb  # noqa: F401

        has_duckdb = True
    except ImportError:
        has_duckdb = False

    if has_duckdb:
        daily_df = _execute_duckdb_workload(raw_df, sql_file)
        engine_used = "duckdb"
    elif allow_sqlite_fallback:
        daily_df = _execute_sqlite_compat_workload(raw_df)
        engine_used = "sqlite3_compat"
    else:
        raise RuntimeError(
            "DuckDB (`duckdb`) is not installed. Install dependencies via "
            "`pip install -r requirements.txt` or pass `--allow-offline-fallback` "
            "for air-gapped CI smoke verification."
        )

    if len(daily_df) % 5 != 0:
        raise AssertionError(
            f"Daily operational spine length ({len(daily_df)}) is not a multiple of 5 weekdays."
        )

    audit = compute_cleaning_audit(raw_df=raw_df, sql_engine_used=engine_used, daily_df=daily_df)
    return daily_df, audit
