-- =============================================================================
-- sql/workload.sql
-- DuckDB SQL Pipeline: Daily Booked Order-Line Workload Aggregation
-- Dataset: UCI Online Retail II (ID 502, Daqing Chen, CC BY 4.0)
--
-- Proxy disclaimer:
--   Each unique valid (invoice_no, stock_code) pair represents one booked order
--   line (a proxy for potential warehouse picking demand). Actual pick times,
--   pick-path batching, physical staffing, and floor queue lengths are not
--   observed in transaction invoices.
--
-- Expected input table:
--   raw_transactions(
--       sheet_name VARCHAR,
--       invoice_no VARCHAR,
--       stock_code VARCHAR,
--       description VARCHAR,
--       quantity DOUBLE,
--       invoice_ts TIMESTAMP,
--       unit_price DOUBLE,
--       customer_id VARCHAR,
--       country VARCHAR
--   )
-- =============================================================================

WITH normalized AS (
    SELECT
        TRIM(CAST(invoice_no AS VARCHAR)) AS invoice_no_clean,
        UPPER(TRIM(CAST(invoice_no AS VARCHAR))) AS invoice_no_upper,
        UPPER(TRIM(CAST(stock_code AS VARCHAR))) AS stock_code_clean,
        CAST(quantity AS DOUBLE) AS quantity_num,
        CAST(unit_price AS DOUBLE) AS unit_price_num,
        CAST(invoice_ts AS TIMESTAMP) AS invoice_ts_clean,
        CAST(invoice_ts AS DATE) AS booking_date
    FROM raw_transactions
    WHERE invoice_no IS NOT NULL
      AND stock_code IS NOT NULL
      AND invoice_ts IS NOT NULL
      AND quantity IS NOT NULL
      AND unit_price IS NOT NULL
      AND TRIM(CAST(invoice_no AS VARCHAR)) <> ''
      AND TRIM(CAST(stock_code AS VARCHAR)) <> ''
),

-- Deduplicate exact cross-worksheet overlapping rows (e.g., Dec 1-9, 2010)
-- and filter out cancellations, bad-debt adjustments, nonpositive quantities/prices,
-- and documented non-merchandise administrative stock codes.
-- Note: Missing Customer ID rows are intentionally RETAINED because guest/wholesale
-- orders without a customer identifier still generate physical picking workload.
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
    GROUP BY
        invoice_no_clean,
        stock_code_clean
),

-- Fold Saturday (ISODOW = 6 -> +2 days) and Sunday (ISODOW = 7 -> +1 day)
-- bookings into the following operational Monday.
folded_lines AS (
    SELECT
        invoice_no,
        stock_code,
        booking_date,
        CASE EXTRACT(ISODOW FROM booking_date)
            WHEN 6 THEN booking_date + INTERVAL 2 DAY
            WHEN 7 THEN booking_date + INTERVAL 1 DAY
            ELSE booking_date
        END::DATE AS operational_date,
        CASE
            WHEN EXTRACT(ISODOW FROM booking_date) IN (6, 7) THEN 1
            ELSE 0
        END AS is_weekend_booking
    FROM valid_lines
),

daily_counts AS (
    SELECT
        operational_date,
        COUNT(*) AS booked_lines,
        SUM(CASE WHEN is_weekend_booking = 0 THEN 1 ELSE 0 END) AS weekday_direct_lines,
        SUM(is_weekend_booking) AS weekend_folded_lines,
        COUNT(DISTINCT invoice_no) AS unique_invoices,
        COUNT(DISTINCT stock_code) AS unique_skus
    FROM folded_lines
    GROUP BY operational_date
),

-- Determine first Monday on or after MIN(operational_date) and last Friday
-- on or before MAX(operational_date) so every week has all 5 Mon-Fri days.
date_bounds AS (
    SELECT
        CASE EXTRACT(ISODOW FROM MIN(operational_date))
            WHEN 1 THEN MIN(operational_date)
            ELSE MIN(operational_date) + (8 - EXTRACT(ISODOW FROM MIN(operational_date)))::INTEGER * INTERVAL 1 DAY
        END::DATE AS first_monday,
        CASE
            WHEN EXTRACT(ISODOW FROM MAX(operational_date)) = 5 THEN MAX(operational_date)
            ELSE MAX(operational_date) - (EXTRACT(ISODOW FROM MAX(operational_date)) + 2)::INTEGER * INTERVAL 1 DAY
        END::DATE AS last_friday
    FROM daily_counts
),

-- Build complete contiguous Monday-Friday spine (including zero-booking bank holidays)
calendar_spine AS (
    SELECT
        CAST(gs.day_dt AS DATE) AS operational_date,
        EXTRACT(ISODOW FROM gs.day_dt)::INTEGER AS iso_weekday
    FROM date_bounds db,
         GENERATE_SERIES(db.first_monday, db.last_friday, INTERVAL 1 DAY) AS gs(day_dt)
    WHERE EXTRACT(ISODOW FROM gs.day_dt) BETWEEN 1 AND 5
)

SELECT
    cs.operational_date,
    cs.iso_weekday,
    (cs.operational_date - (cs.iso_weekday - 1) * INTERVAL 1 DAY)::DATE AS week_monday,
    COALESCE(dc.booked_lines, 0)::BIGINT AS booked_lines,
    COALESCE(dc.weekday_direct_lines, 0)::BIGINT AS weekday_direct_lines,
    COALESCE(dc.weekend_folded_lines, 0)::BIGINT AS weekend_folded_lines,
    COALESCE(dc.unique_invoices, 0)::BIGINT AS unique_invoices,
    COALESCE(dc.unique_skus, 0)::BIGINT AS unique_skus
FROM calendar_spine cs
LEFT JOIN daily_counts dc
    ON cs.operational_date = dc.operational_date
ORDER BY cs.operational_date ASC;
