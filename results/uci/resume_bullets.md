# Auto-Generated Empirical Resume Bullets (Verified from `results/uci/`)

- Engineered a DuckDB SQL and Python demand-planning pipeline over 1,067,371 UCI Online Retail II transactions (992,034 deduplicated order-line picks across 105 weeks), achieving 19.6% sealed-test WAPE (vs. 21.0% naive seasonal lag) via leakage-free rolling Ridge regression.
- Formulated a 5-day mixed-integer linear program (Pyomo, HiGHS) allocating 40 weekly flex hours across indivisible 4-hour blocks (C=1060, q=227), changing modeled backlog objective cost by +5.1% vs. flat staffing (+5.1% with naive forecasts) across 12 held-out test weeks verified against a 381-schedule brute-force enumerator and 9-scenario sensitivity grid.
