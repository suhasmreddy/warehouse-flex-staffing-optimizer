"""Operational calendar utilities, weekend folding, and deterministic UK holiday features.

Note on module naming:
    This file is named ``src/calendar.py`` per repository specification. Always
    import via package qualification (``from src import calendar as ops_calendar``)
    and execute scripts from the repository root (``python -m src.cli``) so
    Python's standard-library ``calendar`` module is never shadowed.
"""

from __future__ import annotations

import datetime as dt
from typing import Iterable

import numpy as np
import pandas as pd

WEEKDAY_NAMES: tuple[str, ...] = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday")


def fold_weekend_date(booking_date: dt.date) -> dt.date:
    """Fold Saturday (+2 days) and Sunday (+1 day) bookings into the next Monday.

    Parameters
    ----------
    booking_date : datetime.date
        Calendar date on which the retail order line was booked.

    Returns
    -------
    datetime.date
        Operational weekday (Monday through Friday) to which the booking applies.
    """
    iso_wd = booking_date.isoweekday()
    if iso_wd == 6:  # Saturday -> Monday
        return booking_date + dt.timedelta(days=2)
    if iso_wd == 7:  # Sunday -> Monday
        return booking_date + dt.timedelta(days=1)
    return booking_date


def fold_weekend_series(dates: pd.Series) -> pd.Series:
    """Vectorized weekend-to-Monday date folding for a pandas Series of dates/timestamps."""
    ts = pd.to_datetime(dates).dt.normalize()
    iso_wd = ts.dt.isocalendar().day.astype(int)
    add_days = np.where(iso_wd == 6, 2, np.where(iso_wd == 7, 1, 0))
    return ts + pd.to_timedelta(add_days, unit="D")


def complete_week_bounds(dates: Iterable[dt.date]) -> tuple[dt.date, dt.date]:
    """Return (first_complete_monday, last_complete_friday) inside observed operational dates.

    Any partial leading week (starting Tue-Fri) or partial trailing week
    (ending Mon-Thu) is trimmed so every operational week contains all five
    weekdays (Monday through Friday).
    """
    date_list = sorted(dates)
    if not date_list:
        raise ValueError("Cannot compute complete week bounds from empty date sequence.")

    min_d, max_d = date_list[0], date_list[-1]
    min_wd = min_d.isoweekday()
    first_monday = min_d if min_wd == 1 else min_d + dt.timedelta(days=(8 - min_wd))

    max_wd = max_d.isoweekday()
    if max_wd == 5:
        last_friday = max_d
    elif max_wd > 5:
        last_friday = max_d - dt.timedelta(days=(max_wd - 5))
    else:
        # Mon(1)->prev Fri(-3), Tue(2)->(-4), Wed(3)->(-5), Thu(4)->(-6)
        last_friday = max_d - dt.timedelta(days=(max_wd + 2))

    if first_monday > last_friday:
        raise ValueError(
            f"Date range [{min_d}, {max_d}] does not span at least one complete Monday-Friday week."
        )
    return first_monday, last_friday


def build_weekday_spine(first_monday: dt.date, last_friday: dt.date) -> pd.DataFrame:
    """Build a contiguous Monday-Friday operational calendar spine with zero gaps."""
    if first_monday.isoweekday() != 1:
        raise ValueError(f"first_monday must be a Monday (got {first_monday}).")
    if last_friday.isoweekday() != 5:
        raise ValueError(f"last_friday must be a Friday (got {last_friday}).")
    if first_monday > last_friday:
        raise ValueError("first_monday must not exceed last_friday.")

    b_days = pd.bdate_range(start=first_monday, end=last_friday, freq="B")
    iso_weekday = b_days.isocalendar().day.astype(int).to_numpy()
    week_monday = (b_days - pd.to_timedelta(iso_weekday - 1, unit="D")).date
    week_index = np.arange(len(b_days)) // 5

    return pd.DataFrame(
        {
            "operational_date": b_days.date,
            "iso_weekday": iso_weekday,
            "weekday_name": [WEEKDAY_NAMES[i - 1] for i in iso_weekday],
            "week_monday": week_monday,
            "week_idx": week_index,
        }
    )


def _easter_sunday(year: int) -> dt.date:
    """Anonymous Gregorian computus for Easter Sunday (purely calendar-deterministic)."""
    a = year % 19
    b, c = divmod(year, 100)
    d, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month, day = divmod(h + l - 7 * m + 114, 31)
    return dt.date(year, month, day + 1)


def uk_statutory_holidays(year: int) -> set[dt.date]:
    """Return statutory England & Wales bank holidays known deterministically in advance.

    Because UCI Online Retail II records orders for a UK-based giftware merchant,
    statutory UK bank holidays are public calendar constants known years ahead of
    every Friday staffing decision (zero future-demand leakage).
    """
    holidays: set[dt.date] = set()

    # 1. New Year's Day (substitute Monday if on weekend)
    nyd = dt.date(year, 1, 1)
    if nyd.isoweekday() == 6:
        holidays.add(dt.date(year, 1, 3))
    elif nyd.isoweekday() == 7:
        holidays.add(dt.date(year, 1, 2))
    else:
        holidays.add(nyd)

    # 2. Good Friday & Easter Monday
    easter = _easter_sunday(year)
    holidays.add(easter - dt.timedelta(days=2))
    holidays.add(easter + dt.timedelta(days=1))

    # 3. Early May Bank Holiday (first Monday in May)
    may1 = dt.date(year, 5, 1)
    early_may = may1 + dt.timedelta(days=(8 - may1.isoweekday()) % 7)
    holidays.add(early_may)

    # 4. Spring Bank Holiday (last Monday in May)
    may31 = dt.date(year, 5, 31)
    spring_bh = may31 - dt.timedelta(days=(may31.isoweekday() - 1) % 7)
    holidays.add(spring_bh)

    # 5. Summer Bank Holiday (last Monday in August)
    aug31 = dt.date(year, 8, 31)
    summer_bh = aug31 - dt.timedelta(days=(aug31.isoweekday() - 1) % 7)
    holidays.add(summer_bh)

    # 6. Christmas Day & Boxing Day (with weekend substitute days)
    xmas = dt.date(year, 12, 25)
    boxing = dt.date(year, 12, 26)
    if xmas.isoweekday() == 6:  # Sat Dec 25, Sun Dec 26 -> Mon Dec 27, Tue Dec 28
        holidays.update({dt.date(year, 12, 27), dt.date(year, 12, 28)})
    elif xmas.isoweekday() == 7:  # Sun Dec 25, Mon Dec 26 -> Mon Dec 26, Tue Dec 27
        holidays.update({dt.date(year, 12, 26), dt.date(year, 12, 27)})
    elif boxing.isoweekday() == 6:  # Fri Dec 25, Sat Dec 26 -> Fri Dec 25, Mon Dec 28
        holidays.update({dt.date(year, 12, 25), dt.date(year, 12, 28)})
    else:
        holidays.update({xmas, boxing})

    # Gazetted special UK public holidays announced >5 months ahead in dataset era
    if year == 2011:
        holidays.add(dt.date(2011, 4, 29))  # Royal Wedding bank holiday (announced Nov 2010)
    if year == 2012:
        holidays.add(dt.date(2012, 6, 4))
        holidays.add(dt.date(2012, 6, 5))

    return holidays


def add_deterministic_calendar_features(df: pd.DataFrame, date_col: str = "operational_date") -> pd.DataFrame:
    """Attach strictly calendar-deterministic features known at prediction time.

    Features added:
    - ``wd_mon`` .. ``wd_fri``: one-hot weekday indicators
    - ``sin_annual_1``, ``cos_annual_1``, ``sin_annual_2``, ``cos_annual_2``: Fourier annual harmonics
    - ``is_q4_peak``: 1 if date falls in Oct 1 - Dec 17 giftware peak window
    - ``is_xmas_shutdown_window``: 1 if date falls in Dec 20 - Jan 3 merchant closure window
    - ``is_uk_bank_holiday``: 1 if statutory England & Wales bank holiday
    - ``is_post_holiday_weekday``: 1 if first working weekday immediately following a bank holiday
    """
    out = df.copy()
    dates = pd.to_datetime(out[date_col])
    d_objs = dates.dt.date.tolist()
    years = sorted({d.year for d in d_objs} | {d.year - 1 for d in d_objs})
    all_holidays: set[dt.date] = set()
    for yr in years:
        all_holidays |= uk_statutory_holidays(yr)

    iso_wd = dates.dt.isocalendar().day.astype(int)
    for wd_num, wd_label in zip(range(1, 6), ("mon", "tue", "wed", "thu", "fri")):
        out[f"wd_{wd_label}"] = (iso_wd == wd_num).astype(float)

    day_of_year = dates.dt.dayofyear.to_numpy(dtype=float)
    angle = 2.0 * np.pi * day_of_year / 365.25
    out["sin_annual_1"] = np.sin(angle)
    out["cos_annual_1"] = np.cos(angle)
    out["sin_annual_2"] = np.sin(2.0 * angle)
    out["cos_annual_2"] = np.cos(2.0 * angle)

    months = dates.dt.month.to_numpy()
    days = dates.dt.day.to_numpy()
    out["is_q4_peak"] = (
        ((months == 10) | (months == 11) | ((months == 12) & (days <= 17)))
    ).astype(float)
    out["is_xmas_shutdown_window"] = (
        (((months == 12) & (days >= 20)) | ((months == 1) & (days <= 3)))
    ).astype(float)

    is_bh = np.array([1.0 if d in all_holidays else 0.0 for d in d_objs], dtype=float)
    post_bh = np.zeros(len(d_objs), dtype=float)
    for idx, d in enumerate(d_objs):
        if d in all_holidays:
            continue
        prev_bday = d - dt.timedelta(days=3 if d.isoweekday() == 1 else 1)
        if prev_bday in all_holidays:
            post_bh[idx] = 1.0

    out["is_uk_bank_holiday"] = is_bh
    out["is_post_holiday_weekday"] = post_bh
    return out
