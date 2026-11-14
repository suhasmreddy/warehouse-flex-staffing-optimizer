"""Five-day warehouse flex staffing MILP (Pyomo + HiGHS) and independent brute-force verifier.

Mathematical Formulation
------------------------
Sets / Indices:
    d in D = {0, 1, 2, 3, 4}   Weekdays Monday (0) through Friday (4).

Parameters (explicitly synthetic capacity calibration):
    f_d >= 0        Forecasted (or oracle) order-line arrivals on day d [lines/day].
    C >= 0          Core daily picking capacity [lines/day], calibrated solely on initial
                    training data as round(0.70 * median_initial_train_daily_lines).
    q >= 1          Baseline flex productivity per 4-hour block [lines/block], calibrated
                    as max(1, round(0.15 * median_initial_train_daily_lines)).
    m > 0           Sensitivity multiplier on flex productivity (default 1.0), yielding
                    effective productivity q_eff = m * q [lines/block].
    B = 10          Weekly flex budget in 4-hour blocks (40 flex labor hours / week).
    U = 4           Daily cap on flex blocks [blocks/day] (0 <= x_d <= 4, max 16 flex hrs/day).
    b_{-1} = 0      Initial weekly backlog entering Monday morning [lines].
    lambda_Fri >= 0 End-of-week Friday carryover penalty weight [extra day-equivalents],
                    default 3.0 (reflecting weekend delay before Monday core shift).

Decision Variables:
    x_d in {0, 1, 2, 3, 4}   Indivisible 4-hour flex staffing blocks assigned to day d [blocks].
    b_d >= 0                 Modeled end-of-day unpicked order-line backlog on day d [lines].

Constraints:
    (1) Exact weekly budget:     sum_{d=0}^4 x_d = 10
    (2) Lindley backlog flow:    b_d >= b_{d-1} + f_d - C - q_eff * x_d    for all d in D
    (3) Nonnegativity & bounds:  x_d in {0..4} integer,  b_d >= 0          for all d in D

Objective:
    Minimize   sum_{d=0}^4 b_d + lambda_Fri * b_4    [weighted line-days]
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp


@dataclass(frozen=True)
class StaffingParameters:
    """Synthetic capacity and MILP configuration parameters."""

    core_capacity_c: int
    flex_productivity_q: int
    weekly_blocks: int = 10
    min_daily_blocks: int = 0
    max_daily_blocks: int = 4
    initial_backlog: float = 0.0
    friday_penalty: float = 3.0
    productivity_multiplier: float = 1.0

    @property
    def effective_q(self) -> float:
        return float(self.flex_productivity_q) * float(self.productivity_multiplier)

    @classmethod
    def from_initial_training_median(
        cls,
        median_daily_lines: float,
        core_fraction: float = 0.70,
        flex_fraction: float = 0.15,
        weekly_blocks: int = 10,
        min_daily_blocks: int = 0,
        max_daily_blocks: int = 4,
        initial_backlog: float = 0.0,
        friday_penalty: float = 3.0,
        productivity_multiplier: float = 1.0,
    ) -> "StaffingParameters":
        c_val = int(round(core_fraction * float(median_daily_lines)))
        q_val = max(1, int(round(flex_fraction * float(median_daily_lines))))
        return cls(
            core_capacity_c=c_val,
            flex_productivity_q=q_val,
            weekly_blocks=weekly_blocks,
            min_daily_blocks=min_daily_blocks,
            max_daily_blocks=max_daily_blocks,
            initial_backlog=initial_backlog,
            friday_penalty=friday_penalty,
            productivity_multiplier=productivity_multiplier,
        )


@dataclass(frozen=True)
class StaffingSolution:
    """Validated solution returned by the five-day MILP optimizer."""

    blocks: tuple[int, int, int, int, int]
    modeled_backlog: tuple[float, float, float, float, float]
    objective_value: float
    solver_status: str
    solver_backend: str


@dataclass(frozen=True)
class BruteForceEnumerationResult:
    """Exact solution across all feasible 5-day integer block schedules."""

    optimal_blocks: tuple[int, int, int, int, int]
    optimal_backlog: tuple[float, float, float, float, float]
    optimal_objective: float
    all_optimal_schedules: list[tuple[int, int, int, int, int]]
    num_feasible_schedules_checked: int


def validate_staffing_solution(
    sol: StaffingSolution,
    forecast: Sequence[float],
    params: StaffingParameters,
    tol: float = 1e-5,
) -> None:
    """Assert solver optimality, integrality, daily bounds, exact 10-block budget, and queue bounds."""
    if sol.solver_status != "optimal":
        raise RuntimeError(f"MILP solver did not terminate optimally: status={sol.solver_status}")
    if len(sol.blocks) != 5 or len(sol.modeled_backlog) != 5 or len(forecast) != 5:
        raise ValueError("Five-day MILP solution and forecast must each have length 5.")

    for d, x_d in enumerate(sol.blocks):
        if not isinstance(x_d, (int, np.integer)):
            raise AssertionError(f"Non-integer block decision on day {d}: {x_d}")
        if x_d < params.min_daily_blocks or x_d > params.max_daily_blocks:
            raise AssertionError(
                f"Block decision x[{d}]={x_d} violates bounds "
                f"[{params.min_daily_blocks}, {params.max_daily_blocks}]."
            )

    if sum(sol.blocks) != params.weekly_blocks:
        raise AssertionError(
            f"Total weekly blocks {sum(sol.blocks)} != required budget {params.weekly_blocks}."
        )

    q_eff = params.effective_q
    prev_b = float(params.initial_backlog)
    for d in range(5):
        b_d = float(sol.modeled_backlog[d])
        if b_d < -tol:
            raise AssertionError(f"Negative modeled backlog on day {d}: {b_d}")
        rhs = prev_b + float(forecast[d]) - float(params.core_capacity_c) - q_eff * sol.blocks[d]
        if b_d + tol < rhs:
            raise AssertionError(
                f"Backlog flow constraint violated on day {d}: b[{d}]={b_d} < rhs={rhs}"
            )
        prev_b = b_d

    expected_obj = sum(sol.modeled_backlog) + float(params.friday_penalty) * sol.modeled_backlog[4]
    if abs(sol.objective_value - expected_obj) > 1e-4:
        raise AssertionError(
            f"Objective mismatch: reported {sol.objective_value} vs recomputed {expected_obj}"
        )


def _solve_pyomo_highs(forecast: Sequence[float], params: StaffingParameters) -> StaffingSolution:
    """Build and solve the five-day MILP using Pyomo and HiGHS (`appsi_highs` / `highs`)."""
    import pyomo.environ as pyo  # type: ignore[import-not-found]

    f_vec = [max(0.0, float(v)) for v in forecast]
    if len(f_vec) != 5:
        raise ValueError(f"Expected 5 weekday forecasts, got {len(f_vec)}.")

    m = pyo.ConcreteModel(name="WarehouseFlexStaffing5DayMILP")
    m.D = pyo.RangeSet(0, 4)

    m.x = pyo.Var(
        m.D,
        domain=pyo.NonNegativeIntegers,
        bounds=(params.min_daily_blocks, params.max_daily_blocks),
    )
    m.b = pyo.Var(m.D, domain=pyo.NonNegativeReals)

    m.weekly_budget = pyo.Constraint(
        expr=sum(m.x[d] for d in m.D) == params.weekly_blocks
    )

    q_eff = params.effective_q

    def _backlog_balance_rule(mdl: pyo.ConcreteModel, d: int):
        prev_b = params.initial_backlog if d == 0 else mdl.b[d - 1]
        return mdl.b[d] >= prev_b + f_vec[d] - float(params.core_capacity_c) - q_eff * mdl.x[d]

    m.backlog_balance = pyo.Constraint(m.D, rule=_backlog_balance_rule)
    m.total_cost = pyo.Objective(
        expr=sum(m.b[d] for d in m.D) + float(params.friday_penalty) * m.b[4],
        sense=pyo.minimize,
    )

    solver_used = None
    results = None
    for solver_name in ("appsi_highs", "highs"):
        solver = pyo.SolverFactory(solver_name)
        if solver is not None and solver.available(exception_flag=False):
            results = solver.solve(m, tee=False)
            solver_used = f"pyomo:{solver_name}"
            break

    if results is None or solver_used is None:
        raise RuntimeError(
            "Pyomo is installed, but neither 'appsi_highs' (`highspy`) nor 'highs' solver "
            "is available. Run `pip install highspy`."
        )

    term = results.solver.termination_condition
    status_str = "optimal" if term == pyo.TerminationCondition.optimal else str(term)

    x_vals: list[int] = []
    for d in range(5):
        raw_x = float(pyo.value(m.x[d]))
        if abs(raw_x - round(raw_x)) > 1e-4:
            raise AssertionError(f"Solver returned fractional block count x[{d}]={raw_x}")
        x_vals.append(int(round(raw_x)))

    b_vals = tuple(max(0.0, float(pyo.value(m.b[d]))) for d in range(5))
    obj_val = float(pyo.value(m.total_cost))

    sol = StaffingSolution(
        blocks=tuple(x_vals),  # type: ignore[arg-type]
        modeled_backlog=b_vals,  # type: ignore[arg-type]
        objective_value=obj_val,
        solver_status=status_str,
        solver_backend=solver_used,
    )
    validate_staffing_solution(sol, f_vec, params)
    return sol


def _solve_scipy_highs(forecast: Sequence[float], params: StaffingParameters) -> StaffingSolution:
    """Solve the identical 10-variable MILP using SciPy's native C++ HiGHS wrapper.

    Decision vector ordering: z = [x_0, x_1, x_2, x_3, x_4, b_0, b_1, b_2, b_3, b_4].
    Used only when `--allow-offline-fallback` is enabled in air-gapped environments
    without the `pyomo` Python wrapper installed.
    """
    f_vec = [max(0.0, float(v)) for v in forecast]
    if len(f_vec) != 5:
        raise ValueError(f"Expected 5 weekday forecasts, got {len(f_vec)}.")

    q_eff = params.effective_q
    c_obj = np.array(
        [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0 + float(params.friday_penalty)],
        dtype=float,
    )
    integrality = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0], dtype=int)
    lb = np.array([float(params.min_daily_blocks)] * 5 + [0.0] * 5, dtype=float)
    ub = np.array([float(params.max_daily_blocks)] * 5 + [np.inf] * 5, dtype=float)

    # Row 0: x_0 + x_1 + x_2 + x_3 + x_4 == weekly_blocks
    # Rows 1..5 (d=0..4): q_eff * x_d + b_d - b_{d-1} >= f_d - C  (with b_{-1} = initial_backlog)
    A = np.zeros((6, 10), dtype=float)
    row_lb = np.zeros(6, dtype=float)
    row_ub = np.full(6, np.inf, dtype=float)

    A[0, 0:5] = 1.0
    row_lb[0] = float(params.weekly_blocks)
    row_ub[0] = float(params.weekly_blocks)

    for d in range(5):
        A[d + 1, d] = q_eff
        A[d + 1, 5 + d] = 1.0
        if d == 0:
            row_lb[d + 1] = float(params.initial_backlog) + f_vec[d] - float(params.core_capacity_c)
        else:
            A[d + 1, 5 + d - 1] = -1.0
            row_lb[d + 1] = f_vec[d] - float(params.core_capacity_c)

    res = milp(
        c=c_obj,
        integrality=integrality,
        bounds=Bounds(lb, ub),
        constraints=LinearConstraint(A, row_lb, row_ub),
    )

    status_str = "optimal" if (res.success and res.status == 0) else f"status_{res.status}:{res.message}"
    if status_str != "optimal" or res.x is None:
        raise RuntimeError(f"HiGHS MILP failed to reach optimality: {status_str}")

    x_vals: list[int] = []
    for d in range(5):
        raw_x = float(res.x[d])
        if abs(raw_x - round(raw_x)) > 1e-4:
            raise AssertionError(f"HiGHS returned fractional x[{d}]={raw_x}")
        x_vals.append(int(round(raw_x)))

    b_vals = tuple(max(0.0, float(res.x[5 + d])) for d in range(5))
    sol = StaffingSolution(
        blocks=tuple(x_vals),  # type: ignore[arg-type]
        modeled_backlog=b_vals,  # type: ignore[arg-type]
        objective_value=float(res.fun),
        solver_status=status_str,
        solver_backend="scipy:highs",
    )
    validate_staffing_solution(sol, f_vec, params)
    return sol


def solve_weekly_staffing_milp(
    forecast: Sequence[float],
    params: StaffingParameters,
    solver_backend: str = "pyomo_highs",
    allow_offline_fallback: bool = False,
) -> StaffingSolution:
    """Solve the five-day flex staffing MILP and validate all solution invariants."""
    if solver_backend == "scipy_highs":
        return _solve_scipy_highs(forecast, params)

    try:
        import pyomo.environ as _pyo  # noqa: F401

        has_pyomo = True
    except ImportError:
        has_pyomo = False

    if has_pyomo:
        return _solve_pyomo_highs(forecast, params)
    if allow_offline_fallback:
        return _solve_scipy_highs(forecast, params)

    raise RuntimeError(
        "Pyomo (`pyomo` + `highspy`) is not installed. Run `pip install -r requirements.txt` "
        "or pass `--allow-offline-fallback` to solve via SciPy's embedded HiGHS backend."
    )


def enumerate_optimal_schedule(
    forecast: Sequence[float],
    params: StaffingParameters,
    tie_tol: float = 1e-9,
) -> BruteForceEnumerationResult:
    """Independently enumerate all feasible 5-day block schedules to verify MILP optimality.

    For ``min_daily_blocks=0``, ``max_daily_blocks=4``, and ``weekly_blocks=10``,
    there are exactly 381 feasible integer schedules in ``{0, 1, 2, 3, 4}^5``
    whose sum equals 10.
    """
    f_vec = [max(0.0, float(v)) for v in forecast]
    if len(f_vec) != 5:
        raise ValueError("enumerate_optimal_schedule expects a 5-day forecast sequence.")

    q_eff = params.effective_q
    c_cap = float(params.core_capacity_c)
    fri_pen = float(params.friday_penalty)
    b_init = float(params.initial_backlog)

    best_obj = float("inf")
    best_blocks: tuple[int, int, int, int, int] | None = None
    best_backlog: tuple[float, float, float, float, float] | None = None
    tied_schedules: list[tuple[int, int, int, int, int]] = []
    feasible_count = 0

    domain = range(params.min_daily_blocks, params.max_daily_blocks + 1)
    for cand in itertools.product(domain, repeat=5):
        if sum(cand) != params.weekly_blocks:
            continue
        feasible_count += 1

        b_list: list[float] = []
        b_prev = b_init
        for d in range(5):
            b_prev = max(0.0, b_prev + f_vec[d] - c_cap - q_eff * cand[d])
            b_list.append(b_prev)

        obj = sum(b_list) + fri_pen * b_list[4]
        if obj < best_obj - tie_tol:
            best_obj = obj
            best_blocks = cand  # type: ignore[assignment]
            best_backlog = tuple(b_list)  # type: ignore[assignment]
            tied_schedules = [cand]  # type: ignore[list-item]
        elif abs(obj - best_obj) <= tie_tol:
            tied_schedules.append(cand)  # type: ignore[arg-type]

    if best_blocks is None or best_backlog is None:
        raise ValueError("No feasible schedule exists for the supplied staffing bounds/budget.")

    return BruteForceEnumerationResult(
        optimal_blocks=best_blocks,
        optimal_backlog=best_backlog,
        optimal_objective=best_obj,
        all_optimal_schedules=tied_schedules,
        num_feasible_schedules_checked=feasible_count,
    )
