"""Transparent research liquidity proxies and recovery optimization."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linprog

HORIZONS_DAYS = np.array([7, 30, 90], dtype=int)


@dataclass(frozen=True)
class BankState:
    deposits: float
    assets: float
    cash: float
    securities: float
    borrowings: float
    equity: float
    loans: float


@dataclass(frozen=True)
class Scenario:
    deposit_runoff_fraction_90d: float
    borrowing_substitution_fraction_assets: float = 0.0
    liquidity_deterioration_fraction_assets: float = 0.0
    securities_haircut: float = .10
    secured_access_reduction: float = .0
    commitment_draw_fraction_loans: float = .02
    capital_asset_loss_fraction: float = .0
    capital_funding_trigger: float = .08
    capital_funding_access_penalty: float = .25


def calculate(state: BankState, scenario: Scenario,
              weekly_shape: np.ndarray | None = None) -> dict:
    if min(state.deposits, state.assets, state.cash, state.securities, state.loans) < 0:
        raise ValueError("Negative balance-sheet input")
    if state.assets <= 0:
        raise ValueError("Assets must be positive")
    for value in [scenario.deposit_runoff_fraction_90d,
                  scenario.borrowing_substitution_fraction_assets,
                  scenario.liquidity_deterioration_fraction_assets,
                  scenario.securities_haircut, scenario.secured_access_reduction,
                  scenario.commitment_draw_fraction_loans,
                  scenario.capital_asset_loss_fraction]:
        if value < 0:
            raise ValueError("Stress assumptions must be nonnegative")
    if weekly_shape is None:
        path = np.array([7 / 90, 30 / 90, 1.0])
    else:
        shape = np.asarray(weekly_shape, dtype=float)
        if shape.shape != (13,) or np.any(np.diff(shape) < -1e-8) or shape[-1] <= 0:
            raise ValueError("Weekly shape must be 13 nondecreasing positive cumulative fractions")
        path = shape[[0, 3, 12]] / shape[-1]
    runoff = state.deposits * min(scenario.deposit_runoff_fraction_90d, 1.0) * path
    borrowing_need = state.assets * scenario.borrowing_substitution_fraction_assets * path
    commitment = state.loans * scenario.commitment_draw_fraction_loans * path
    asset_loss = state.assets * scenario.capital_asset_loss_fraction
    stressed_capital_ratio = (state.equity - asset_loss) / max(state.assets - asset_loss, 1e-9)
    capital_penalty = (scenario.capital_funding_access_penalty
                       if stressed_capital_ratio < scenario.capital_funding_trigger else 0.0)
    effective_access_reduction = 1 - (1 - min(scenario.secured_access_reduction, 1.0)) * (1 - capital_penalty)
    collateral_value = state.securities * (1 - min(scenario.securities_haircut, 1.0))
    # Sellable and pledgeable shares are disjoint: 50% sale, 50% pledge.
    sale_capacity = collateral_value * .5
    pledge_capacity = collateral_value * .5 * .7 * (1 - effective_access_reduction)
    resource_activation = np.array([.25, .70, 1.0])
    resources = state.cash + resource_activation * (sale_capacity + pledge_capacity)
    liquid_deterioration = state.assets * scenario.liquidity_deterioration_fraction_assets * path
    net_outflows = runoff + borrowing_need + commitment + liquid_deterioration
    gap = net_outflows - resources
    peak_need = max(float(gap.max()), 0.0)
    shortfall_day = next((int(day) for day, value in zip(HORIZONS_DAYS, gap) if value > 0), None)
    utilization = np.divide(net_outflows, resources, out=np.full_like(net_outflows, np.nan), where=resources > 0)
    return {
        "horizon_days": HORIZONS_DAYS.tolist(),
        "path_fraction": path.tolist(),
        "deposit_runoff": runoff.tolist(),
        "borrowing_need": borrowing_need.tolist(),
        "commitment_draw_proxy": commitment.tolist(),
        "liquid_buffer_deterioration": liquid_deterioration.tolist(),
        "liquid_resources": resources.tolist(),
        "net_outflows": net_outflows.tolist(),
        "funding_gap": gap.tolist(),
        "peak_funding_need_proxy": peak_need,
        "maximum_liquidity_shortfall": peak_need,
        "buffer_utilization_max": float(np.nanmax(utilization)),
        "liquidity_runway_proxy_days": shortfall_day if shortfall_day is not None else 91,
        "stressed_capital_asset_ratio": stressed_capital_ratio,
        "effective_secured_access_reduction": effective_access_reduction,
    }


def reverse_runoff_threshold(state: BankState, base: Scenario,
                             *, horizon_index: int = 1,
                             weekly_shape: np.ndarray | None = None) -> float:
    if horizon_index not in (0, 1, 2):
        raise ValueError(horizon_index)
    zero = Scenario(**{**base.__dict__, "deposit_runoff_fraction_90d": 0.0})
    unit = Scenario(**{**base.__dict__, "deposit_runoff_fraction_90d": 1.0})
    gap_zero = calculate(state, zero, weekly_shape)["funding_gap"][horizon_index]
    gap_unit = calculate(state, unit, weekly_shape)["funding_gap"][horizon_index]
    marginal = gap_unit - gap_zero
    if marginal <= 0:
        return float("inf")
    threshold = max(0.0, -gap_zero / marginal)
    return threshold if threshold <= 1.0 else float("inf")


def optimize_recovery(state: BankState, funding_gap: float,
                      *, horizon_days: int = 30,
                      securities_haircut: float = .10,
                      secured_access_reduction: float = .0) -> dict:
    if funding_gap <= 0:
        return {"status": "no_action_needed", "action_amounts": [0.0] * 4,
                "action_cost": 0.0, "residual_shortfall": 0.0}
    collateral = state.securities * (1 - securities_haircut)
    baseline_activation = {7: .25, 30: .70, 90: 1.0}.get(horizon_days)
    if baseline_activation is None:
        raise ValueError("Recovery horizon must be 7, 30, or 90 days")
    incremental_share = 1 - baseline_activation
    capacities = np.array([
        incremental_share * .5 * collateral,
        incremental_share * .5 * collateral * .7 * (1 - secured_access_reduction),
        .05 * state.assets,
        .03 * state.loans,
    ])
    activation = np.array([2, 7, 30, 90])
    capacities[activation > horizon_days] = 0
    unit_cost = np.array([.03, .04, .08, .12])
    result = linprog(unit_cost, A_ub=-np.ones((1, 4)), b_ub=[-funding_gap],
                     bounds=[(0, float(capacity)) for capacity in capacities], method="highs")
    if result.success:
        amounts = result.x
        residual = max(funding_gap - amounts.sum(), 0)
        return {"status": "feasible", "action_amounts": amounts.tolist(),
                "action_cost": float(result.fun), "residual_shortfall": float(residual),
                "capacity": capacities.tolist(), "unit_cost": unit_cost.tolist()}
    # If fully covering the gap is infeasible, use all available actions and
    # disclose the residual shortfall rather than silently relaxing the target.
    return {"status": "infeasible_full_coverage", "action_amounts": capacities.tolist(),
            "action_cost": float(capacities @ unit_cost),
            "residual_shortfall": float(max(funding_gap - capacities.sum(), 0)),
            "capacity": capacities.tolist(), "unit_cost": unit_cost.tolist()}


def heuristic_recovery(state: BankState, funding_gap: float,
                       *, horizon_days: int = 30, securities_haircut: float = .10,
                       secured_access_reduction: float = .0) -> dict:
    baseline = optimize_recovery(state, funding_gap, horizon_days=horizon_days,
                                 securities_haircut=securities_haircut,
                                 secured_access_reduction=secured_access_reduction)
    capacities = np.asarray(baseline.get("capacity", [0.0] * 4))
    costs = np.asarray(baseline.get("unit_cost", [.03, .04, .08, .12]))
    amounts = np.zeros(4)
    remaining = max(funding_gap, 0)
    for index in [1, 0, 2, 3]:  # customary secured-funding-first ordering
        used = min(capacities[index], remaining)
        amounts[index] = used
        remaining -= used
    return {"status": "feasible" if remaining <= 1e-8 else "infeasible_full_coverage",
            "action_amounts": amounts.tolist(), "action_cost": float(amounts @ costs),
            "residual_shortfall": float(max(remaining, 0))}
