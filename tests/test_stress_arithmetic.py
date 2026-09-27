from __future__ import annotations

import math

import numpy as np

from treasuryshield.stress.arithmetic import (
    BankState, Scenario, calculate, heuristic_recovery,
    optimize_recovery, reverse_runoff_threshold,
)


def _state() -> BankState:
    return BankState(deposits=100, assets=150, cash=10, securities=20,
                     borrowings=5, equity=15, loans=100)


def test_funding_need_is_nonnegative_and_monotone_in_runoff() -> None:
    low = calculate(_state(), Scenario(deposit_runoff_fraction_90d=.10))
    high = calculate(_state(), Scenario(deposit_runoff_fraction_90d=.50))
    assert low["peak_funding_need_proxy"] >= 0
    assert high["peak_funding_need_proxy"] >= low["peak_funding_need_proxy"]
    assert high["liquidity_runway_proxy_days"] <= low["liquidity_runway_proxy_days"]


def test_capital_loss_cannot_improve_funding_access() -> None:
    base = calculate(_state(), Scenario(deposit_runoff_fraction_90d=.35))
    weakened = calculate(_state(), Scenario(deposit_runoff_fraction_90d=.35,
                                            capital_asset_loss_fraction=.05))
    assert weakened["effective_secured_access_reduction"] >= base["effective_secured_access_reduction"]
    assert weakened["peak_funding_need_proxy"] >= base["peak_funding_need_proxy"]


def test_reverse_threshold_is_break_even_for_selected_horizon() -> None:
    scenario = Scenario(deposit_runoff_fraction_90d=0, commitment_draw_fraction_loans=.02)
    threshold = reverse_runoff_threshold(_state(), scenario, horizon_index=1)
    assert 0 < threshold < 1
    at_threshold = calculate(_state(), Scenario(**{**scenario.__dict__,
                                                   "deposit_runoff_fraction_90d": threshold}))
    assert abs(at_threshold["funding_gap"][1]) < 1e-8


def test_optimized_recovery_cost_no_more_than_feasible_heuristic() -> None:
    state = _state()
    optimized = optimize_recovery(state, 2, horizon_days=30)
    heuristic = heuristic_recovery(state, 2, horizon_days=30)
    assert optimized["residual_shortfall"] <= 1e-8
    assert heuristic["residual_shortfall"] <= 1e-8
    assert optimized["action_cost"] <= heuristic["action_cost"] + 1e-8
    assert np.sum(optimized["action_amounts"]) >= 2 - 1e-8
