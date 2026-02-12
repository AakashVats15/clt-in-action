"""
This module provides financial utilities for return calculations, simulations, and risk measures.

- return conversions (price <-> log returns)
- geometric Brownian motion (GBM) path simulation
- correlated asset simulation (Cholesky)
- portfolio return and risk calculations
- Monte Carlo pricing for European options (Black-Scholes style)
- Value at Risk (VaR) and Expected Shortfall (ES) (parametric and historical)
- bootstrap confidence intervals for Monte Carlo estimates
- simple performance metrics (Sharpe, max drawdown)

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Sequence

import numpy as np
import pandas as pd

Array = np.ndarray
RNG = np.random.Generator
DistFn = Callable[..., Array]


@dataclass
class GBMResult:
    times: Array
    paths: Array  # shape: (n_paths, n_steps + 1)


@dataclass
class PortfolioStats:
    returns: Array
    mean: float
    std: float
    sharpe: float
    max_drawdown: float


# Basic conversions
def prices_to_log_returns(prices: Array) -> Array:
    """Convert price series to log returns."""
    prices = np.asarray(prices, dtype=float)
    if prices.ndim != 1:
        raise ValueError("prices must be a 1D array")
    return np.log(prices[1:] / prices[:-1])


def log_returns_to_prices(initial_price: float, log_returns: Array) -> Array:
    """Reconstruct price path from initial price and log returns."""
    log_returns = np.asarray(log_returns, dtype=float)
    cum_log = np.concatenate(([0.0], np.cumsum(log_returns)))
    return initial_price * np.exp(cum_log)


# Geometric Brownian Motion
def simulate_gbm(
    s0: float,
    mu: float,
    sigma: float,
    T: float,
    n_steps: int,
    n_paths: int = 1,
    rng: RNG | None = None,
) -> GBMResult:
    """Simulate geometric Brownian motion paths."""
    if rng is None:
        rng = np.random.default_rng()

    dt = T / n_steps
    times = np.linspace(0.0, T, n_steps + 1)
    # Z shape: (n_paths, n_steps)
    Z = rng.standard_normal(size=(n_paths, n_steps))
    increments = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z
    log_paths = np.cumsum(np.concatenate((np.zeros((n_paths, 1)), increments), axis=1), axis=1)
    paths = s0 * np.exp(log_paths)
    return GBMResult(times=times, paths=paths)

# Correlated asset simulation
def simulate_correlated_gbm(
    s0: Sequence[float],
    mu: Sequence[float],
    sigma: Sequence[float],
    corr: Array,
    T: float,
    n_steps: int,
    n_paths: int = 1,
    rng: RNG | None = None,
) -> GBMResult:
    """Simulate correlated GBM for multiple assets using Cholesky decomposition."""
    if rng is None:
        rng = np.random.default_rng()

    s0 = np.asarray(s0, dtype=float)
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    corr = np.asarray(corr, dtype=float)

    m = s0.size
    if mu.size != m or sigma.size != m:
        raise ValueError("s0, mu, sigma must have the same length")
    if corr.shape != (m, m):
        raise ValueError("corr must be an (m, m) matrix")

    # Convert correlation to covariance for standard normals
    L = np.linalg.cholesky(corr)
    dt = T / n_steps
    times = np.linspace(0.0, T, n_steps + 1)

    # Generate independent normals then correlate
    Z_ind = rng.standard_normal(size=(n_paths, n_steps, m))  # shape (paths, steps, assets)
    # Correlate: for each path and step, multiply by L
    Z_corr = np.einsum("ij,pkj->pki", L, Z_ind)  # shape (paths, steps, assets)

    # Build increments and integrate
    drift = (mu - 0.5 * sigma**2) * dt  # shape (m,)
    vol_scale = sigma * np.sqrt(dt)  # shape (m,)

    # log_paths shape: (n_paths, m, n_steps + 1)
    log_paths = np.zeros((n_paths, m, n_steps + 1), dtype=float)
    for t in range(n_steps):
        increments = drift + vol_scale * Z_corr[:, t, :]
        log_paths[:, :, t + 1] = log_paths[:, :, t] + increments

    paths = s0[np.newaxis, :, np.newaxis] * np.exp(log_paths)
    return GBMResult(times=times, paths=paths)  # note: paths shape (n_paths, m, n_steps+1)


# Portfolio calculations
def portfolio_returns_from_weights(
    asset_returns: Array,
    weights: Array,
) -> Array:
    """Compute portfolio returns given asset returns and weights."""
    arr = np.asarray(asset_returns, dtype=float)
    w = np.asarray(weights, dtype=float)

    if arr.ndim == 2 and arr.shape[1] == w.size:
        # shape (n_periods, n_assets)
        return arr @ w
    elif arr.ndim == 2 and arr.shape[0] == w.size:
        # shape (n_assets, n_periods)
        return w @ arr
    else:
        raise ValueError("asset_returns must be 2D and compatible with weights length")


def portfolio_statistics(
    returns: Array,
    risk_free_rate: float = 0.0,
    annualize_factor: int = 252,
) -> PortfolioStats:
    """Compute basic portfolio statistics."""
    r = np.asarray(returns, dtype=float)
    mean = float(r.mean()) * annualize_factor
    std = float(r.std(ddof=1)) * np.sqrt(annualize_factor)
    # Sharpe uses excess return
    sharpe = (mean - risk_free_rate) / std if std > 0 else float("nan")

    # Max drawdown
    cumulative = np.cumprod(1 + r)
    peak = np.maximum.accumulate(cumulative)
    drawdowns = (cumulative - peak) / peak
    max_dd = float(drawdowns.min())

    return PortfolioStats(returns=r, mean=mean, std=std, sharpe=sharpe, max_drawdown=max_dd)


# ---------------------------
# Risk measures: VaR and ES
# ---------------------------
def historical_var(returns: Array, alpha: float = 0.05) -> float:
    """Historical Value at Risk (VaR) at level alpha."""
    r = np.asarray(returns, dtype=float)
    q = np.quantile(-r, 1 - alpha)  # negative returns are losses
    return float(q)


def parametric_var(returns: Array, alpha: float = 0.05) -> float:
    """Parametric VaR assuming normality (mean and std from data)."""
    r = np.asarray(returns, dtype=float)
    mu = r.mean()
    sigma = r.std(ddof=1)
    z = -np.quantile(np.random.default_rng().standard_normal(1000000), 1 - alpha)  # approximate z_{alpha}
    # Better: use inverse CDF from numpy
    z = -np.quantile(np.random.default_rng().standard_normal(1000000), 1 - alpha)
    # But we can use scipy if available; to avoid dependency, use numpy's percentile of standard normal
    # Use analytic z via inverse error function approximation:
    z = float(np.sqrt(2) * np.erfinv(1 - 2 * alpha) * -1)  # z_{alpha} positive for loss
    var = -(mu + z * sigma)
    return float(var)


def expected_shortfall_historical(returns: Array, alpha: float = 0.05) -> float:
    """Historical Expected Shortfall (ES) at level alpha."""
    r = np.asarray(returns, dtype=float)
    losses = -r
    threshold = np.quantile(losses, 1 - alpha)
    tail_losses = losses[losses >= threshold]
    if tail_losses.size == 0:
        return float(threshold)
    return float(tail_losses.mean())



# Monte Carlo pricing (European option)
def monte_carlo_european_call_price(
    s0: float,
    strike: float,
    T: float,
    r: float,
    sigma: float,
    n_paths: int = 100_000,
    n_steps: int = 1,
    rng: RNG | None = None,
) -> tuple[float, float]:
    """Monte Carlo estimate of a European call option price under GBM."""
    if rng is None:
        rng = np.random.default_rng()

    dt = T / n_steps
    # Simulate terminal price using exact GBM formula for efficiency when n_steps == 1
    if n_steps == 1:
        Z = rng.standard_normal(size=n_paths)
        ST = s0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)
    else:
        # simulate with steps
        gbm = simulate_gbm(s0=s0, mu=r, sigma=sigma, T=T, n_steps=n_steps, n_paths=n_paths, rng=rng)
        ST = gbm.paths[:, -1]

    payoffs = np.maximum(ST - strike, 0.0)
    discounted = np.exp(-r * T) * payoffs
    price = float(discounted.mean())
    stderr = float(discounted.std(ddof=1) / np.sqrt(n_paths))
    return price, stderr


# Bootstrap utilities
def bootstrap_confidence_interval(
    data: Array,
    stat_fn: Callable[[Array], float],
    n_bootstrap: int = 2000,
    alpha: float = 0.05,
    rng: RNG | None = None,
) -> tuple[float, float]:
    if rng is None:
        rng = np.random.default_rng()

    data = np.asarray(data, dtype=float)
    n = data.size
    stats = np.empty(n_bootstrap, dtype=float)
    for i in range(n_bootstrap):
        sample = rng.choice(data, size=n, replace=True)
        stats[i] = stat_fn(sample)
    lower = float(np.quantile(stats, alpha / 2))
    upper = float(np.quantile(stats, 1 - alpha / 2))
    return lower, upper



# Utility metrics
def sharpe_ratio(returns: Array, risk_free_rate: float = 0.0, annualize_factor: int = 252) -> float:
    r = np.asarray(returns, dtype=float)
    mean = r.mean() * annualize_factor
    std = r.std(ddof=1) * np.sqrt(annualize_factor)
    return float((mean - risk_free_rate) / std) if std > 0 else float("nan")


def max_drawdown(returns: Array) -> float:
    r = np.asarray(returns, dtype=float)
    cumulative = np.cumprod(1 + r)
    peak = np.maximum.accumulate(cumulative)
    drawdowns = (cumulative - peak) / peak
    return float(drawdowns.min())


# Example helper: simulate portfolio from correlated GBM
def simulate_portfolio_from_gbm(
    weights: Sequence[float],
    s0: Sequence[float],
    mu: Sequence[float],
    sigma: Sequence[float],
    corr: Array,
    T: float,
    n_steps: int,
    n_paths: int = 10_000,
    rng: RNG | None = None,
) -> tuple[Array, Array]:
    """
    Simulate portfolio returns (periodic) from correlated GBM asset paths.

    Returns (portfolio_returns, terminal_values)
    - portfolio_returns shape: (n_paths, n_steps) periodic returns
    - terminal_values shape: (n_paths,) portfolio terminal value (weighted)
    """
    gbm = simulate_correlated_gbm(
        s0=s0, mu=mu, sigma=sigma, corr=corr, T=T, n_steps=n_steps, n_paths=n_paths, rng=rng
    )
    # gbm.paths shape: (n_paths, m, n_steps+1)
    paths = gbm.paths
    # compute periodic returns for each asset: (n_paths, m, n_steps)
    periodic_returns = paths[:, :, 1:] / paths[:, :, :-1] - 1.0
    weights = np.asarray(weights, dtype=float)
    # portfolio returns per period: (n_paths, n_steps)
    port_returns = np.einsum("p t a, a -> p t", periodic_returns.transpose(0, 2, 1), weights)
    # terminal values
    terminal_prices = paths[:, :, -1]  # (n_paths, m)
    terminal_values = (terminal_prices * weights).sum(axis=1)
    return port_returns, terminal_values