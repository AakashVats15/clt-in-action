from __future__ import annotations

import argparse
import pathlib
import sys
from datetime import datetime

import numpy as np

# repo root
repo_root = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from clt.finance import (
    simulate_gbm_paths,
    compute_log_returns,
    compute_portfolio_returns,
    compute_covariance_matrix,
    compute_var_cvar,
)
from clt.plotting import (
    plot_gbm_paths,
    plot_portfolio_return_distribution,
    hist_with_normal_overlay,
)

# timestamped output directory
timestamp = datetime.now().strftime("%Y%m%d_%H-%M")
script_name = "Finance_Demo"
OUT_DIR = repo_root / "plots" / f"{script_name}_{timestamp}"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _rng(seed: int | None) -> np.random.Generator:
    return np.random.default_rng(seed)


def demo_gbm(
    n_assets: int,
    n_paths: int,
    n_steps: int,
    T: float,
    mu: float,
    sigma: float,
    rng: np.random.Generator,
):
    times, paths = simulate_gbm_paths(
        n_assets=n_assets,
        n_paths=n_paths,
        n_steps=n_steps,
        T=T,
        mu=mu,
        sigma=sigma,
        rng=rng,
    )
    fig_file = OUT_DIR / "gbm_paths.png"
    plot_gbm_paths(times, paths, n_paths_to_plot=20, filename=fig_file)
    return times, paths


def demo_portfolio(paths: np.ndarray, weights: np.ndarray):
    logret = compute_log_returns(paths)
    port = compute_portfolio_returns(logret, weights)
    cov = compute_covariance_matrix(logret)
    flat_port = port.flatten()
    var, cvar = compute_var_cvar(flat_port, alpha=0.95)


    hist_file = OUT_DIR / "portfolio_hist.png"
    plot_portfolio_return_distribution(port, bins=60, filename=hist_file)

    norm_file = OUT_DIR / "portfolio_normal_overlay.png"
    hist_with_normal_overlay(
        port,
        bins=60,
        title="Portfolio Return Distribution with Normal Overlay",
        filename=norm_file,
    )

    return {
        "mean": float(np.mean(port)),
        "std": float(np.std(port, ddof=1)),
        "var_95": float(var),
        "cvar_95": float(cvar),
        "covariance": cov,
    }


def demo_forward_simulation(
    mu: float,
    sigma: float,
    S0: float,
    horizon: float,
    n_sims: int,
    rng: np.random.Generator,
):
    dt = horizon
    shocks = rng.normal(mu * dt, sigma * np.sqrt(dt), size=n_sims)
    prices = S0 * np.exp(shocks)

    file = OUT_DIR / "forward_price_distribution.png"
    hist_with_normal_overlay(
        prices,
        bins=80,
        title="Forward Price Distribution",
        filename=file,
    )

    return {
        "mean": float(np.mean(prices)),
        "std": float(np.std(prices, ddof=1)),
        "min": float(np.min(prices)),
        "max": float(np.max(prices)),
    }


def parse_args():
    p = argparse.ArgumentParser(prog="run_finance_demo.py")
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--assets", type=int, default=4)
    p.add_argument("--paths", type=int, default=2000)
    p.add_argument("--steps", type=int, default=252)
    p.add_argument("--T", type=float, default=1.0)
    p.add_argument("--mu", type=float, default=0.08)
    p.add_argument("--sigma", type=float, default=0.25)
    p.add_argument("--save-plots", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    rng = _rng(args.seed)

    times, paths = demo_gbm(
        n_assets=args.assets,
        n_paths=args.paths,
        n_steps=args.steps,
        T=args.T,
        mu=args.mu,
        sigma=args.sigma,
        rng=rng,
    )

    weights = np.ones(args.assets) / args.assets
    stats = demo_portfolio(paths, weights)

    forward = demo_forward_simulation(
        mu=args.mu,
        sigma=args.sigma,
        S0=100.0,
        horizon=1 / 12,
        n_sims=50000,
        rng=rng,
    )

    print("\nPortfolio statistics:")
    for k, v in stats.items():
        if k != "covariance":
            print(f"{k}: {v:.6f}")
    print("\nCovariance matrix:\n", stats["covariance"])

    print("\nForward simulation summary:")
    for k, v in forward.items():
        print(f"{k}: {v:.6f}")

    print(f"\nPlots saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()