"""
A compact but feature-rich example that demonstrates the Central Limit Theorem
using several underlying distributions. Designed to be readable and reusable.
"""

from __future__ import annotations

import argparse
import os
import sys
import pathlib
import numpy as np
from scipy.stats import norm

# Ensure repo root is on path
repo_root = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from clt.core import (
    clt_sample_means,
    clt_convergence_experiment,
    summarize_array,
    generate_distribution,
)
from clt.distributions import get_distribution
from clt import plotting

# Default output directory for plots
OUT_DIR = os.path.join(os.path.dirname(__file__), "plots")
os.makedirs(OUT_DIR, exist_ok=True)


def _seed_rng(seed: int | None) -> np.random.Generator:
    return np.random.default_rng(seed)


def demo_single_distribution(
    dist_name: str,
    dist_kwargs: dict,
    sample_size: int,
    num_trials: int,
    rng: np.random.Generator,
    save_plots: bool = True,
) -> None:
    print(f"\n=== Demo: {dist_name} (n={sample_size}, trials={num_trials}) ===")

    try:
        dist = get_distribution(dist_name, rng=rng, **dist_kwargs)
    except Exception:
        dist = generate_distribution(dist_name, rng=rng)

    result = clt_sample_means(
        dist_fn=dist,
        sample_size=sample_size,
        num_trials=num_trials,
        dist_kwargs={},
        return_sums=True,
    )

    means = result.sample_means
    sums = result.sample_sums

    print("Sample means summary:", summarize_array(means))
    if sums is not None:
        print("Sample sums summary:", summarize_array(sums))

    hist_file = os.path.join(OUT_DIR, f"hist_{dist_name}_n{sample_size}.png") if save_plots else None
    plotting.hist_with_normal_overlay(
        means,
        bins=60,
        show_theoretical=True,
        title=f"Sample Means (n={sample_size}) — {dist_name}",
        filename=hist_file,
    )

    qq_file = os.path.join(OUT_DIR, f"qq_{dist_name}_n{sample_size}.png") if save_plots else None
    plotting.qq_plot(means, title=f"QQ Plot of Sample Means — {dist_name}", filename=qq_file)


def demo_convergence(
    dist_name: str,
    dist_kwargs: dict,
    sample_sizes: list[int],
    num_trials: int,
    rng: np.random.Generator,
    save_plots: bool = True,
) -> None:
    print(f"\n=== Convergence experiment: {dist_name} ===")

    try:
        dist = get_distribution(dist_name, rng=rng, **dist_kwargs)
    except Exception:
        dist = generate_distribution(dist_name, rng=rng)

    results = clt_convergence_experiment(dist_fn=dist, sample_sizes=sample_sizes, num_trials=num_trials)

    for n, arr in results.items():
        stats = summarize_array(arr)
        print(f"n={n:3d} mean={stats['mean']:.4f} std={stats['std']:.4f}")

    conv_file = os.path.join(OUT_DIR, f"convergence_{dist_name}.png") if save_plots else None
    plotting.convergence_plot(results, bins=50, filename=conv_file)


def demo_mixture_noise(
    n_sources: int,
    samples: int,
    rng: np.random.Generator,
    save_plots: bool = True,
) -> None:
    print(f"\n=== Aggregated noise demo: {n_sources} sources, {samples} samples ===")

    gens = []
    for i in range(n_sources):
        if i % 3 == 0:
            gens.append(lambda shape, r, scale=0.5: r.normal(0.0, scale, size=shape))
        elif i % 3 == 1:
            gens.append(lambda shape, r, scale=0.3: r.exponential(scale=scale, size=shape) - scale)
        else:
            gens.append(lambda shape, r, p=0.05: (r.binomial(1, p, size=shape) - p) * 5.0)

    def make_gen(fn, seed_offset: int):
        def gen(shape, rng_local):
            return fn(shape, rng_local)
        return gen

    source_gens = [make_gen(g, i) for i, g in enumerate(gens)]

    from clt.communications import aggregate_noise_sources
    total = aggregate_noise_sources(source_gens * (n_sources // len(gens) + 1), shape=(samples,), rng=rng)

    print("Aggregated noise summary:", summarize_array(total))

    hist_file = os.path.join(OUT_DIR, f"aggregated_noise_{n_sources}.png") if save_plots else None
    plotting.hist_with_normal_overlay(total, bins=80, title="Aggregated Noise (many small sources)", filename=hist_file)

    qq_file = os.path.join(OUT_DIR, f"aggregated_noise_qq_{n_sources}.png") if save_plots else None
    plotting.qq_plot(total, title="QQ Plot — Aggregated Noise", filename=qq_file)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="run_basic_clt.py")
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--trials", type=int, default=10_000)
    p.add_argument("--sample-size", type=int, default=30)
    p.add_argument("--save-plots", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rng = _seed_rng(args.seed)

    demo_single_distribution("normal", {"loc": 0.0, "scale": 1.0}, args.sample_size, args.trials, rng, args.save_plots)
    demo_single_distribution("uniform", {"low": -1.0, "high": 1.0}, args.sample_size, args.trials, rng, args.save_plots)
    demo_single_distribution("exponential", {"scale": 1.0}, args.sample_size, args.trials, rng, args.save_plots)
    demo_single_distribution("pareto", {"shape": 1.8}, args.sample_size, args.trials, rng, args.save_plots)

    sample_sizes = [1, 2, 5, 10, 30, 100]
    demo_convergence("exponential", {"scale": 1.0}, sample_sizes, 5000, rng, args.save_plots)

    demo_mixture_noise(60, 200_000, rng, args.save_plots)

    print("\nDone. If save_plots was enabled, check the 'plots' folder for images.")


if __name__ == "__main__":
    main()