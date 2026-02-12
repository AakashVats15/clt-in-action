from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Literal
import numpy as np


ArrayLike = np.ndarray
DistFn = Callable[..., ArrayLike]


@dataclass
class CLTResult:
    sample_means: ArrayLike
    sample_sums: ArrayLike | None
    mean_theoretical: float | None
    var_theoretical: float | None


def clt_sample_means(
    dist_fn: DistFn,
    sample_size: int = 30,
    num_trials: int = 10_000,
    *,
    dist_kwargs: dict | None = None,
    return_sums: bool = False,
    mean_theoretical: float | None = None,
    var_theoretical: float | None = None,
) -> CLTResult:

    if dist_kwargs is None:
        dist_kwargs = {}

    # Shape: (num_trials, sample_size)
    samples = dist_fn(size=(num_trials, sample_size), **dist_kwargs)

    # Means across axis=1 → one mean per trial
    sample_means = samples.mean(axis=1)

    sample_sums: ArrayLike | None = None
    if return_sums:
        sample_sums = samples.sum(axis=1)

    return CLTResult(
        sample_means=sample_means,
        sample_sums=sample_sums,
        mean_theoretical=mean_theoretical,
        var_theoretical=var_theoretical,
    )


def clt_convergence_experiment(
    dist_fn: DistFn,
    sample_sizes: list[int],
    num_trials: int = 5_000,
    *,
    dist_kwargs: dict | None = None,
) -> dict[int, ArrayLike]:

    if dist_kwargs is None:
        dist_kwargs = {}

    results: dict[int, ArrayLike] = {}

    for n in sample_sizes:
        samples = dist_fn(size=(num_trials, n), **dist_kwargs)
        means = samples.mean(axis=1)
        results[n] = means

    return results


def summarize_array(
    x: ArrayLike,
    *,
    ddof: int = 1,
) -> dict[str, float]:

    x = np.asarray(x)

    return {
        "mean": float(x.mean()),
        "var": float(x.var(ddof=ddof)),
        "std": float(x.std(ddof=ddof)),
        "min": float(x.min()),
        "max": float(x.max()),
    }


def generate_distribution(
    kind: Literal["exponential", "uniform", "normal"],
    *,
    rng: np.random.Generator | None = None,
) -> DistFn:

    if rng is None:
        rng = np.random.default_rng()

    if kind == "exponential":
        def dist_fn(*, size: tuple[int, ...], scale: float = 1.0) -> ArrayLike:
            return rng.exponential(scale=scale, size=size)
    elif kind == "uniform":
        def dist_fn(*, size: tuple[int, ...], low: float = 0.0, high: float = 1.0) -> ArrayLike:
            return rng.uniform(low=low, high=high, size=size)
    elif kind == "normal":
        def dist_fn(*, size: tuple[int, ...], loc: float = 0.0, scale: float = 1.0) -> ArrayLike:
            return rng.normal(loc=loc, scale=scale, size=size)
    else:
        msg = f"Unknown distribution kind: {kind!r}. Expected 'exponential', 'uniform', or 'normal'."
        raise ValueError(msg)

    return dist_fn