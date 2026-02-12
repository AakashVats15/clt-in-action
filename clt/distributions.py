"""
This module provides:
- light-tailed distributions (normal, uniform, exponential)
- heavy-tailed distributions (Pareto, Cauchy)
- discrete distributions (Bernoulli, Poisson)
- skewed or bounded distributions

Each distribution is implemented as a function returning a callable
that matches the DistFn signature used in core.py.
"""

from __future__ import annotations

from typing import Callable
import numpy as np

ArrayLike = np.ndarray
DistFn = Callable[..., ArrayLike]


def _get_rng(rng: np.random.Generator | None) -> np.random.Generator:
    return rng if rng is not None else np.random.default_rng()



# Light-tailed distributions
def normal_dist(
    *,
    rng: np.random.Generator | None = None,
    loc: float = 0.0,
    scale: float = 1.0,
) -> DistFn:
    rng = _get_rng(rng)

    def dist_fn(*, size: tuple[int, ...]) -> ArrayLike:
        return rng.normal(loc=loc, scale=scale, size=size)

    return dist_fn


def uniform_dist(
    *,
    rng: np.random.Generator | None = None,
    low: float = 0.0,
    high: float = 1.0,
) -> DistFn:
    rng = _get_rng(rng)

    def dist_fn(*, size: tuple[int, ...]) -> ArrayLike:
        return rng.uniform(low=low, high=high, size=size)

    return dist_fn


def exponential_dist(
    *,
    rng: np.random.Generator | None = None,
    scale: float = 1.0,
) -> DistFn:
    rng = _get_rng(rng)

    def dist_fn(*, size: tuple[int, ...]) -> ArrayLike:
        return rng.exponential(scale=scale, size=size)

    return dist_fn



# Heavy-tailed distributions (CLT stress tests)
def pareto_dist(
    *,
    rng: np.random.Generator | None = None,
    shape: float = 2.0,
) -> DistFn:
    """
    Pareto distribution (heavy-tailed).
    Mean exists only if shape > 1.
    Variance exists only if shape > 2.
    """
    rng = _get_rng(rng)

    def dist_fn(*, size: tuple[int, ...]) -> ArrayLike:
        return rng.pareto(a=shape, size=size)

    return dist_fn


def cauchy_dist(
    *,
    rng: np.random.Generator | None = None,
    loc: float = 0.0,
    scale: float = 1.0,
) -> DistFn:
    """
    Cauchy distribution (infinite variance).
    CLT does NOT apply — useful for demonstrating failure cases.
    """
    rng = _get_rng(rng)

    def dist_fn(*, size: tuple[int, ...]) -> ArrayLike:
        return rng.standard_cauchy(size=size) * scale + loc

    return dist_fn



# Discrete distributions
def bernoulli_dist(
    *,
    rng: np.random.Generator | None = None,
    p: float = 0.5,
) -> DistFn:
    rng = _get_rng(rng)

    def dist_fn(*, size: tuple[int, ...]) -> ArrayLike:
        return rng.binomial(n=1, p=p, size=size)

    return dist_fn


def poisson_dist(
    *,
    rng: np.random.Generator | None = None,
    lam: float = 3.0,
) -> DistFn:
    rng = _get_rng(rng)

    def dist_fn(*, size: tuple[int, ...]) -> ArrayLike:
        return rng.poisson(lam=lam, size=size)

    return dist_fn


# Bounded / skewed distributions
def triangular_dist(
    *,
    rng: np.random.Generator | None = None,
    left: float = 0.0,
    mode: float = 0.5,
    right: float = 1.0,
) -> DistFn:
    rng = _get_rng(rng)

    def dist_fn(*, size: tuple[int, ...]) -> ArrayLike:
        return rng.triangular(left, mode, right, size=size)

    return dist_fn


def beta_dist(
    *,
    rng: np.random.Generator | None = None,
    a: float = 2.0,
    b: float = 5.0,
) -> DistFn:
    rng = _get_rng(rng)

    def dist_fn(*, size: tuple[int, ...]) -> ArrayLike:
        return rng.beta(a, b, size=size)

    return dist_fn


# Convenience factory
def get_distribution(name: str, **kwargs) -> DistFn:
    """
    Convenience function to fetch a distribution by name.

    Examples:
        dist = get_distribution("normal", loc=0, scale=1)
        dist = get_distribution("pareto", shape=3)
        dist = get_distribution("bernoulli", p=0.3)
    """
    name = name.lower()

    mapping = {
        "normal": normal_dist,
        "uniform": uniform_dist,
        "exponential": exponential_dist,
        "pareto": pareto_dist,
        "cauchy": cauchy_dist,
        "bernoulli": bernoulli_dist,
        "poisson": poisson_dist,
        "triangular": triangular_dist,
        "beta": beta_dist,
    }

    if name not in mapping:
        raise ValueError(
            f"Unknown distribution {name!r}. "
            f"Available: {', '.join(mapping.keys())}"
        )

    return mapping[name](**kwargs)