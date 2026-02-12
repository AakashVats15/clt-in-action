from __future__ import annotations

from typing import Iterable, Optional, Sequence
import os

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Matplotlib style defaults
plt.style.use("seaborn-v0_8-darkgrid")


def _ensure_dir(path: str) -> None:
    """Create parent directory if it doesn't exist."""
    parent = os.path.dirname(os.path.abspath(path))
    if parent and not os.path.exists(parent):
        os.makedirs(parent, exist_ok=True)


# ---------------------------
# Basic helpers
# ---------------------------
def save_or_show(fig: plt.Figure, filename: Optional[str]) -> None:
    """
    Save the figure to filename if provided, otherwise call plt.show().
    """
    if filename:
        _ensure_dir(filename)
        fig.savefig(filename, bbox_inches="tight", dpi=150)
        plt.close(fig)
    else:
        fig.show()


def _to_1d_array(x) -> np.ndarray:
    """Convert input (Series, DataFrame column, ndarray) to 1D numpy array."""
    if isinstance(x, pd.Series):
        return x.to_numpy()
    if isinstance(x, pd.DataFrame):
        # if DataFrame, flatten all values
        return x.to_numpy().ravel()
    return np.asarray(x).ravel()


# ---------------------------
# Histogram with normal overlay
# ---------------------------
def hist_with_normal_overlay(
    data,
    bins: int = 50,
    density: bool = True,
    alpha: float = 0.7,
    color: str = "steelblue",
    show_theoretical: bool = True,
    mean: Optional[float] = None,
    std: Optional[float] = None,
    xlabel: str = "Value",
    ylabel: str = "Density",
    title: Optional[str] = None,
    filename: Optional[str] = None,
) -> None:
    """
    Plot histogram of `data` and optionally overlay a normal PDF using provided mean/std.
    If mean/std are None and show_theoretical is True, they are computed from data.
    """
    x = _to_1d_array(data)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(x, bins=bins, density=density, alpha=alpha, color=color, edgecolor="black")

    if show_theoretical:
        if mean is None:
            mean = float(np.mean(x))
        if std is None:
            std = float(np.std(x, ddof=1))
        xs = np.linspace(x.min(), x.max(), 400)
        pdf = (1 / (std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((xs - mean) / std) ** 2)
        ax.plot(xs, pdf, color="darkred", lw=2, label="Normal PDF")
        ax.legend()

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    save_or_show(fig, filename)


# ---------------------------
# Convergence plot for sample means
# ---------------------------
def convergence_plot(
    results: dict[int, np.ndarray],
    bins: int = 50,
    filename: Optional[str] = None,
) -> None:
    """
    Plot histograms of sample means for multiple sample sizes side-by-side.
    `results` is a mapping sample_size -> array_of_sample_means.
    """
    sizes = sorted(results.keys())
    n = len(sizes)
    fig, axes = plt.subplots(nrows=1, ncols=n, figsize=(4 * n, 4), sharey=True)
    if n == 1:
        axes = [axes]
    for ax, s in zip(axes, sizes):
        data = results[s]
        ax.hist(data, bins=bins, density=True, alpha=0.7, color="C0", edgecolor="black")
        ax.set_title(f"n = {s}")
        ax.set_xlabel("Sample mean")
    axes[0].set_ylabel("Density")
    fig.suptitle("Convergence of Sample Means (CLT demonstration)")
    save_or_show(fig, filename)


# ---------------------------
# QQ plot (against normal)
# ---------------------------
def qq_plot(
    data,
    ax: Optional[plt.Axes] = None,
    title: Optional[str] = None,
    filename: Optional[str] = None,
) -> None:
    """
    Create a QQ-plot of `data` against a standard normal distribution.
    """
    x = _to_1d_array(data)
    x_sorted = np.sort(x)
    n = x_sorted.size
    # theoretical quantiles
    probs = (np.arange(1, n + 1) - 0.5) / n
    theo_q = np.sqrt(2) * np.erfinv(2 * probs - 1)  # inverse CDF of standard normal
    fig, ax_local = (plt.subplots(figsize=(6, 6)) if ax is None else (None, ax))
    ax_plot = ax_local if ax is None else ax
    ax_plot.scatter(theo_q, x_sorted, s=10, alpha=0.6)
    # reference line
    slope = np.std(x_sorted, ddof=1)
    intercept = np.mean(x_sorted)
    xs = np.array([theo_q.min(), theo_q.max()])
    ax_plot.plot(xs, intercept + slope * xs, color="red", lw=1.5, label="Reference")
    ax_plot.set_xlabel("Theoretical quantiles (Normal)")
    ax_plot.set_ylabel("Sample quantiles")
    if title:
        ax_plot.set_title(title)
    ax_plot.legend()
    if ax is None:
        save_or_show(fig, filename)


# ---------------------------
# Time series / GBM plotting
# ---------------------------
def plot_gbm_paths(
    times: np.ndarray,
    paths: np.ndarray,
    n_paths_to_plot: int = 20,
    alpha: float = 0.6,
    xlabel: str = "Time",
    ylabel: str = "Price",
    title: Optional[str] = "GBM sample paths",
    filename: Optional[str] = None,
) -> None:
    """
    Plot GBM paths.
    - times: 1D array of time points (n_steps + 1)
    - paths: shape (n_paths, n_steps + 1) or (n_paths, m, n_steps + 1) for multi-asset
    """
    arr = np.asarray(paths)
    if arr.ndim == 3:
        # choose first asset
        arr = arr[:, 0, :]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    n_paths = arr.shape[0]
    idx = np.linspace(0, n_paths - 1, min(n_paths, n_paths_to_plot)).astype(int)
    for i in idx:
        ax.plot(times, arr[i], lw=1, alpha=alpha)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    save_or_show(fig, filename)


# ---------------------------
# Portfolio return visualization
# ---------------------------
def plot_portfolio_return_distribution(
    returns,
    bins: int = 50,
    cumulative: bool = False,
    xlabel: str = "Return",
    title: Optional[str] = "Portfolio Return Distribution",
    filename: Optional[str] = None,
) -> None:
    """
    Plot histogram of portfolio returns and optionally the cumulative distribution.
    Accepts 1D array-like or pandas Series.
    """
    r = _to_1d_array(returns)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(r, bins=bins, density=not cumulative, cumulative=cumulative, alpha=0.7, color="C1", edgecolor="black")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Cumulative" if cumulative else "Density")
    if title:
        ax.set_title(title)
    save_or_show(fig, filename)


# ---------------------------
# OFDM PAPR histogram
# ---------------------------
def plot_papr_histogram(
    papr_values,
    bins: int = 60,
    xlabel: str = "PAPR (linear)",
    title: Optional[str] = "OFDM PAPR Distribution",
    filename: Optional[str] = None,
) -> None:
    """
    Plot histogram of PAPR values (linear scale). For readability, also show dB axis on top.
    """
    papr = _to_1d_array(papr_values)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(papr, bins=bins, color="purple", alpha=0.75, edgecolor="black")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    if title:
        ax.set_title(title)

    # secondary axis for dB
    def lin_to_db(x): return 10 * np.log10(x + 1e-12)
    secax = ax.secondary_xaxis('top', functions=(lin_to_db, lambda db: 10 ** (db / 10.0)))
    secax.set_xlabel("PAPR (dB)")
    save_or_show(fig, filename)


# ---------------------------
# Antenna array pattern plotting
# ---------------------------
def plot_array_pattern(
    weights,
    n_elements: int,
    angles_rad: Optional[np.ndarray] = None,
    element_spacing_wavelengths: float = 0.5,
    polar: bool = False,
    title: Optional[str] = "Array Pattern",
    filename: Optional[str] = None,
) -> None:
    """
    Plot array response power vs angle.
    - weights: complex weights (n_elements,)
    - angles_rad: array of angles in radians. If None, uses -pi/2..pi/2
    - polar: if True, produce a polar plot
    """
    w = np.asarray(weights, dtype=complex)
    if angles_rad is None:
        angles_rad = np.linspace(-np.pi / 2, np.pi / 2, 721)
    pattern = []
    for a in angles_rad:
        n = np.arange(n_elements)
        sv = np.exp(-2j * np.pi * element_spacing_wavelengths * n * np.sin(a))
        resp = np.vdot(w, sv)
        pattern.append(np.abs(resp) ** 2)
    pattern = np.array(pattern)
    # normalize
    pattern_db = 10 * np.log10(pattern / (pattern.max() + 1e-12) + 1e-12)

    if polar:
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, projection="polar")
        ax.plot(angles_rad, pattern_db)
        ax.set_title(title)
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        ax.set_rlabel_position(135)
        ax.set_ylabel("Normalized power (dB)")
    else:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.plot(np.degrees(angles_rad), pattern_db)
        ax.set_xlabel("Angle (degrees)")