from __future__ import annotations

import argparse
import pathlib
import sys
from datetime import datetime

import numpy as np

repo_root = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from clt.communications import (
    simulate_interference_aggregate,
    rayleigh_multipath_channel,
    apply_multipath_to_signal,
    ula_steering_vector,
    array_response_power,
    awgn_noise,
)
from clt.plotting import hist_with_normal_overlay
import matplotlib.pyplot as plt

timestamp = datetime.now().strftime("%Y%m%d_%H-%M")
script_name = "Aircraft_Antenna_Noise"
OUT_DIR = repo_root / "plots" / f"{script_name}_{timestamp}"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _rng(seed: int | None) -> np.random.Generator:
    return np.random.default_rng(seed)


def db(x: np.ndarray) -> np.ndarray:
    return 10 * np.log10(np.maximum(x, 1e-15))


def demo_array_pattern(
    n_elements: int,
    spacing_lambda: float,
    look_angle_deg: float,
    interferer_angles_deg: list[float],
):
    angles = np.linspace(-90, 90, 721) * np.pi / 180
    w = ula_steering_vector(n_elements, np.deg2rad(look_angle_deg), spacing_lambda)
    pattern = array_response_power(w, angles, n_elements, spacing_lambda)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(angles * 180 / np.pi, db(pattern / pattern.max()))
    for a in interferer_angles_deg:
        ax.axvline(a, color="red", ls="--", alpha=0.5)
    ax.set_xlabel("Angle (deg)")
    ax.set_ylabel("Array gain (dB, normalized)")
    ax.set_title("Aircraft ULA Pattern vs Angle")
    ax.grid(True, ls="--", alpha=0.4)

    fig.savefig(OUT_DIR / "array_pattern.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def demo_interference_vs_angle(
    n_elements: int,
    spacing_lambda: float,
    interferer_angles_deg: list[float],
    samples: int,
    rng: np.random.Generator,
):
    angles = np.linspace(-90, 90, 181)
    powers = []

    for look in angles:
        w = ula_steering_vector(n_elements, np.deg2rad(look), spacing_lambda)
        total = np.zeros(samples, dtype=complex)
        for a in interferer_angles_deg:
            phase = rng.uniform(0, 2 * np.pi, size=samples)
            amp = rng.rayleigh(scale=1.0, size=samples)
            sig = amp * np.exp(1j * phase)
            sv = ula_steering_vector(n_elements, np.deg2rad(a), spacing_lambda)
            gain = np.vdot(w, sv)
            total += gain * sig
        powers.append(np.mean(np.abs(total) ** 2))

    powers = np.array(powers)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(angles, db(powers / powers.max()))
    ax.set_xlabel("Look angle (deg)")
    ax.set_ylabel("Relative interference power (dB)")
    ax.set_title("Interference Power vs Look Angle (Aircraft ULA)")
    ax.grid(True, ls="--", alpha=0.4)

    fig.savefig(OUT_DIR / "interference_vs_angle.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def demo_air_to_ground_link(
    carrier_freq_hz: float,
    tx_power_dbm: float,
    noise_figure_db: float,
    altitudes_m: np.ndarray,
    distance_ground_m: float,
    samples: int,
    rng: np.random.Generator,
):
    c = 3e8
    wavelength = c / carrier_freq_hz
    k = 1.38e-23
    T = 290.0
    B = 1e6

    tx_power_w = 1e-3 * 10 ** (tx_power_dbm / 10)
    noise_figure_lin = 10 ** (noise_figure_db / 10)
    N0 = k * T * B * noise_figure_lin

    snr_lin = []
    for h in altitudes_m:
        d = np.sqrt(h**2 + distance_ground_m**2)
        path_loss = (4 * np.pi * d / wavelength) ** 2
        rx_power = tx_power_w / path_loss

        h_taps = rayleigh_multipath_channel(n_paths=3, samples=1, avg_power=1.0, rng=rng)[0]
        tx = np.ones(samples, dtype=complex)
        rx = apply_multipath_to_signal(tx, h_taps[np.newaxis, :])
        noise = awgn_noise(rx.shape, sigma=np.sqrt(N0 / 2), rng=rng) + 1j * awgn_noise(
            rx.shape, sigma=np.sqrt(N0 / 2), rng=rng
        )
        rx_total = np.sqrt(rx_power) * rx + noise

        sig_power = np.mean(np.abs(np.sqrt(rx_power) * rx) ** 2)
        noise_power = np.mean(np.abs(noise) ** 2)
        snr_lin.append(sig_power / noise_power)

    snr_lin = np.array(snr_lin)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(altitudes_m / 1000, 10 * np.log10(snr_lin))
    ax.set_xlabel("Altitude (km)")
    ax.set_ylabel("SNR (dB)")
    ax.set_title("Air-to-Ground Link SNR vs Altitude")
    ax.grid(True, ls="--", alpha=0.4)

    fig.savefig(OUT_DIR / "snr_vs_altitude.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    return rx_total


def demo_received_histogram(rx_signal: np.ndarray):
    hist_with_normal_overlay(
        rx_signal.real,
        bins=80,
        title="Received In-Phase Component (Aircraft Link)",
        filename=OUT_DIR / "rx_histogram.png",
    )


def parse_args():
    p = argparse.ArgumentParser(prog="run_aircraft_antenna_noise.py")
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--elements", type=int, default=8)
    p.add_argument("--spacing", type=float, default=0.5)
    p.add_argument("--look-angle", type=float, default=30.0)
    p.add_argument("--samples", type=int, default=200000)
    return p.parse_args()


def main():
    args = parse_args()
    rng = _rng(args.seed)

    interferers = [-60.0, -20.0, 10.0, 45.0]

    demo_array_pattern(
        n_elements=args.elements,
        spacing_lambda=args.spacing,
        look_angle_deg=args.look_angle,
        interferer_angles_deg=interferers,
    )

    demo_interference_vs_angle(
        n_elements=args.elements,
        spacing_lambda=args.spacing,
        interferer_angles_deg=interferers,
        samples=args.samples,
        rng=rng,
    )

    altitudes = np.linspace(1e3, 12e3, 20)
    rx = demo_air_to_ground_link(
        carrier_freq_hz=1.5e9,
        tx_power_dbm=30.0,
        noise_figure_db=5.0,
        altitudes_m=altitudes,
        distance_ground_m=30e3,
        samples=args.samples,
        rng=rng,
    )

    demo_received_histogram(rx)

    print(f"\nPlots saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()