from __future__ import annotations

import argparse
import pathlib
import sys
from datetime import datetime

import numpy as np

repo_root = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from clt.communications import (
    aggregate_noise_sources,
    awgn_channel,
    qpsk_modulate,
    qpsk_demodulate,
    qam16_modulate,
    qam16_demodulate,
    qam64_modulate,
    qam64_demodulate,
    ber,
    rayleigh_fading,
    ofdm_modulate,
    ofdm_demodulate,
    mimo_alamouti_encode,
    mimo_alamouti_decode,
)
from clt.plotting import (
    hist_with_normal_overlay,
    plot_constellation,
    plot_ber_curve,
    plot_psd,
)

timestamp = datetime.now().strftime("%Y%m%d_%H-%M")
script_name = "Communications_Demo"
OUT_DIR = repo_root / "plots" / f"{script_name}_{timestamp}"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _rng(seed: int | None) -> np.random.Generator:
    return np.random.default_rng(seed)


# -----------------------------
# Noise Aggregation
# -----------------------------
def demo_noise_aggregation(n_sources: int, samples: int, rng):
    gens = []
    for i in range(n_sources):
        if i % 3 == 0:
            gens.append(lambda shape, r, s=0.5: r.normal(0, s, size=shape))
        elif i % 3 == 1:
            gens.append(lambda shape, r, s=0.3: r.exponential(scale=s, size=shape) - s)
        else:
            gens.append(lambda shape, r, p=0.05: (r.binomial(1, p, size=shape) - p) * 5)

    def wrap(fn):
        return lambda shape, r: fn(shape, r)

    source_gens = [wrap(g) for g in gens]
    total = aggregate_noise_sources(
        source_gens * (n_sources // len(gens) + 1),
        shape=(samples,),
        rng=rng,
    )

    file = OUT_DIR / "aggregated_noise.png"
    hist_with_normal_overlay(total, bins=80, title="Aggregated Noise", filename=file)
    return total


# -----------------------------
# QPSK Constellation
# -----------------------------
def demo_qpsk_constellation(n_symbols: int, snr_db: float, rng):
    bits = rng.integers(0, 2, size=(n_symbols, 2))
    symbols = qpsk_modulate(bits)
    noisy = awgn_channel(symbols, snr_db, rng)

    file = OUT_DIR / f"qpsk_constellation_{snr_db}dB.png"
    plot_constellation(noisy, title=f"QPSK @ {snr_db} dB", filename=file)
    return bits, noisy


# -----------------------------
# BER Curve
# -----------------------------
def demo_ber_curve(n_symbols: int, snr_range: list[float], rng):
    bits = rng.integers(0, 2, size=(n_symbols, 2))
    symbols = qpsk_modulate(bits)

    bers = []
    for snr in snr_range:
        noisy = awgn_channel(symbols, snr, rng)
        decoded = qpsk_demodulate(noisy)
        bers.append(ber(bits, decoded))

    file = OUT_DIR / "ber_curve.png"
    plot_ber_curve(snr_range, bers, title="BER vs SNR (QPSK)", filename=file)
    return bers


# -----------------------------
# Rayleigh Fading
# -----------------------------
def demo_rayleigh_fading(n_symbols: int, snr_db: float, rng):
    bits = rng.integers(0, 2, size=(n_symbols, 2))
    symbols = qpsk_modulate(bits)

    h = rayleigh_fading(n_symbols, rng)
    faded = symbols * h
    noisy = awgn_channel(faded, snr_db, rng)
    equalized = noisy / h

    file = OUT_DIR / "rayleigh_constellation.png"
    plot_constellation(equalized, title="Rayleigh Fading Equalized", filename=file)

    decoded = qpsk_demodulate(equalized)
    return ber(bits, decoded)


# -----------------------------
# OFDM Demo
# -----------------------------
def demo_ofdm(n_symbols: int, n_subcarriers: int, snr_db: float, rng):
    bits = rng.integers(0, 2, size=(n_symbols, 2))
    symbols = qpsk_modulate(bits)

    # --- FIX: ensure length is divisible by n_subcarriers ---
    remainder = symbols.size % n_subcarriers
    if remainder != 0:
        symbols = symbols[: symbols.size - remainder]  # trim extra symbols

    ofdm_signal = ofdm_modulate(symbols, n_subcarriers)
    noisy = awgn_channel(ofdm_signal, snr_db, rng)
    recovered = ofdm_demodulate(noisy, n_subcarriers)

    file = OUT_DIR / "ofdm_constellation.png"
    plot_constellation(recovered, title="OFDM QPSK", filename=file)

    decoded = qpsk_demodulate(recovered)
    bits_used = bits[: decoded.shape[0]]  # align lengths
    return ber(bits_used, decoded)


# -----------------------------
# MIMO 2×2 Alamouti
# -----------------------------
def demo_mimo(n_symbols: int, snr_db: float, rng):
    bits = rng.integers(0, 2, size=(n_symbols, 2))
    symbols = qpsk_modulate(bits)

    tx1, tx2 = mimo_alamouti_encode(symbols)
    h1 = rayleigh_fading(n_symbols, rng)
    h2 = rayleigh_fading(n_symbols, rng)

    r1 = h1 * tx1 + h2 * tx2
    r2 = -h2.conjugate() * tx1 + h1.conjugate() * tx2

    noisy1 = awgn_channel(r1, snr_db, rng)
    noisy2 = awgn_channel(r2, snr_db, rng)

    decoded_symbols = mimo_alamouti_decode(noisy1, noisy2, h1, h2)
    file = OUT_DIR / "mimo_constellation.png"
    plot_constellation(decoded_symbols, title="MIMO 2x2 Alamouti", filename=file)

    decoded_bits = qpsk_demodulate(decoded_symbols)
    return ber(bits, decoded_bits)


# -----------------------------
# Adaptive Modulation
# -----------------------------
def demo_adaptive_modulation(n_symbols: int, snr_db: float, rng):
    bits = rng.integers(0, 2, size=(n_symbols, 6))

    if snr_db < 8:
        mod = qpsk_modulate
        demod = qpsk_demodulate
        used = bits[:, :2]
    elif snr_db < 16:
        mod = qam16_modulate
        demod = qam16_demodulate
        used = bits[:, :4]
    else:
        mod = qam64_modulate
        demod = qam64_demodulate
        used = bits[:, :6]

    symbols = mod(used)
    noisy = awgn_channel(symbols, snr_db, rng)

    file = OUT_DIR / f"adaptive_modulation_{snr_db}dB.png"
    plot_constellation(noisy, title=f"Adaptive Modulation @ {snr_db} dB", filename=file)

    decoded = demod(noisy)
    return ber(used, decoded)


# -----------------------------
# PSD Estimation
# -----------------------------
def demo_psd(signal: np.ndarray):
    file = OUT_DIR / "psd.png"
    plot_psd(signal, title="Power Spectral Density", filename=file)


# -----------------------------
# Main
# -----------------------------
def parse_args():
    p = argparse.ArgumentParser(prog="run_communications_demo.py")
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--samples", type=int, default=200000)
    p.add_argument("--sources", type=int, default=60)
    p.add_argument("--symbols", type=int, default=50000)
    return p.parse_args()


def main():
    args = parse_args()
    rng = _rng(args.seed)

    noise = demo_noise_aggregation(args.sources, args.samples, rng)
    bits, noisy = demo_qpsk_constellation(args.symbols, 10, rng)

    snr_range = list(range(0, 21, 2))
    bers = demo_ber_curve(args.symbols, snr_range, rng)

    rayleigh_ber = demo_rayleigh_fading(args.symbols, 10, rng)
    ofdm_ber = demo_ofdm(args.symbols, 256, 12, rng)
    mimo_ber = demo_mimo(args.symbols, 12, rng)
    adaptive_ber = demo_adaptive_modulation(args.symbols, 14, rng)

    demo_psd(noisy)

    print("\nRayleigh BER:", rayleigh_ber)
    print("OFDM BER:", ofdm_ber)
    print("MIMO BER:", mimo_ber)
    print("Adaptive BER:", adaptive_ber)
    print(f"\nPlots saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()