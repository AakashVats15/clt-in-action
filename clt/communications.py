"""
This module provides utilities for simulating basic communication system components.

- AWGN generation and aggregated-noise modeling (CLT justification)
- Interference aggregation utilities
- Multipath channel simulators (Rayleigh, Rician)
- OFDM signal generator (time-domain) and basic PAPR computation
- Antenna array steering and array response (ULA)
- Simple MIMO channel simulation (flat fading)
- SNR, matched-filter detection, and BER estimators for BPSK/QPSK
- Monte Carlo BER simulation harness
- All functions accept a numpy RNG for reproducibility

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence, Tuple

import numpy as np
from scipy.special import erfc  # for Q-function via erfc

Array = np.ndarray
RNG = np.random.Generator


# ---------------------------
# Data containers
# ---------------------------
@dataclass
class ChannelResult:
    """Container for channel simulation outputs."""
    received: Array
    channel_matrix: Array | None
    noise: Array | None


@dataclass
class OFDMResult:
    """Container for OFDM signal outputs."""
    time_signal: Array
    symbols: Array
    ofdm_symbols: Array
    cp_length: int


# ---------------------------
# Utility: Q-function
# ---------------------------
def qfunc(x: Array) -> Array:
    """
    Q-function using complementary error function.

    Q(x) = 0.5 * erfc(x / sqrt(2))
    """
    return 0.5 * erfc(x / np.sqrt(2))


# ---------------------------
# AWGN and aggregated noise
# ---------------------------
def awgn_noise(
    shape: tuple[int, ...],
    sigma: float,
    rng: RNG | None = None,
) -> Array:
    """
    Generate additive white Gaussian noise (real-valued).

    Parameters
    ----------
    shape : tuple[int, ...]
        Output shape.
    sigma : float
        Standard deviation of the Gaussian noise.
    rng : np.random.Generator | None
        RNG for reproducibility.

    Returns
    -------
    np.ndarray
    """
    if rng is None:
        rng = np.random.default_rng()
    return rng.normal(loc=0.0, scale=sigma, size=shape)


def aggregate_noise_sources(
    source_generators: Sequence[Callable[[tuple[int, ...], RNG | None], Array]],
    shape: tuple[int, ...],
    rng: RNG | None = None,
) -> Array:
    """
    Aggregate many independent noise sources to demonstrate CLT.

    Each generator is a callable: gen(shape, rng) -> ndarray.

    Returns the sum across all sources (same shape).
    """
    if rng is None:
        rng = np.random.default_rng()

    total = np.zeros(shape, dtype=float)
    # Use independent substreams for each source for reproducibility
    for i, gen in enumerate(source_generators):
        subrng = np.random.default_rng(rng.integers(0, 2**32 - 1))
        total += gen(shape, subrng)
    return total


# ---------------------------
# Interference aggregation
# ---------------------------
def simulate_interference_aggregate(
    n_interferers: int,
    samples: int,
    power_per_interferer: float = 1.0,
    fading: bool = True,
    rng: RNG | None = None,
) -> Array:
    """
    Simulate aggregate interference as the sum of many independent interferers.

    Each interferer is modeled as a random amplitude (Rayleigh if fading=True)
    times a random phase (uniform). Returns complex baseband interference samples.
    """
    if rng is None:
        rng = np.random.default_rng()

    phases = rng.uniform(0, 2 * np.pi, size=(n_interferers, samples))
    if fading:
        # Rayleigh amplitude (magnitude)
        amps = rng.rayleigh(scale=np.sqrt(power_per_interferer / 2.0), size=(n_interferers, samples))
    else:
        amps = np.sqrt(power_per_interferer) * np.ones((n_interferers, samples))

    interferers = amps * np.exp(1j * phases)
    aggregate = interferers.sum(axis=0)  # sum across interferers -> shape (samples,)
    return aggregate


# ---------------------------
# Multipath channel models
# ---------------------------
def rayleigh_multipath_channel(
    n_paths: int,
    samples: int,
    avg_power: float = 1.0,
    rng: RNG | None = None,
) -> Array:
    """
    Simulate a complex baseband Rayleigh multipath channel impulse response
    for each sample (n_paths taps per sample). Returns shape (samples, n_paths)
    complex gains. Each tap is complex Gaussian with variance avg_power/n_paths.
    """
    if rng is None:
        rng = np.random.default_rng()

    sigma = np.sqrt(avg_power / (2.0 * n_paths))
    real = rng.normal(0.0, sigma, size=(samples, n_paths))
    imag = rng.normal(0.0, sigma, size=(samples, n_paths))
    return real + 1j * imag


def rician_multipath_channel(
    n_paths: int,
    samples: int,
    k_factor: float = 3.0,
    avg_power: float = 1.0,
    rng: RNG | None = None,
) -> Array:
    """
    Simulate Rician multipath channel taps. k_factor is the ratio of LOS power
    to scattered power. We model one dominant LOS tap plus scattered taps.
    """
    if rng is None:
        rng = np.random.default_rng()

    # LOS power fraction
    los_power = (k_factor / (k_factor + 1.0)) * avg_power
    scat_power = avg_power - los_power
    # LOS tap: deterministic phase random per sample
    los_phase = rng.uniform(0, 2 * np.pi, size=samples)
    los_tap = np.sqrt(los_power) * np.exp(1j * los_phase)  # shape (samples,)

    # scattered taps: Rayleigh with remaining power distributed across n_paths-1
    if n_paths <= 1:
        # Single-tap Rician (LOS only + small scatter)
        scat = np.zeros((samples, 0), dtype=complex)
        taps = los_tap[:, np.newaxis]
    else:
        sigma = np.sqrt(scat_power / (2.0 * (n_paths - 1)))
        real = rng.normal(0.0, sigma, size=(samples, n_paths - 1))
        imag = rng.normal(0.0, sigma, size=(samples, n_paths - 1))
        scat = real + 1j * imag
        taps = np.concatenate((los_tap[:, np.newaxis], scat), axis=1)
    return taps  # shape (samples, n_paths)


def apply_multipath_to_signal(
    tx_signal: Array,
    channel_taps: Array,
    delays: Sequence[int] | None = None,
) -> Array:
    """
    Convolve a real/complex tx_signal (1D) with a time-varying multipath channel.
    channel_taps shape: (samples, n_paths) where samples >= len(tx_signal)
    For simplicity, this function assumes a static channel for the entire signal:
    use channel_taps[0] as the impulse response.

    delays: optional list of integer sample delays per tap (length n_paths).
    """
    taps = np.asarray(channel_taps)
    if taps.ndim != 2:
        raise ValueError("channel_taps must be 2D (samples, n_paths)")
    h = taps[0]  # static approximation
    n_paths = h.size
    if delays is None:
        delays = list(range(n_paths))
    # Build impulse response vector
    max_delay = int(max(delays))
    h_vec = np.zeros(max_delay + 1, dtype=complex)
    for amp, d in zip(h, delays):
        h_vec[int(d)] += amp
    # Convolve
    return np.convolve(tx_signal, h_vec, mode="full")[: tx_signal.size]


# ---------------------------
# OFDM utilities
# ---------------------------
def generate_ofdm_time_signal(
    qam_symbols: Array,
    n_subcarriers: int,
    cp_length: int,
    rng: RNG | None = None,
) -> OFDMResult:
    """
    Convert a 1D array of QAM symbols into a time-domain OFDM signal with cyclic prefix.

    qam_symbols length must be a multiple of n_subcarriers.
    Returns OFDMResult with time_signal (complex), symbols (reshaped), ofdm_symbols (freq->time).
    """
    if rng is None:
        rng = np.random.default_rng()

    symbols = np.asarray(qam_symbols, dtype=complex)
    if symbols.size % n_subcarriers != 0:
        raise ValueError("qam_symbols length must be a multiple of n_subcarriers")

    n_ofdm = symbols.size // n_subcarriers
    symbols_matrix = symbols.reshape(n_ofdm, n_subcarriers)

    ofdm_time = []
    for row in symbols_matrix:
        # IFFT to time domain
        time_sym = np.fft.ifft(row)
        # cyclic prefix
        cp = time_sym[-cp_length:]
        ofdm_with_cp = np.concatenate((cp, time_sym))
        ofdm_time.append(ofdm_with_cp)
    time_signal = np.concatenate(ofdm_time)
    return OFDMResult(time_signal=time_signal, symbols=symbols_matrix, ofdm_symbols=np.array(ofdm_time), cp_length=cp_length)


def compute_papr(time_signal: Array) -> float:
    """
    Compute Peak-to-Average Power Ratio (PAPR) in linear scale for a complex time signal.
    """
    power = np.abs(time_signal) ** 2
    return float(power.max() / (power.mean() + 1e-12))


# ---------------------------
# Antenna array utilities (Uniform Linear Array)
# ---------------------------
def ula_steering_vector(
    n_elements: int,
    angle_rad: float,
    element_spacing_wavelengths: float = 0.5,
) -> Array:
    """
    Compute steering vector for a ULA (n_elements) at angle (radians).
    element_spacing_wavelengths is d / lambda.
    Returns complex vector of length n_elements.
    """
    n = np.arange(n_elements)
    phase_shifts = -2j * np.pi * element_spacing_wavelengths * n * np.sin(angle_rad)
    return np.exp(phase_shifts)


def array_response_power(
    weights: Array,
    angles_rad: Array,
    n_elements: int,
    element_spacing_wavelengths: float = 0.5,
) -> Array:
    """
    Compute array response power for a set of steering angles.
    weights: complex weights (n_elements,)
    angles_rad: 1D array of angles
    Returns power pattern (len(angles_rad),)
    """
    w = np.asarray(weights, dtype=complex)
    angles = np.asarray(angles_rad, dtype=float)
    pattern = []
    for a in angles:
        sv = ula_steering_vector(n_elements, a, element_spacing_wavelengths)
        response = np.vdot(w, sv)  # conjugate dot
        pattern.append(np.abs(response) ** 2)
    return np.array(pattern)


# ---------------------------
# MIMO flat-fading channel
# ---------------------------
def simulate_mimo_flat_fading(
    n_tx: int,
    n_rx: int,
    n_samples: int,
    avg_power: float = 1.0,
    rng: RNG | None = None,
) -> Array:
    """
    Simulate a sequence of MIMO flat-fading channel matrices H of shape
    (n_samples, n_rx, n_tx). Entries are complex Gaussian with variance avg_power/(2).
    """
    if rng is None:
        rng = np.random.default_rng()

    sigma = np.sqrt(avg_power / 2.0)
    real = rng.normal(0.0, sigma, size=(n_samples, n_rx, n_tx))
    imag = rng.normal(0.0, sigma, size=(n_samples, n_rx, n_tx))
    return real + 1j * imag


def mimo_transmit_receive(
    tx_symbols: Array,
    channel_matrices: Array,
    noise_sigma: float,
    rng: RNG | None = None,
) -> ChannelResult:
    """
    Transmit tx_symbols through time-varying MIMO channel_matrices with AWGN.

    tx_symbols shape: (n_samples, n_tx) complex
    channel_matrices shape: (n_samples, n_rx, n_tx)
    Returns ChannelResult with received shape (n_samples, n_rx)
    """
    if rng is None:
        rng = np.random.default_rng()

    tx = np.asarray(tx_symbols, dtype=complex)
    H = np.asarray(channel_matrices, dtype=complex)
    n_samples, n_tx = tx.shape
    _, n_rx, _ = H.shape

    received = np.empty((n_samples, n_rx), dtype=complex)
    noise = np.empty((n_samples, n_rx), dtype=complex)
    for t in range(n_samples):
        y = H[t] @ tx[t]
        n = rng.normal(0.0, noise_sigma / np.sqrt(2.0), size=n_rx) + 1j * rng.normal(0.0, noise_sigma / np.sqrt(2.0), size=n_rx)
        received[t] = y + n
        noise[t] = n
    return ChannelResult(received=received, channel_matrix=H, noise=noise)


# ---------------------------
# Detection and BER (BPSK, QPSK)
# ---------------------------
def bpsk_mod(bits: Array) -> Array:
    """Map bits {0,1} -> BPSK symbols {-1,+1} (real)."""
    b = np.asarray(bits, dtype=int)
    return 2 * b - 1.0


def bpsk_demod(symbols: Array) -> Array:
    """Hard-decision demodulation for BPSK."""
    return (symbols.real > 0).astype(int)


def ber_bpsk_awgn(
    snr_db: float,
) -> float:
    """
    Analytical BER for BPSK in AWGN:
    BER = Q(sqrt(2 * Eb/N0)) where Eb/N0 = SNR (for unit energy symbols)
    """
    snr_linear = 10 ** (snr_db / 10.0)
    return float(qfunc(np.sqrt(2.0 * snr_linear)))


def qpsk_mod(bits: Array) -> Array:
    """
    Map bits to QPSK symbols (Gray mapping).
    bits length must be even. Returns complex symbols.
    """
    b = np.asarray(bits, dtype=int).reshape(-1, 2)
    mapping = {
        (0, 0): 1 + 1j,
        (0, 1): -1 + 1j,
        (1, 1): -1 - 1j,
        (1, 0): 1 - 1j,
    }
    symbols = np.array([mapping[tuple(row)] for row in b], dtype=complex) / np.sqrt(2.0)
    return symbols


def qpsk_demod(symbols: Array) -> Array:
    """
    Hard-decision demodulation for QPSK Gray mapping returning bits.
    """
    s = np.asarray(symbols, dtype=complex)
    bits = []
    for sym in s:
        re = sym.real
        im = sym.imag
        if re >= 0 and im >= 0:
            bits.extend([0, 0])
        elif re < 0 and im >= 0:
            bits.extend([0, 1])
        elif re < 0 and im < 0:
            bits.extend([1, 1])
        else:
            bits.extend([1, 0])
    return np.array(bits, dtype=int)


def ber_qpsk_awgn(snr_db: float) -> float:
    """
    Approximate BER for QPSK in AWGN (same as BPSK per bit):
    BER = Q(sqrt(2 * Eb/N0))
    """
    return ber_bpsk_awgn(snr_db)


# ---------------------------
# Monte Carlo BER simulation harness
# ---------------------------
def monte_carlo_ber(
    modulation: str,
    snr_db: float,
    n_bits: int = 100_000,
    rng: RNG | None = None,
) -> float:
    """
    Monte Carlo estimate of BER for BPSK or QPSK over AWGN.

    Returns estimated BER (float).
    """
    if rng is None:
        rng = np.random.default_rng()

    modulation = modulation.lower()
    if modulation == "bpsk":
        bits = rng.integers(0, 2, size=n_bits)
        symbols = bpsk_mod(bits)
        # AWGN noise
        snr_linear = 10 ** (snr_db / 10.0)
        # For unit-energy BPSK, Eb = 1, so noise sigma^2 = 1/(2*snr)
        sigma = np.sqrt(1.0 / (2.0 * snr_linear))
        noise = rng.normal(0.0, sigma, size=symbols.shape)
        rx = symbols + noise
        bits_hat = bpsk_demod(rx)
        ber = (bits != bits_hat).mean()
        return float(ber)

    elif modulation == "qpsk":
        if n_bits % 2 != 0:
            n_bits += 1
        bits = rng.integers(0, 2, size=n_bits)
        symbols = qpsk_mod(bits)
        snr_linear = 10 ** (snr_db / 10.0)
        # For normalized QPSK symbol energy = 1, Eb = 1/2, so noise variance per dimension = 1/(2*Es/N0)...
        # Simpler: sigma = sqrt(1/(2*snr_linear))
        sigma = np.sqrt(1.0 / (2.0 * snr_linear))
        noise = rng.normal(0.0, sigma, size=symbols.shape) + 1j * rng.normal(0.0, sigma, size=symbols.shape)
        rx = symbols + noise
        bits_hat = qpsk_demod(rx)
        ber = (bits != bits_hat).mean()
        return float(ber)

    else:
        raise ValueError("Unsupported modulation. Use 'bpsk' or 'qpsk'.")


# ---------------------------
# Matched filter detection (simple)
# ---------------------------
def matched_filter_detect(
    rx_signal: Array,
    template: Array,
    noise_sigma: float,
) -> Tuple[float, float]:
    """
    Matched filter detection statistic for a single symbol:
    returns (test_statistic, threshold) where threshold can be set for a desired false alarm.
    For Gaussian noise, threshold for Pfa can be computed; here we return the raw statistic and
    a default threshold of 0 (for symmetric BPSK).
    """
    # correlation
    stat = np.vdot(template, rx_signal).real
    threshold = 0.0
    return float(stat), float(threshold)


# ---------------------------
# Example composite: simulate receiver chain for single-carrier BPSK
# ---------------------------
def simulate_bpsk_receiver_chain(
    bits: Array,
    channel_gain: complex,
    noise_sigma: float,
    rng: RNG | None = None,
) -> Tuple[Array, Array]:
    """
    Transmit BPSK bits through a flat channel (complex gain) with AWGN and return
    received symbols and hard-decoded bits.
    """
    if rng is None:
        rng = np.random.default_rng()

    tx = bpsk_mod(bits)
    # apply complex channel gain
    rx_clean = tx * channel_gain
    noise = rng.normal(0.0, noise_sigma / np.sqrt(2.0), size=tx.shape) + 1j * rng.normal(0.0, noise_sigma / np.sqrt(2.0), size=tx.shape)
    rx = rx_clean + noise
    # Equalize (assuming known channel)
    rx_eq = rx / channel_gain
    bits_hat = bpsk_demod(rx_eq.real)
    return rx_eq, bits_hat

# -----------------------------
# AWGN Channel (complex)
# -----------------------------
def awgn_channel(x: Array, snr_db: float, rng: RNG) -> Array:
    snr_linear = 10 ** (snr_db / 10)
    power = np.mean(np.abs(x) ** 2)
    noise_var = power / snr_linear
    noise = (rng.normal(0, np.sqrt(noise_var / 2), size=x.shape) +
             1j * rng.normal(0, np.sqrt(noise_var / 2), size=x.shape))
    return x + noise


# -----------------------------
# QPSK
# -----------------------------
def qpsk_modulate(bits: Array) -> Array:
    b = bits.reshape(-1, 2)
    return (1 - 2*b[:,0]) + 1j*(1 - 2*b[:,1])


def qpsk_demodulate(symbols: Array) -> Array:
    b0 = (symbols.real < 0).astype(int)
    b1 = (symbols.imag < 0).astype(int)
    return np.column_stack((b0, b1))


# -----------------------------
# 16QAM
# -----------------------------
def qam16_modulate(bits: Array) -> Array:
    b = bits.reshape(-1, 4)
    I = (1 - 2*b[:,0]) * (2 - b[:,2])
    Q = (1 - 2*b[:,1]) * (2 - b[:,3])
    return I + 1j*Q


def qam16_demodulate(symbols: Array) -> Array:
    I = symbols.real
    Q = symbols.imag
    b0 = (I < 0).astype(int)
    b1 = (Q < 0).astype(int)
    b2 = (np.abs(I) < 2).astype(int)
    b3 = (np.abs(Q) < 2).astype(int)
    return np.column_stack((b0, b1, b2, b3))


# -----------------------------
# 64QAM
# -----------------------------
def qam64_modulate(bits: Array) -> Array:
    b = bits.reshape(-1, 6)
    I = (1 - 2*b[:,0]) * (4 - 2*b[:,2] - b[:,4])
    Q = (1 - 2*b[:,1]) * (4 - 2*b[:,3] - b[:,5])
    return I + 1j*Q


def qam64_demodulate(symbols: Array) -> Array:
    I = symbols.real
    Q = symbols.imag
    b0 = (I < 0).astype(int)
    b1 = (Q < 0).astype(int)
    b2 = (np.abs(I) < 4).astype(int)
    b3 = (np.abs(Q) < 4).astype(int)
    b4 = ((np.abs(I) % 4) < 2).astype(int)
    b5 = ((np.abs(Q) % 4) < 2).astype(int)
    return np.column_stack((b0, b1, b2, b3, b4, b5))


# -----------------------------
# Rayleigh fading (flat)
# -----------------------------
def rayleigh_fading(n: int, rng: RNG) -> Array:
    return (rng.normal(0, 1/np.sqrt(2), n) +
            1j * rng.normal(0, 1/np.sqrt(2), n))


# -----------------------------
# OFDM (IFFT/FFT)
# -----------------------------
def ofdm_modulate(symbols: Array, n_subcarriers: int, cp_len: int = 32) -> Array:
    blocks = symbols.reshape(-1, n_subcarriers)
    time_blocks = np.fft.ifft(blocks, axis=1)
    cp = time_blocks[:, -cp_len:]
    return np.hstack((cp, time_blocks)).reshape(-1)


def ofdm_demodulate(signal: Array, n_subcarriers: int, cp_len: int = 32) -> Array:
    block_len = n_subcarriers + cp_len
    n_blocks = len(signal) // block_len
    blocks = signal[:n_blocks*block_len].reshape(n_blocks, block_len)
    blocks = blocks[:, cp_len:]
    freq = np.fft.fft(blocks, axis=1)
    return freq.reshape(-1)


# -----------------------------
# MIMO 2×2 Alamouti STBC
# -----------------------------
def mimo_alamouti_encode(symbols: Array) -> tuple[Array, Array]:
    s0 = symbols[0::2]
    s1 = symbols[1::2]
    tx1 = np.zeros_like(symbols, dtype=complex)
    tx2 = np.zeros_like(symbols, dtype=complex)
    tx1[0::2] = s0
    tx1[1::2] = -np.conjugate(s1)
    tx2[0::2] = s1
    tx2[1::2] = np.conjugate(s0)
    return tx1, tx2


def mimo_alamouti_decode(r1: Array, r2: Array, h1: Array, h2: Array) -> Array:
    s0_hat = np.conjugate(h1)*r1 + h2*r2
    s1_hat = np.conjugate(h2)*r1 - h1*r2
    denom = np.abs(h1)**2 + np.abs(h2)**2
    out = np.zeros(r1.shape, dtype=complex)
    out[0::2] = s0_hat[0::2] / denom[0::2]
    out[1::2] = s1_hat[1::2] / denom[1::2]
    return out


# -----------------------------
# BER
# -----------------------------
def ber(bits_tx: Array, bits_rx: Array) -> float:
    return float(np.mean(bits_tx != bits_rx))


# -----------------------------
# PSD (Welch)
# -----------------------------
def psd_estimate(signal: Array, fs: float = 1.0):
    from scipy.signal import welch
    f, Pxx = welch(signal, fs=fs, nperseg=1024)
    return f, Pxx
