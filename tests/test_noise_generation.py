import os
from pathlib import Path

import numpy as np

from bayesdawn.noisegenerator import generate_freq_noise_from_psd, generate_noise_from_psd


def _plot_enabled():
    return os.getenv("BAYESDAWN_PLOT_NOISE_TESTS", "0").lower() in {
        "1",
        "true",
        "yes",
    }


def _plot_dir():
    out_dir = Path(os.getenv("BAYESDAWN_NOISE_TEST_PLOT_DIR", "tests/_artifacts"))
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _save_univariate_plot(freq_pos, psd_pos, empirical_psd):
    if not _plot_enabled():
        return

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(freq_pos, psd_pos, label="Target PSD", linewidth=2)
    ax.plot(freq_pos, empirical_psd, label="Averaged empirical PSD", alpha=0.85)
    ax.set_xlabel("Frequency")
    ax.set_ylabel("PSD")
    ax.set_title("Univariate frequency-noise PSD check")
    ax.legend()
    fig.tight_layout()
    fig.savefig(_plot_dir() / "test_noise_generation_univariate.png", dpi=160)
    plt.close(fig)


def _save_multivariate_plot(freq_pos, psd_pos, empirical_spec):
    if not _plot_enabled():
        return

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 7), sharex=True)

    ax1.plot(freq_pos, np.real(psd_pos[:, 0, 0]), label="Target S11", linewidth=2)
    ax1.plot(freq_pos, np.real(empirical_spec[:, 0, 0]), label="Averaged empirical S11", alpha=0.85)
    ax1.plot(freq_pos, np.real(psd_pos[:, 1, 1]), label="Target S22", linewidth=2)
    ax1.plot(freq_pos, np.real(empirical_spec[:, 1, 1]), label="Averaged empirical S22", alpha=0.85)
    ax1.set_ylabel("Auto-PSD")
    ax1.legend(ncol=2)
    ax1.set_title("Multivariate frequency-noise spectral check")

    ax2.plot(freq_pos, np.abs(psd_pos[:, 0, 1]), label="|S12|", linewidth=2)
    ax2.plot(
        freq_pos,
        np.abs(empirical_spec[:, 0, 1]),
        label="Empirical |S12|",
        alpha=0.85,
    )
    ax2.plot(freq_pos, np.imag(psd_pos[:, 0, 2]), label="Target Im(S12)", linewidth=2)
    ax2.plot(
        freq_pos,
        np.imag(empirical_spec[:, 0, 2]),
        label="Empirical Im(S12)",
        alpha=0.85,
    )
    ax2.set_xlabel("Frequency")
    ax2.set_ylabel("CSD")
    ax2.legend(ncol=2)

    fig.tight_layout()
    fig.savefig(_plot_dir() / "test_noise_generation_multivariate.png", dpi=160)
    plt.close(fig)


def _save_multivariate_time_domain_plot(freq_pos, psd_pos, empirical_spec):
    if not _plot_enabled():
        return

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 7), sharex=True)

    ax1.plot(
        freq_pos,
        np.real(psd_pos[:, 0, 0]),
        label="Target S11",
        linewidth=2,
    )
    ax1.plot(
        freq_pos,
        np.real(empirical_spec[:, 0, 0]),
        label="Empirical S11 from time samples",
        alpha=0.85,
    )
    ax1.plot(
        freq_pos,
        np.real(psd_pos[:, 1, 1]),
        label="Target S22",
        linewidth=2,
    )
    ax1.plot(
        freq_pos,
        np.real(empirical_spec[:, 1, 1]),
        label="Empirical S22 from time samples",
        alpha=0.85,
    )
    ax1.set_ylabel("Auto-PSD")
    ax1.legend(ncol=2)
    ax1.set_title("Multivariate time-noise spectral check")

    ax2.plot(freq_pos, np.abs(psd_pos[:, 0, 1]), label="Target |S12|", linewidth=2)
    ax2.plot(
        freq_pos,
        np.abs(empirical_spec[:, 0, 1]),
        label="Empirical |S12| from time samples",
        alpha=0.85,
    )
    ax2.plot(
        freq_pos,
        np.imag(psd_pos[:, 0, 2]),
        label="Target Im(S13)",
        linewidth=2,
    )
    ax2.plot(
        freq_pos,
        np.imag(empirical_spec[:, 0, 2]),
        label="Empirical Im(S13) from time samples",
        alpha=0.85,
    )
    ax2.set_xlabel("Frequency")
    ax2.set_ylabel("CSD")
    ax2.legend(ncol=2)

    fig.tight_layout()
    fig.savefig(_plot_dir() / "test_noise_generation_multivariate_time_domain.png", dpi=160)
    plt.close(fig)


def arbitrary_univariate_psd(freq):
    """Positive, non-flat PSD used for validation."""
    x = np.abs(freq)
    return 0.2 + 0.7 / (1.0 + (x / 0.12) ** 1.8) + 0.05 * np.cos(3.0 * np.pi * x) ** 2


def arbitrary_multivariate_psd(freq):
    """Build a positive-definite spectrum matrix at each frequency."""
    x = np.abs(freq)
    n_freq = len(x)
    p = 3
    psd = np.zeros((n_freq, p, p), dtype=complex)

    d1 = 1.2 + 0.5 / (1.0 + (x / 0.08) ** 1.5)
    d2 = 1.0 + 0.4 / (1.0 + (x / 0.10) ** 1.2)
    d3 = 1.4 + 0.3 / (1.0 + (x / 0.06) ** 1.8)

    c12 = 0.12 * np.exp(-x / 0.25) * np.exp(1j * (0.15 + 0.9 * x))
    c13 = 0.08 * np.exp(-x / 0.20) * np.exp(1j * (-0.30 + 0.7 * x))
    c23 = 0.10 * np.exp(-x / 0.30) * np.exp(1j * (0.45 - 0.6 * x))

    psd[:, 0, 0] = d1
    psd[:, 1, 1] = d2
    psd[:, 2, 2] = d3
    psd[:, 0, 1] = c12
    psd[:, 1, 0] = np.conj(c12)
    psd[:, 0, 2] = c13
    psd[:, 2, 0] = np.conj(c13)
    psd[:, 1, 2] = c23
    psd[:, 2, 1] = np.conj(c23)

    return psd


def test_generate_freq_noise_from_univariate_psd_matches_target_power():
    fs = 2.0
    n_psd = 129  # odd size avoids Nyquist-specific branch
    freq = np.fft.fftfreq(n_psd, d=1.0 / fs)
    psd = arbitrary_univariate_psd(freq)

    # Exact reproducibility for a fixed seed.
    a = generate_freq_noise_from_psd(psd, fs, myseed=7)
    b = generate_freq_noise_from_psd(psd, fs, myseed=7)
    np.testing.assert_allclose(a, b)

    # Monte Carlo check at all interior positive frequency bins.
    inds = np.where((freq > 0) & (freq < fs / 2.0))[0]
    n_draws = 2500
    # Outputs a n_draws x n_freq array of samples at the selected frequency bins.
    samples = np.array(
        [generate_freq_noise_from_psd(psd, fs, myseed=i) for i in range(n_draws)]
    )

    # Average periodogram power at each frequency bin should match the target PSD.
    empirical_power = np.mean(np.abs(samples[:, inds]) ** 2, axis=0)
    scale = n_psd * fs / 2.0
    expected_power = scale * psd[inds]
    # Compute the ratio and average across frequency bins.
    ratio = np.mean(empirical_power / expected_power)
    np.testing.assert_allclose(ratio, 1.0, atol=3/np.sqrt(n_draws))

    if _plot_enabled():
        _save_univariate_plot(freq[inds], psd[inds], empirical_power / scale)


def test_generate_freq_noise_from_multivariate_psd_matches_target_covariance():
    fs = 1.5
    n_psd = 97  # odd size avoids Nyquist-specific branch
    freq = np.fft.fftfreq(n_psd, d=1.0 / fs)
    psd = arbitrary_multivariate_psd(freq)

    # Exact reproducibility for a fixed seed.
    a = generate_freq_noise_from_psd(psd, fs, myseed=19)
    b = generate_freq_noise_from_psd(psd, fs, myseed=19)
    np.testing.assert_allclose(a, b)

    inds = np.where((freq > 0) & (freq < fs / 2.0))[0]
    n_draws = 4000
    # Outputs a n_draws x n_freq x p array of samples at frequency bin k, where p = psd.shape[1].
    samples = np.array(
        [generate_freq_noise_from_psd(psd, fs, myseed=i) for i in range(n_draws)]
    )

    # For each frequency bin, computes E[z z^H] which should match scale * PSD
    # (output has size n_draws x n_freq x p x p).
    empirical_cov = np.asarray([samples[:, inds[i], :].T.conj() @ samples[:, inds[i], :] / n_draws
                                for i in range(len(inds))])
    scale = n_psd * fs / 2.0
    expected_cov = scale * psd[inds]

    # The product of the inverse of the expected covariance with the empirical covariance should be 
    # close to identity.
    ratio = np.mean(np.einsum("...jk, ...kl -> ...jl", np.linalg.inv(expected_cov), empirical_cov),
                    axis=0)
    np.testing.assert_allclose(ratio, np.eye(psd.shape[1]), atol=3/np.sqrt(n_draws))


    if _plot_enabled():
        scale = n_psd * fs / 2.0
        empirical_spec = empirical_cov / scale
        _save_multivariate_plot(freq[inds], psd[inds], empirical_spec)


def test_generate_time_noise_from_multivariate_psd_matches_target_covariance():
    fs = 1.5
    n_psd = 97  # odd size avoids Nyquist-specific branch
    freq = np.fft.fftfreq(n_psd, d=1.0 / fs)
    psd = arbitrary_multivariate_psd(freq)

    # Exact reproducibility for a fixed seed.
    a = generate_noise_from_psd(psd, fs, myseed=31)
    b = generate_noise_from_psd(psd, fs, myseed=31)
    np.testing.assert_allclose(a, b)

    # Generate time-domain samples then map back to frequency domain to
    # estimate the implied spectral covariance.
    inds = np.where((freq > 0) & (freq < fs / 2.0))[0]
    n_draws = 3000
    time_samples = np.array(
        [generate_noise_from_psd(psd, fs, myseed=i) for i in range(n_draws)]
    )
    freq_samples = np.fft.fft(time_samples, axis=1)

    empirical_cov = np.asarray(
        [
            freq_samples[:, inds[i], :].T.conj() @ freq_samples[:, inds[i], :] / n_draws
            for i in range(len(inds))
        ]
    )
    scale = n_psd * fs / 2.0
    expected_cov = scale * psd[inds]

    ratio = np.mean(
        np.einsum("...jk, ...kl -> ...jl", np.linalg.inv(expected_cov), empirical_cov),
        axis=0,
    )
    np.testing.assert_allclose(ratio, np.eye(psd.shape[1]), atol=3 / np.sqrt(n_draws))

    if _plot_enabled():
        empirical_spec = empirical_cov / scale
        _save_multivariate_time_domain_plot(freq[inds], psd[inds], empirical_spec)
