import os
from pathlib import Path

import numpy as np

from bayesdawn.datamodel import MultivariateGaussianStationaryProcess


def _plot_enabled():
    return os.getenv("BAYESDAWN_PLOT_MULTIVARIATE_IMPUTATION_TESTS", "0").lower() in {
        "1",
        "true",
        "yes",
    }


def _plot_dir():
    out_dir = Path(
        os.getenv(
            "BAYESDAWN_MULTIVARIATE_IMPUTATION_PLOT_DIR",
            "tests/_artifacts",
        )
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _save_covariance_diagnostic(cov_theory, empirical_cov, empirical_mean, mu_theory):
    if not _plot_enabled():
        return

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    diff_cov = empirical_cov - cov_theory
    diff_mean = empirical_mean - mu_theory

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    im00 = axes[0, 0].imshow(cov_theory, aspect="auto", origin="lower")
    axes[0, 0].set_title("Theoretical conditional covariance")
    fig.colorbar(im00, ax=axes[0, 0], fraction=0.046, pad=0.04)

    im01 = axes[0, 1].imshow(empirical_cov, aspect="auto", origin="lower")
    axes[0, 1].set_title("Empirical covariance from draws")
    fig.colorbar(im01, ax=axes[0, 1], fraction=0.046, pad=0.04)

    im10 = axes[1, 0].imshow(diff_cov, aspect="auto", origin="lower")
    axes[1, 0].set_title("Empirical - theoretical covariance")
    fig.colorbar(im10, ax=axes[1, 0], fraction=0.046, pad=0.04)

    axes[1, 1].plot(mu_theory, label="Theoretical mean", linewidth=2)
    axes[1, 1].plot(empirical_mean, label="Empirical mean", alpha=0.85)
    axes[1, 1].plot(diff_mean, label="Difference", alpha=0.85)
    axes[1, 1].set_title("Conditional mean comparison")
    axes[1, 1].legend()

    fig.tight_layout()
    fig.savefig(
        _plot_dir() / "test_multivariate_imputation_covariance.png",
        dpi=160,
    )
    plt.close(fig)


class DummyMultivariatePSD:
    """Minimal PSD/CSD provider for multivariate imputation tests."""

    def __init__(self, n_data, fs=1.0):
        self.n_data = n_data
        self.fs = fs

    def calculate_autocorr(self, n):
        lags = np.arange(n)
        r11 = np.exp(-lags / 7.0)
        r22 = 1.2 * np.exp(-lags / 10.0)
        r12 = 0.30 * np.exp(-lags / 8.0)

        corr = np.zeros((2, 2, n), dtype=float)
        corr[0, 0, :] = r11
        corr[1, 1, :] = r22
        corr[0, 1, :] = r12
        corr[1, 0, :] = r12
        return corr

    def calculate(self, n_freq):
        # Only needed for interface compatibility in this implementation stage.
        return np.zeros((n_freq, 2, 2), dtype=complex)


def _build_full_block_cov(crosscorr, n_data):
    n_chan = crosscorr.shape[0]
    cov = np.zeros((n_data * n_chan, n_data * n_chan), dtype=float)
    for i in range(n_data):
        for j in range(n_data):
            lag = abs(i - j)
            i0 = i * n_chan
            i1 = i0 + n_chan
            j0 = j * n_chan
            j1 = j0 + n_chan
            cov[i0:i1, j0:j1] = crosscorr[:, :, lag]
    return cov


def _local_conditional_moments(imp, yj, ind_mis, ind_obs):
    seg_len = yj.shape[0]
    c_mo, c_oo = imp._build_local_covariance_blocks(ind_mis, ind_obs, seg_len)
    c_mm = imp._build_local_covariance(ind_mis, ind_mis, seg_len)
    y_obs = imp._flatten_channel_major(yj[ind_obs, :])

    rhs = np.linalg.solve(c_oo, y_obs)
    mu = c_mo.dot(rhs)
    cond_cov = c_mm - c_mo.dot(np.linalg.solve(c_oo, c_mo.T))
    cond_cov = 0.5 * (cond_cov + cond_cov.T)

    return mu, cond_cov


def test_multivariate_nearest_conditional_mean_shared_mask_matches_local_block_solve():
    n = 72
    n_chan = 2
    fs = 1.0
    mask = np.ones(n)
    mask[24:31] = 0

    psd_cls = DummyMultivariatePSD(n_data=n, fs=fs)
    crosscorr = psd_cls.calculate_autocorr(n)
    full_cov = _build_full_block_cov(crosscorr, n)

    rng = np.random.default_rng(123)
    y = rng.multivariate_normal(np.zeros(n * n_chan), full_cov).reshape(n, n_chan)

    y_mean = np.zeros_like(y)
    imp = MultivariateGaussianStationaryProcess(
        y_mean,
        mask,
        psd_cls,
        method="nearest",
        na=12,
        nb=12,
        shared_mask=True,
    )

    y_masked = y.copy()
    y_masked[mask == 0, :] = 0.0
    y_rec = imp.impute(y_masked, draw=False)

    # Observed samples must remain unchanged.
    np.testing.assert_allclose(y_rec[mask == 1, :], y_masked[mask == 1, :])

    # Validate missing reconstruction against direct local block conditional mean.
    seg = imp.segment_meta[0]
    yj = (y_masked - y_mean)[seg["indices"], :]
    seg_len = yj.shape[0]
    c_mo, c_oo = imp._build_local_covariance_blocks(
        seg["ind_mis"], seg["ind_obs"], seg_len
    )
    rhs = np.linalg.solve(c_oo, imp._flatten_channel_major(yj[seg["ind_obs"], :]))
    mu_local = imp._reshape_channel_major_missing(c_mo.dot(rhs), seg["n_mis"])

    np.testing.assert_allclose(y_rec[imp.ind_mis_t, :], mu_local, rtol=1e-10, atol=1e-10)


def test_multivariate_draw_shared_mask_is_reproducible_and_preserves_observed():
    n = 56
    n_chan = 2
    mask = np.ones(n)
    mask[18:24] = 0

    psd_cls = DummyMultivariatePSD(n_data=n)
    crosscorr = psd_cls.calculate_autocorr(n)
    full_cov = _build_full_block_cov(crosscorr, n)

    rng = np.random.default_rng(7)
    y = rng.multivariate_normal(np.zeros(n * n_chan), full_cov).reshape(n, n_chan)
    y_mean = np.zeros_like(y)

    imp = MultivariateGaussianStationaryProcess(
        y_mean,
        mask,
        psd_cls,
        method="nearest",
        na=10,
        nb=10,
        shared_mask=True,
    )

    y_masked = y.copy()
    y_masked[mask == 0, :] = 0.0

    np.random.seed(2024)
    y_draw1 = imp.impute(y_masked, draw=True)
    np.random.seed(2024)
    y_draw2 = imp.impute(y_masked, draw=True)

    np.testing.assert_allclose(y_draw1, y_draw2)
    np.testing.assert_allclose(y_draw1[mask == 1, :], y_masked[mask == 1, :])

    y_mean_rec = imp.impute(y_masked, draw=False)
    # Draw should generally differ from deterministic conditional mean on missing entries.
    assert np.linalg.norm(y_draw1[mask == 0, :] - y_mean_rec[mask == 0, :]) > 0.0


def test_multivariate_draw_shared_mask_matches_theoretical_local_covariance():
    n = 48
    n_chan = 2
    mask = np.ones(n)
    mask[20:23] = 0

    psd_cls = DummyMultivariatePSD(n_data=n)
    crosscorr = psd_cls.calculate_autocorr(n)
    full_cov = _build_full_block_cov(crosscorr, n)

    rng = np.random.default_rng(99)
    y = rng.multivariate_normal(np.zeros(n * n_chan), full_cov).reshape(n, n_chan)
    y_mean = np.zeros_like(y)

    imp = MultivariateGaussianStationaryProcess(
        y_mean,
        mask,
        psd_cls,
        method="nearest",
        na=8,
        nb=8,
        shared_mask=True,
    )

    y_masked = y.copy()
    y_masked[mask == 0, :] = 0.0

    if imp.crosscorr is None or imp.s2_matrix is None:
        imp.compute_offline()

    seg = imp.segment_meta[0]
    yj = (y_masked - y_mean)[seg["indices"], :]
    mu_theory, cov_theory = _local_conditional_moments(
        imp, yj, seg["ind_mis"], seg["ind_obs"]
    )

    n_draws = 4000
    draws = np.empty((n_draws, seg["n_mis"] * n_chan), dtype=float)
    for idx in range(n_draws):
        np.random.seed(50_000 + idx)
        y_draw = imp.impute(y_masked, draw=True)
        draws[idx, :] = imp._flatten_channel_major(y_draw[imp.ind_mis_t, :])

    empirical_mean = np.mean(draws, axis=0)
    empirical_cov = np.cov(draws, rowvar=False, bias=True)

    mean_rel_err = np.linalg.norm(empirical_mean - mu_theory) / max(
        1.0, np.linalg.norm(mu_theory)
    )
    cov_rel_err = np.linalg.norm(empirical_cov - cov_theory) / np.linalg.norm(cov_theory)

    _save_covariance_diagnostic(cov_theory, empirical_cov, empirical_mean, mu_theory)

    assert mean_rel_err < 0.12
    assert cov_rel_err < 0.18
