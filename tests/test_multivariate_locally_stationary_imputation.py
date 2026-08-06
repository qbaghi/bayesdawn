import os
from pathlib import Path

import numpy as np

from bayesdawn.datamodel import MultivariateGaussianLocallyStationaryProcess


def _plot_enabled():
    return os.getenv(
        "BAYESDAWN_PLOT_MULTIVARIATE_LOCAL_IMPUTATION_TESTS", "0"
    ).lower() in {"1", "true", "yes"}


def _plot_dir():
    out_dir = Path(
        os.getenv(
            "BAYESDAWN_MULTIVARIATE_LOCAL_IMPUTATION_PLOT_DIR",
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
    axes[0, 0].set_title("Theoretical local conditional covariance")
    fig.colorbar(im00, ax=axes[0, 0], fraction=0.046, pad=0.04)

    im01 = axes[0, 1].imshow(empirical_cov, aspect="auto", origin="lower")
    axes[0, 1].set_title("Empirical local covariance")
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
        _plot_dir() / "test_multivariate_locally_stationary_imputation_covariance.png",
        dpi=160,
    )
    plt.close(fig)


class DummyMultivariateEvolutionaryPSD:
    """Simple evolutionary PSD/CSD provider for local-stationarity tests."""

    def __init__(self, fs=1.0, n_chan=2):
        self.fs = fs
        self.n_chan = n_chan

    def _corr_params(self, t):
        amp = 1.0 + 0.15 * np.sin(0.2 * t)
        corr = 0.22 + 0.05 * np.cos(0.17 * t)
        d1 = 7.0 + 1.0 * np.sin(0.11 * t)
        d2 = 10.0 + 1.2 * np.cos(0.09 * t)
        d12 = 8.0 + 0.8 * np.sin(0.13 * t)
        return amp, corr, d1, d2, d12

    def calculate_autocorr(self, n, t=0.0):
        lags = np.arange(n)
        amp, corr, d1, d2, d12 = self._corr_params(t)

        r11 = amp * np.exp(-lags / d1)
        r22 = (1.2 * amp) * np.exp(-lags / d2)
        r12 = (corr * amp) * np.exp(-lags / d12)

        corr_tensor = np.zeros((self.n_chan, self.n_chan, n), dtype=float)
        corr_tensor[0, 0, :] = r11
        corr_tensor[1, 1, :] = r22
        corr_tensor[0, 1, :] = r12
        corr_tensor[1, 0, :] = r12
        return corr_tensor

    def calculate(self, n_freq, t=0.0):
        # Interface-compatible placeholder for this class; offline path in these
        # tests relies on calculate_autocorr for lag-domain moments.
        s = np.zeros((n_freq, self.n_chan, self.n_chan), dtype=complex)
        diag = 0.8 + 0.1 * np.sin(0.07 * t)
        s[:, 0, 0] = diag
        s[:, 1, 1] = 1.2 * diag
        s[:, 0, 1] = 0.15 * diag
        s[:, 1, 0] = 0.15 * diag
        return s


def _flatten_channel_major(arr):
    return np.concatenate([arr[:, k] for k in range(arr.shape[1])])


def test_multivariate_local_offline_builds_segment_models_and_cache():
    n = 96
    mask = np.ones(n)
    mask[20:25] = 0
    mask[55:60] = 0

    y_mean = np.zeros((n, 2), dtype=float)
    psd_cls = DummyMultivariateEvolutionaryPSD(fs=1.0, n_chan=2)

    imp = MultivariateGaussianLocallyStationaryProcess(
        y_mean,
        mask,
        psd_cls,
        method="nearest",
        na=10,
        nb=10,
        shared_mask=True,
    )

    imp.compute_offline()

    n_seg = len(imp.segment_meta)
    assert len(imp.crosscorr) == n_seg
    assert len(imp.s2_matrix) == n_seg
    assert len(imp._segment_cov_cache) == n_seg

    for j, seg in enumerate(imp.segment_meta):
        seg_len = len(seg["indices"])
        cov = imp._segment_cov_cache[j]
        assert cov.shape == (seg_len * imp.n_chan, seg_len * imp.n_chan)


def test_multivariate_local_conditional_mean_matches_local_block_solve():
    n = 80
    mask = np.ones(n)
    mask[30:36] = 0

    y_mean = np.zeros((n, 2), dtype=float)
    psd_cls = DummyMultivariateEvolutionaryPSD(fs=1.0, n_chan=2)

    rng = np.random.default_rng(123)
    y = rng.normal(size=(n, 2))
    y_masked = y.copy()
    y_masked[mask == 0, :] = 0.0

    imp = MultivariateGaussianLocallyStationaryProcess(
        y_mean,
        mask,
        psd_cls,
        method="nearest",
        na=12,
        nb=12,
        shared_mask=True,
    )

    y_rec = imp.impute(y_masked, draw=False)

    seg = imp.segment_meta[0]
    yj = (y_masked - y_mean)[seg["indices"], :]
    c_seg = imp._segment_cov_cache[0]

    ind_obs_flat = imp._channel_major_time_indices(seg["ind_obs"], yj.shape[0])
    ind_mis_flat = imp._channel_major_time_indices(seg["ind_mis"], yj.shape[0])

    c_mo = c_seg[np.ix_(ind_mis_flat, ind_obs_flat)]
    c_oo = c_seg[np.ix_(ind_obs_flat, ind_obs_flat)]
    y_obs = _flatten_channel_major(yj[seg["ind_obs"], :])

    mu_vec = c_mo.dot(np.linalg.solve(c_oo, y_obs))
    mu_local = imp._reshape_channel_major_missing(mu_vec, seg["n_mis"])

    np.testing.assert_allclose(y_rec[mask == 1, :], y_masked[mask == 1, :])
    np.testing.assert_allclose(y_rec[mask == 0, :], mu_local, rtol=1e-10, atol=1e-10)


def test_multivariate_local_draw_matches_local_conditional_moments():
    n = 72
    mask = np.ones(n)
    mask[24:28] = 0

    y_mean = np.zeros((n, 2), dtype=float)
    psd_cls = DummyMultivariateEvolutionaryPSD(fs=1.0, n_chan=2)

    rng = np.random.default_rng(99)
    y = rng.normal(size=(n, 2))
    y_masked = y.copy()
    y_masked[mask == 0, :] = 0.0

    imp = MultivariateGaussianLocallyStationaryProcess(
        y_mean,
        mask,
        psd_cls,
        method="nearest",
        na=10,
        nb=10,
        shared_mask=True,
    )
    imp.compute_offline()

    seg = imp.segment_meta[0]
    yj = (y_masked - y_mean)[seg["indices"], :]
    c_seg = imp._segment_cov_cache[0]

    ind_obs_flat = imp._channel_major_time_indices(seg["ind_obs"], yj.shape[0])
    ind_mis_flat = imp._channel_major_time_indices(seg["ind_mis"], yj.shape[0])

    c_mo = c_seg[np.ix_(ind_mis_flat, ind_obs_flat)]
    c_oo = c_seg[np.ix_(ind_obs_flat, ind_obs_flat)]
    c_mm = c_seg[np.ix_(ind_mis_flat, ind_mis_flat)]
    y_obs = _flatten_channel_major(yj[seg["ind_obs"], :])

    mu_theory = c_mo.dot(np.linalg.solve(c_oo, y_obs))
    cov_theory = c_mm - c_mo.dot(np.linalg.solve(c_oo, c_mo.T))
    cov_theory = 0.5 * (cov_theory + cov_theory.T)

    np.random.seed(2026)
    y_draw1 = imp.impute(y_masked, draw=True)
    np.random.seed(2026)
    y_draw2 = imp.impute(y_masked, draw=True)
    np.testing.assert_allclose(y_draw1, y_draw2)

    n_draws = 2500
    draws = np.empty((n_draws, seg["n_mis"] * imp.n_chan), dtype=float)
    for i in range(n_draws):
        np.random.seed(60_000 + i)
        y_draw = imp.impute(y_masked, draw=True)
        draws[i, :] = _flatten_channel_major(y_draw[mask == 0, :])

    empirical_mean = np.mean(draws, axis=0)
    empirical_cov = np.cov(draws, rowvar=False, bias=True)

    mean_rel_err = np.linalg.norm(empirical_mean - mu_theory) / max(
        1.0, np.linalg.norm(mu_theory)
    )
    cov_rel_err = np.linalg.norm(empirical_cov - cov_theory) / np.linalg.norm(cov_theory)

    _save_covariance_diagnostic(cov_theory, empirical_cov, empirical_mean, mu_theory)

    assert mean_rel_err < 0.14
    assert cov_rel_err < 0.20
