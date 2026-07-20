import numpy as np
import pytest

from bayesdawn import datamodel, psdmodel


def psd_function(freq):
    """PSD model used in the imputation tutorial notebook."""
    return 1.0 / (1.0 + 10000.0 / (1.0 + (freq / 2e-2) ** (4 / np.log10(5))))


class TutorialPSD(psdmodel.PSD):
    """Minimal PSD wrapper mirroring the notebook example."""

    def __init__(self, n_data, fs):
        super().__init__(n_data, fs, fmin=None, fmax=None)

    def psd_fn(self, x):
        return psd_function(x)


def build_heterogeneous_mask(n_data):
    """Create non-overlapping gaps with different lengths."""
    mask = np.ones(n_data)
    starts = np.array([200, 520, 900, 1330, 1800, 2400, 3020, 3600])
    lengths = np.array([8, 19, 11, 27, 14, 35, 9, 22])
    for start, length in zip(starts, lengths):
        mask[start:start + length] = 0
    return mask


def build_tutorial_dataset(n_data=4096, fs=1.0, seed=42):
    """Generate the same kind of signal+colored-noise data as the notebook."""
    rng = np.random.default_rng(seed)
    noise = rng.normal(loc=0.0, scale=1.0, size=n_data)

    # Build complex white FFT and color it with the PSD model.
    f = np.fft.rfftfreq(n_data) * fs
    n_fft = noise[0:n_data // 2 + 1].astype(np.complex128)
    n_fft[1:] = n_fft[1:] + 1j * noise[n_data // 2:]
    n_fft = (
        np.sqrt(psd_function(f))
        * n_fft
        * np.sqrt(n_data * fs / 4.0)
    )
    n = np.fft.irfft(n_fft)

    # Deterministic mean component.
    t = np.arange(n_data) / fs
    f0 = 1e-2
    a0 = 5e-3
    s = a0 * np.sin(2 * np.pi * f0 * t)

    return s + n, s


@pytest.fixture(scope='module')
def imputation_setup():
    """Build a reusable setup for heterogeneous multi-gap imputation tests."""
    n_data = 4096
    fs = 1.0
    y, s = build_tutorial_dataset(n_data=n_data, fs=fs, seed=123)
    mask = build_heterogeneous_mask(n_data)
    y_masked = mask * y

    psd_cls = TutorialPSD(n_data, fs)
    imp_cls = datamodel.GaussianStationaryProcess(
        s,
        mask,
        psd_cls,
        method='nearest',
        na=int(50 * fs),
        nb=int(50 * fs),
    )
    imp_cls.compute_offline()

    return {
        'y': y,
        's': s,
        'mask': mask,
        'y_masked': y_masked,
        'imp_cls': imp_cls,
    }


def test_conditional_mean_improves_multi_gap_reconstruction(imputation_setup):
    setup = imputation_setup
    y = setup['y']
    mask = setup['mask']
    y_masked = setup['y_masked']
    imp_cls = setup['imp_cls']

    y_rec = imp_cls.impute(y_masked, draw=False)

    # Observed samples must be unchanged.
    np.testing.assert_allclose(y_rec[mask == 1], y_masked[mask == 1])

    # Imputation should reduce error on missing samples compared to zero-filled masked data.
    mse_masked = np.mean((y_masked[mask == 0] - y[mask == 0]) ** 2)
    mse_rec = np.mean((y_rec[mask == 0] - y[mask == 0]) ** 2)
    assert mse_rec < mse_masked

    # It should also improve the FFT amplitude fit compared to gapped data.
    y_fft = np.abs(np.fft.rfft(y))
    y_masked_fft = np.abs(np.fft.rfft(y_masked))
    y_rec_fft = np.abs(np.fft.rfft(y_rec))
    err_masked = np.linalg.norm(y_masked_fft - y_fft) / np.linalg.norm(y_fft)
    err_rec = np.linalg.norm(y_rec_fft - y_fft) / np.linalg.norm(y_fft)
    assert err_rec < err_masked


def test_stochastic_draw_is_seed_reproducible_and_keeps_observed_data(imputation_setup):
    setup = imputation_setup
    mask = setup['mask']
    y_masked = setup['y_masked']
    imp_cls = setup['imp_cls']

    np.random.seed(321)
    y_draw_1 = imp_cls.impute(y_masked, draw=True)
    np.random.seed(321)
    y_draw_2 = imp_cls.impute(y_masked, draw=True)

    # With a fixed seed, stochastic draw should be exactly reproducible.
    np.testing.assert_allclose(y_draw_1, y_draw_2)

    # Observed samples must remain untouched.
    np.testing.assert_allclose(y_draw_1[mask == 1], y_masked[mask == 1])

    # Drawn missing values should have non-trivial variance.
    assert np.var(y_draw_1[mask == 0]) > 0.0
