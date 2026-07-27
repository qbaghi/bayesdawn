import numpy as np

from bayesdawn.psdmodel import MultivariatePSD


class DummyMultivariatePSD(MultivariatePSD):
    def __init__(self, n_data, fs=1.0):
        super().__init__(n_data, fs, n_chan=2)

    def psd_fn(self, x):
        x = np.asarray(x)
        amp1 = 1.0 + 0.4 / (1.0 + (x / 0.12) ** 2)
        amp2 = 0.8 + 0.3 / (1.0 + (x / 0.18) ** 2)
        csd = 0.15 * np.exp(-x / 0.25) * np.exp(1j * (0.3 + 0.8 * x))

        spec = np.zeros((len(x), 2, 2), dtype=complex)
        spec[:, 0, 0] = amp1
        spec[:, 1, 1] = amp2
        spec[:, 0, 1] = csd
        spec[:, 1, 0] = np.conj(csd)
        return spec


def test_multivariate_psd_calculate_returns_hermitian_symmetric_fft_grid():
    psd = DummyMultivariatePSD(n_data=17, fs=2.0)
    spec = psd.calculate(17)

    assert spec.shape == (17, 2, 2)
    np.testing.assert_allclose(spec[:, 0, 1], np.conj(spec[:, 1, 0]))

    # Negative-frequency part should be conjugate-transposed mirror of positive part.
    np.testing.assert_allclose(spec[1:, 0, 1], np.conj(spec[:0:-1, 1, 0]))


def test_multivariate_psd_calculate_autocorr_matches_ifft_definition():
    n = 12
    fs = 1.5
    psd = DummyMultivariatePSD(n_data=n, fs=fs)

    corr = psd.calculate_autocorr(n)
    spec = psd.calculate(2 * n)
    expected = np.real(np.fft.ifft(spec, axis=0)[0:n]) * fs / 2

    assert corr.shape == (n, 2, 2)
    np.testing.assert_allclose(corr, expected)
    np.testing.assert_allclose(corr[:, 0, 1], corr[:, 1, 0])


def test_multivariate_psd_periodogram_returns_cross_spectral_matrix():
    psd = DummyMultivariatePSD(n_data=9, fs=1.0)
    y_fft = np.array([
        [1.0 + 1.0j, 2.0 - 1.0j],
        [0.5 - 0.2j, -1.2 + 0.7j],
        [2.3 + 0.0j, 0.0 + 0.4j],
    ])

    per = psd.periodogram(y_fft)
    expected = np.einsum("fi,fj->fij", y_fft, np.conj(y_fft)) / len(y_fft) * 2 / psd.fs

    assert per.shape == (3, 2, 2)
    np.testing.assert_allclose(per, expected)
