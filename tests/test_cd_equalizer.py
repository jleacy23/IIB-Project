import numpy as np
import pytest
from fxpmath import Fxp
from iib_project.cd_equalizer import CD_Equalizer


# ---- Helper mock FFT implementation using NumPy ---- #
class MockFFT:
    def __init__(self, M_fft, DW_io, DW_fft):
        self.N = 2**M_fft

    def fft(self, x, inverse=False):
        arr = np.array([complex(v.get_val()) for v in x])
        if inverse:
            result = np.fft.ifft(arr)
        else:
            result = np.fft.fft(arr)
        return [Fxp(val) for val in result]


@pytest.fixture
def eq(monkeypatch):
    # Patch the FFT_fp used inside CD_Equalizer with the mock FFT
    def mock_fft_class(M_fft, DW_io, DW_fft):
        return MockFFT(M_fft, DW_io, DW_fft)

    monkeypatch.setattr("iib_project.fft_fp.FFT_fp", mock_fft_class)

    # Small sizes for fast testing
    return CD_Equalizer(
        D=17,        # typical SMF dispersion ps/nm/km
        L=10,        # km
        symbol_rate=32, 
        sps=2,
        wavelength=1550,
        M_fft=5,     # FFT size = 32
        DW_io=16,
        DW_fft=16
    )


# ---- Initialization Tests ---- #
def test_initial_values(eq):
    assert eq.N_fft == 32
    assert eq.D > 0
    assert eq.L > 0
    assert eq.nyq_freq > 0

    # Check H_cd shapes
    assert len(eq.H_cd) == eq.N_fft
    assert len(eq.H_cd_fxp) == eq.N_fft


# ---- get_Ncd check ---- #
def test_overlap_length(eq):
    assert 0 < eq.N_cd < eq.N_fft


# ---- Pipeline behavior ---- #
def test_pipeline_updates(eq):
    # Create two test blocks
    x1 = [Fxp(i) for i in range(eq.N_fft)]
    x2 = [Fxp(i + 100) for i in range(eq.N_fft)]

    eq.pipeline(x1)
    assert eq.curr == x1
    assert eq.prev.count(Fxp(0).like(eq.io_t)) == eq.N_fft  # still zeros initially

    eq.pipeline(x2)
    assert eq.curr == x2
    assert eq.prev == x1  # previous block saved correctly

    # Overlap check: first N_cd should match tail of prev
    for i in range(eq.N_cd):
        expected = x1[eq.N_fft - eq.N_cd + i]
        assert eq.overlap[i].get_val() == expected.get_val()

    # Remaining should match start of current block
    for i in range(eq.N_cd, eq.N_fft):
        expected = x2[i - eq.N_cd]
        assert eq.overlap[i].get_val() == expected.get_val()


# ---- Equalization correctness test ---- #
def test_equalize_matches_floating(eq):
    # Random input
    rng = np.random.default_rng(0)
    input_float = rng.normal(0, 1, eq.N_fft) + 1j * rng.normal(0, 1, eq.N_fft)
    x = [Fxp(val).like(eq.io_t) for val in input_float]

    # Fixed-point equalizer output
    y_fxp = eq.equalize(x)

    # Floating reference: pipeline overlap logic
    prev = np.zeros(eq.N_fft, dtype=complex)
    overlap = np.zeros(eq.N_fft, dtype=complex)

    # First run: no prev effect yet
    eq.pipeline(x)  # reuse to get overlap for reference
    prev = np.copy(input_float)
    overlap = np.copy(input_float)

    # Reference FFT equalization
    H = eq.H_cd
    X = np.fft.fft(overlap)
    Y = X * H
    y_ref_full = np.fft.ifft(Y)

    # Remove overlap samples
    y_ref = y_ref_full[eq.N_cd:]

    # Compare fixed-point vs ref (allow quantization tolerance)
    y_fxp_float = np.array([complex(v.get_val()) for v in y_fxp])
    assert np.allclose(y_fxp_float, y_ref, atol=1e-1, rtol=1e-1)
