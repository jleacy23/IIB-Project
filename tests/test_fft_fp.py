import numpy as np
from fxpmath import Fxp
from iib_project.fft_fp import FFT_fp

M_fft = 6
DW_io = 16
DW_acc = 24

fft_fp = FFT_fp(M_fft, DW_io, DW_acc)


def test_fft():
    """ Tests forward FFT against numpy FFT """
    #generate input data between -1 and 1
    N = 2**M_fft
    x_real = np.random.uniform(-1, 1, N)
    x_imag = np.random.uniform(-1, 1, N)
    x = x_real + 1j * x_imag
    x_fxp = [Fxp(val, dtype=f'fxp-s{DW_io}/{DW_io-1}-complex') for val in x]

    X = np.fft.fft(x)
    X_fxp = fft_fp.fft(x_fxp, inverse=False)

    print("FFT Test:")
    assert len(X) == len(X_fxp), "FFT output length mismatch"
    assert all(np.isclose(X[i], X_fxp[i].get_val(), atol=1e-3) for i in range(N)), "FFT output values mismatch"
    print("FFT test passed!")

def test_fft_ifft():
    """ Tests FFT followed by IFFT returns original signal """
    N = 2**M_fft
    x_real = np.random.uniform(-1, 1, N)
    x_imag = np.random.uniform(-1, 1, N)
    x = x_real + 1j * x_imag
    x_fxp = [Fxp(val, dtype=f'fxp-s{DW_io}/{DW_io-1}-complex') for val in x]

    X_fxp = fft_fp.fft(x_fxp, inverse=False)
    x_reconstructed_fxp = fft_fp.fft(X_fxp, inverse=True)

    print("FFT followed by IFFT Test:")
    assert len(x) == len(x_reconstructed_fxp), "Reconstructed signal length mismatch"
    assert all(np.isclose(x[i], x_reconstructed_fxp[i].get_val(), atol=1e-3) for i in range(N)), "Reconstructed signal values mismatch"
    print("FFT followed by IFFT test passed!")

if __name__ == "__main__":
    test_fft()
    test_fft_ifft()

