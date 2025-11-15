import numpy as np
import matplotlib.pyplot as plt
from fxpmath import Fxp
from iib_project.adaptive_equalizer import Adaptive_Equalizer
from iib_project.modulator import Modulator
from iib_project.channel import Channel

def test_pmd():
    SNR = 20
    sps = 1
    symbol_rate = 32  # GBd
    D = 16  # ps/(nm*km)
    L = 80  # km
    wavelength = 1550  # nm
    DGDSpec = 0.1  # ps/sqrt(km)
    N_pmd = 10
    N_symbols = 1000

    modulator = Modulator()
    channel = Channel(SNR, sps, symbol_rate, D, L, wavelength, DGDSpec, N_pmd)

    xv = modulator.qpsk_symbols(N_symbols)
    xh = modulator.qpsk_symbols(N_symbols)
    x = np.vstack((xv, xh)).T  # Shape (N_symbols
    x = channel.add_AWGN(x)
    x_channel = channel.add_pmd(x)

    # plot symbols before and after PMD
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.scatter(x[:, 0].real, x[:, 0].imag, color='blue', label='Vertical Polarization')
    plt.scatter(x[:, 1].real, x[:, 1].imag, color='red', label='Horizontal Polarization')
    plt.title('Symbols Before PMD')
    plt.xlabel('In-Phase')
    plt.ylabel('Quadrature')
    plt.axis('equal')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.scatter(x_channel[:, 0].real, x_channel[:, 0].imag, color='blue', label='Vertical Polarization')
    plt.scatter(x_channel[:, 1].real, x_channel[:, 1].imag, color='red', label='Horizontal Polarization')
    plt.title('Symbols After PMD')
    plt.xlabel('In-Phase')
    plt.ylabel('Quadrature')
    plt.axis('equal')
    plt.legend()

    plt.tight_layout()
    plt.show()

def test_adaptive_equalizer():
    SNR = 20
    sps = 1
    symbol_rate = 32  # GBd
    D = 16  # ps/(nm*km)
    L = 80  # km
    wavelength = 1550  # nm
    DGDSpec = 0.1  # ps/sqrt(km)
    N_pmd = 10
    N_symbols = 1000

    modulator = Modulator()
    channel = Channel(SNR, sps, symbol_rate, D, L, wavelength, DGDSpec, N_pmd)
    equalizer = Adaptive_Equalizer(num_taps=11, step_size=0.01)

    xv = modulator.qpsk_symbols(N_symbols)
    xh = modulator.qpsk_symbols(N_symbols)
    x = np.vstack((xv, xh)).T  # Shape (N_symbols, 2)
    x = channel.add_AWGN(x)
    x_channel = channel.add_pmd(x)

    # Apply adaptive equalization

    # plot symbols before and after equalization
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.scatter(x_channel[:, 0].real, x_channel[:, 0].imag, color='blue', label='Vertical Polarization')
    plt.scatter(x_channel[:, 1].real, x_channel[:, 1].imag, color='red', label='Horizontal Polarization')
    plt.title('Symbols Before Equalization')
    plt.xlabel('In-Phase')
    plt.ylabel('Quadrature')
    plt.axis('equal')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.scatter(y_equalized[:, 0].real, y_equalized[:, 0].imag, color='blue', label='Vertical Polarization')
    plt.scatter(y_equalized[:, 1].real, y_equalized[:, 1].imag, color='red', label='Horizontal Polarization')
    plt.title('Symbols After Equalization')
    plt.xlabel('In-Phase')
    plt.ylabel('Quadrature')
    plt.axis('equal')
    plt.legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    test_pmd()
