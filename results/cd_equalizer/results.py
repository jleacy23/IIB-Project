import numpy as np
import matplotlib.pyplot as plt
from fxpmath import Fxp
from iib_project.cd_equalizer import CD_Equalizer
from iib_project.channel import Channel
from iib_project.modulator import Modulator
from iib_project.demodulator import Demodulator

def test_fft_width():
    """ Compares SER for different FFT fixed-point widths """
    DW_io = 8
    DW_acc_values = [8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32]
    M_fft = 7
    num_symbols = 2**(M_fft + 1) # 2 blocks

    D = 17
    L = 100 
    symbol_rate = 30 
    wavelength = 1550
    sps = 1
    SNR = 20 

    
    modulator = Modulator()
    channel = Channel(SNR, sps, symbol_rate, D, L, wavelength)
    demodulator = Demodulator()

    x = modulator.qpsk_symbols(num_symbols)
    x_noisy = channel.add_AWGN(x)
    x_noisy = channel.add_chromatic_dispersion(x_noisy)
    x_noisy = x_noisy / np.max(np.abs(x_noisy))  # Normalize to avoid overflow

    ser_results = {}

    for DW_acc in DW_acc_values:
        cd_equalizer = CD_Equalizer(D, L, symbol_rate, sps, wavelength, M_fft, DW_io, DW_acc)
        cd_equalizer.reset()

        x_fxp = [Fxp(val).like(cd_equalizer.io_t) for val in x_noisy]
        y_eq = cd_equalizer.equalize(x_fxp)

        y_eq_np = np.array([val.get_val() for val in y_eq])
        y_decided = demodulator.qpsk_decide(y_eq_np)
        ser = np.sum(y_decided != x) / len(x)
        ser_results[DW_acc] = ser

    # plot results, showing IO width and SNR
    plt.figure()
    plt.semilogy(list(ser_results.keys()), list(ser_results.values()), marker='o')
    plt.xlabel('FFT Accumulator Bit-Width')
    plt.ylabel('Symbol Error Rate (SER)')
    plt.title(f'SER vs FFT Accumulator Bit-Width (IO Width={DW_io}, SNR={SNR}dB, N_fft = {cd_equalizer.N_fft}, N_cd={cd_equalizer.N_cd})')
    plt.grid(True)
    plt.xticks(DW_acc_values)
    plt.ylim(1e-3, 1)
    plt.show()

def test_io_width():
    """ Compares SER for different IO fixed-point widths """
    DW_acc = 24 
    DW_io_values = [2, 3, 4, 5, 6, 7, 8, 9, 10]
    M_fft = 7
    num_symbols = 2**(M_fft + 1) # 2 blocks

    D = 17
    L = 100 
    symbol_rate = 30 
    wavelength = 1550
    sps = 1
    SNR = 20 

    
    modulator = Modulator()
    channel = Channel(SNR, sps, symbol_rate, D, L, wavelength)
    demodulator = Demodulator()

    x = modulator.qpsk_symbols(num_symbols)
    x_noisy = channel.add_AWGN(x)
    x_noisy = channel.add_chromatic_dispersion(x_noisy)
    x_noisy = x_noisy / np.max(np.abs(x_noisy))  # Normalize to avoid overflow

    ser_results = {}

    for DW_io in DW_io_values:
        cd_equalizer = CD_Equalizer(D, L, symbol_rate, sps, wavelength, M_fft, DW_io, DW_acc)
        cd_equalizer.reset()

        x_fxp = [Fxp(val).like(cd_equalizer.io_t) for val in x_noisy]
        y_eq = cd_equalizer.equalize(x_fxp)

        y_eq_np = np.array([val.get_val() for val in y_eq])
        y_decided = demodulator.qpsk_decide(y_eq_np)
        ser = np.sum(y_decided != x) / len(x)
        ser_results[DW_io] = ser

    # plot results, showing Acc width and SNR
    plt.figure()
    plt.semilogy(list(ser_results.keys()), list(ser_results.values()), marker='o')
    plt.xlabel('FFT IO Bit-Width')
    plt.ylabel('Symbol Error Rate (SER)')
    plt.title(f'SER vs FFT IO Bit-Width (Acc Width={DW_acc}, SNR={SNR}dB), , N_fft = {cd_equalizer.N_fft}, N_cd={cd_equalizer.N_cd}')
    plt.grid(True)
    plt.xticks(DW_io_values)
    plt.ylim(1e-3, 1)
    plt.show()


if __name__ == "__main__":
    test_fft_width()
    #test_io_width()
