import numpy as np
import matplotlib.pyplot as plt
from fxpmath import Fxp
from iib_project.cd_equalizer import CD_Equalizer
from iib_project.channel import Channel
from iib_project.modulator import Modulator
from iib_project.demodulator import Demodulator
from iib_project.plotting import plot_constellation

def test_fft_width():
    """ Compares SER for different FFT fixed-point widths """
    DW_io = 8
    DW_acc_values = [8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32]
    M_fft = 8
    num_symbols = 2**(M_fft + 1) # 2 blocks

    D = 17
    L = 100 
    symbol_rate = 30 
    wavelength = 1550
    sps = 1
    SNR = 20 

    M = 64 

    
    modulator = Modulator(M)
    channel = Channel(SNR, sps, symbol_rate, D, L, wavelength, DGDSpec=0.1, N_pmd=1)
    demodulator = Demodulator(M)

    x = modulator.modulate(num_symbols)

    ser_results = {}
    bits_results = {}

    for SNR in [15,20,25]:
        channel.SNR = SNR
        x_noisy = channel.add_AWGN(x.reshape(1,-1))
        x_noisy = channel.add_chromatic_dispersion(x_noisy)[0]
        x_noisy = x_noisy / np.max(np.abs(x_noisy))  
        ser_vals = []

        for DW_acc in DW_acc_values:
            cd_equalizer = CD_Equalizer(D, L, symbol_rate, sps, wavelength, M_fft, DW_io, DW_acc)
            bits = bits_per_symbol(cd_equalizer.N_fft, cd_equalizer.N_cd, DW_acc)
            cd_equalizer.reset()
            bits_results[DW_acc] = bits

            x_fxp = [Fxp(val).like(cd_equalizer.io_t) for val in x_noisy]
            y_eq = cd_equalizer.equalize(x_fxp)

            y_eq_np = np.array([val.get_val() for val in y_eq])
            y_decided = demodulator.decide(y_eq_np)
            ser = np.sum(y_decided != x) / len(x)
            ser_vals.append(ser)

        ser_results[SNR] = ser_vals
    
    # plot results showing accumulator width and SNR on bar chart
    plt.figure()
    bar_width = 0.2
    index = np.arange(len(DW_acc_values))
    for i, SNR in enumerate([15,20,25]):
        plt.bar(index + i * bar_width, ser_results[SNR], bar_width, label=f'SNR={SNR}dB')
    plt.xlabel('Accumulator Bit-Width')
    plt.ylabel('Symbol Error Rate (SER)')
    plt.title('SER vs Accumulator Bit-Width')
    plt.xticks(index + bar_width, DW_acc_values)
    plt.yscale('log')
    plt.legend()
    plt.grid(True, which="both", ls="--")
    plt.tight_layout()
    plt.show()

    # now find bit operations per correct symbol
    ser_bits_results = {}
    for SNR in ser_results:
        ser_bits = []
        for i, ser in enumerate(ser_results[SNR]):
            DW_acc = DW_acc_values[i]
            bits = bits_results[DW_acc]
            ser_bits.append(bits / (1 - ser))
        ser_bits_results[SNR] = ser_bits

    # plot bit operations per correct symbol
    plt.figure()
    bar_width = 0.2
    index = np.arange(len(DW_acc_values))
    for i, SNR in enumerate([15,20,25]):
        plt.bar(index + i * bar_width, ser_bits_results[SNR], bar_width, label=f'SNR={SNR}dB')
    plt.xlabel('Accumulator Bit-Width')
    plt.ylabel('Bit Operations per Correct Symbol')
    plt.title('Bit Operations per Correct Symbol vs Accumulator Bit-Width')
    plt.xticks(index + bar_width, DW_acc_values)
    plt.yscale('log')
    plt.legend()
    plt.grid(True, which="both", ls="--")
    plt.tight_layout()
    plt.show()
    


def test_io_width():
    """ Compares SER for different IO fixed-point widths """
    # same implementation as test_fft_width but varying DW_io instead of DW_acc
    DW_acc = 24 
    DW_io_values = [2, 4, 6, 8, 10, 12, 14, 16]
    M_fft = 9 
    num_symbols = 2**(M_fft + 1) # 2 blocks
    D = 17
    L = 100 
    symbol_rate = 30 
    wavelength = 1550
    sps = 1

    M = 64 

    modulator = Modulator(M)
    channel = Channel(20, sps, symbol_rate, D, L, wavelength, DGDSpec=0.1, N_pmd=1)
    demodulator = Demodulator(M)

    x = modulator.modulate(num_symbols)
    ser_results = {}

    for SNR in [15,20,25]:
        channel.SNR = SNR
        x_noisy = channel.add_AWGN(x.reshape(1,-1))
        x_noisy = channel.add_chromatic_dispersion(x_noisy)[0]
        x_noisy = x_noisy / np.max(np.abs(x_noisy))  
        ser_vals = []

        for DW_io in DW_io_values:
            cd_equalizer = CD_Equalizer(D, L, symbol_rate, sps, wavelength, M_fft, DW_io, DW_acc)
            cd_equalizer.reset()

            x_fxp = [Fxp(val).like(cd_equalizer.io_t) for val in x_noisy]
            y_eq = cd_equalizer.equalize(x_fxp)

            y_eq_np = np.array([val.get_val() for val in y_eq])
            y_decided = demodulator.decide(y_eq_np)
            ser = np.sum(y_decided != x) / len(x)
            ser_vals.append(ser)

        ser_results[SNR] = ser_vals

    # plot results, showing IO width and SNR on bar chart
    plt.figure()
    bar_width = 0.2
    index = np.arange(len(DW_io_values))
    for i, SNR in enumerate([15,20,25]):
        plt.bar(index + i * bar_width, ser_results[SNR], bar_width, label=f'SNR={SNR}dB')
    plt.xlabel('ADC resolution')
    plt.ylabel('Symbol Error Rate (SER)')
    plt.title('SER vs ADC resolution')
    plt.xticks(index + bar_width, DW_io_values)
    plt.yscale('log')
    plt.legend()
    plt.grid(True, which="both", ls="--")
    plt.tight_layout()
    plt.show()

    return ser_results

def bits_per_symbol(N_fft, N_cd, DW):
    n_mult = N_fft * (6 * 0.5 * np.log2(N_fft) + 3) / (N_fft - N_cd + 1)
    n_add_re = 5 * n_mult
    n_mult_re = 3 * n_mult
    total_bits = 2.5 * n_add_re * DW + 3 * n_mult_re * DW**2
    return total_bits



if __name__ == "__main__":
    test_io_width()
