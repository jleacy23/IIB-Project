import numpy as np
import matplotlib.pyplot as plt
from fxpmath import Fxp
from iib_project.adaptive_equalizer import Adaptive_Equalizer
from iib_project.modulator import Modulator
from iib_project.channel import Channel
from iib_project.plotting import plot_constellation

def compare_acc_width():
    num_taps = 5
    step_size = 0.01
    DW_io = 16 
    DW_acc_list = [20, 24]
    num_symbols = 5000 
    SNR_list = [20]
    sps = 1
    symbol_rate = 32
    D = 16
    L = 80
    wavelength = 1550
    DGDSpec = 1.8
    N_pmd = 1
    M = 16

    modulator = Modulator(M)
    channel = Channel(SNR=0, sps=sps, symbol_rate=symbol_rate, D=D, L=L, wavelength=wavelength, DGDSpec=DGDSpec, N_pmd=N_pmd)
    xv = modulator.modulate(num_symbols)
    xh = modulator.modulate(num_symbols)
    x = np.vstack((xv, xh))
    x_pmd = channel.add_pmd(x.T).T

    #plot_constellation(xv, title="Transmitted Symbols - Vertical Polarization")
    #plot_constellation(xh, title="Transmitted Symbols - Horizontal Polarization")

    # for each SNR, plot variance in symbol energy vs acc width
    results = {}
    for SNR in SNR_list:
        channel.SNR = SNR
        x_n_pmd = channel.add_AWGN(x_pmd)
        #plot_constellation(x_n[0,:], title=f"Received Symbols (SNR={SNR} dB) - Vertical Polarization")
        #plot_constellation(x_n[1,:], title=f"Received Symbols (SNR={SNR} dB) - Horizontal Polarization")
        plot_constellation(x_n_pmd[0,:], title=f"Received Symbols with PMD (SNR={SNR} dB) - Vertical Polarization")
        plot_constellation(x_n_pmd[1,:], title=f"Received Symbols with PMD (SNR={SNR} dB) - Horizontal Polarization")


        variances = {'y1': [], 'y2': []}
        for DW_acc in DW_acc_list:
            equalizer = Adaptive_Equalizer(num_taps=num_taps, step_size=step_size, DW_io=DW_io, DW_acc=DW_acc)
            y_1, y_2 = equalizer.equalize(x_n_pmd[0], x_n_pmd[1], type='CMA')
            # convert to numpy for variance calculation and plotting
            y_1_np = np.array(y_1.tolist())
            y_2_np = np.array(y_2.tolist())

            plot_constellation(y_1_np, title=f"Equalized Symbols (SNR={SNR} dB, DW_acc={DW_acc}) - Vertical Polarization")
            plot_constellation(y_2_np, title=f"Equalized Symbols (SNR={SNR} dB, DW_acc={DW_acc}) - Horizontal Polarization")

            # separate plots for y1 and y2
            var_y1 = np.var(np.abs(y_1_np)**2)
            var_y2 = np.var(np.abs(y_2_np)**2)
            variances['y1'].append(var_y1)
            variances['y2'].append(var_y2)

        results[SNR] = variances
    # Plot results - separate graphs for y1 and y2
    for pol in ['y1', 'y2']:
        plt.figure()
        for SNR in SNR_list:
            plt.plot(DW_acc_list, results[SNR][pol], marker='o', label=f'SNR={SNR} dB')
        plt.xlabel('Accumulator Bit-Width (DW_acc)')
        plt.ylabel(f'Variance of Symbol Energy ({pol})')
        plt.title(f'Variance of Symbol Energy vs Accumulator Bit-Width ({pol})')
        plt.grid(True)
        plt.legend()
        plt.show()

def compare_io_width():
    num_taps = 15
    step_size = 0.01
    DW_io_list = [8, 10, 12, 14, 16, 18, 20]
    num_symbols = 5000 
    SNR_list = [15, 20, 25]
    sps = 1 
    symbol_rate = 32
    D = 16
    L = 80
    wavelength = 1550
    DGDSpec = 0.1 
    N_pmd = 1

    modulator = Modulator()
    channel = Channel(SNR=0, sps=sps, symbol_rate=symbol_rate, D=D, L=L, wavelength=wavelength, DGDSpec=DGDSpec, N_pmd=N_pmd)
    xv = modulator.qpsk_symbols(num_symbols)
    xh = modulator.qpsk_symbols(num_symbols)
    x = np.vstack((xv, xh))

    results = {}
    for SNR in SNR_list:
        channel.SNR = SNR
        x_n = channel.add_AWGN(x)
        x_n_pmd = channel.add_pmd(x_n.T).T

        variances = {'y1': [], 'y2': []}
        for DW_io in DW_io_list:
            equalizer = Adaptive_Equalizer(num_taps=num_taps, step_size=step_size, DW_io=DW_io, DW_acc= DW_io + 16)
            y_1, y_2 = equalizer.equalize(Fxp(x_n_pmd[0]).like(equalizer.acc_t), Fxp(x_n_pmd[1]).like(equalizer.acc_t), type='CMA')
            y_1_np = np.array(y_1.tolist())
            y_2_np = np.array(y_2.tolist())
            #plot_constellation(y_1_np, title=f"Equalized Symbols (SNR={SNR} dB, DW_io={DW_io}) - Vertical Polarization")
            #plot_constellation(y_2_np, title=f"Equalized Symbols (SNR={SNR} dB, DW_io={DW_io}) - Horizontal Polarization")

            # compare variation around R_d
            R_dv = equalizer.get_Rd(x_n_pmd[0]) 
            R_dh = equalizer.get_Rd(x_n_pmd[1])

            #normalize to same energy
            y_1_np = y_1_np * np.sqrt(R_dv / np.mean(np.abs(y_1_np)**2))
            y_2_np = y_2_np * np.sqrt(R_dh / np.mean(np.abs(y_2_np)**2))

            # find variance
            var_y1 = np.var(np.abs(y_1_np))
            var_y2 = np.var(np.abs(y_2_np))
            variances['y1'].append(var_y1)
            variances['y2'].append(var_y2)


        results[SNR] = variances
    for pol in ['y1', 'y2']:
        #plot bar chart
        plt.figure()
        bar_width = 0.2
        index = np.arange(len(DW_io_list))
        for i, SNR in enumerate(SNR_list):
            plt.bar(index + i * bar_width, results[SNR][pol], bar_width, label=f'SNR={SNR} dB')
        plt.xlabel('Accumulator Bit-Width')
        plt.ylabel(f'Var(|{pol}|)')
        plt.title(f'Variance of Equalized Symbol Amplitude vs Accumulator Bit-Width ({pol})')
        plt.xticks(index + bar_width, DW_io_list)
        plt.legend()
        plt.grid(True)
        plt.show()

        return results


def compare_bits_per_symbol_CMA():
    # compare how filter length affects bits per symbol
    num_taps_list = [5, 11, 15, 21, 25]
    step_size = 0.01
    DW_io = 8
    DW_acc = 16
    bits_per_symbol_filter = []
    for num_taps in num_taps_list:
        equalizer = Adaptive_Equalizer(num_taps=num_taps, step_size=step_size, DW_io=DW_io, DW_acc=DW_acc)
        bps = equalizer.bits_per_symbol_CMA()
        bits_per_symbol_filter.append(bps)
    plt.figure()
    plt.plot(num_taps_list, bits_per_symbol_filter, marker='o')
    plt.xlabel('Number of Taps')
    plt.ylabel('Operations per Symbol')
    plt.title(f'Bit-Level Operations per Decoded Symbol vs Number of Taps (CMA), DW_acc={DW_acc}')
    plt.grid(True)
    plt.show()

    # compare how acc_t size affects bits per symbol
    num_taps = 15
    DW_acc_list = [8, 10, 12, 14, 16, 18, 20]
    bits_per_symbol_word = []
    for DW_acc in DW_acc_list:
        equalizer = Adaptive_Equalizer(num_taps=num_taps, step_size=step_size, DW_io=DW_io, DW_acc=DW_acc)
        bps = equalizer.bits_per_symbol_CMA()
        bits_per_symbol_word.append(bps)
    plt.figure()
    plt.plot(DW_acc_list, bits_per_symbol_word, marker='o')
    plt.xlabel('Accumulator Bit-Width (DW_acc)')
    plt.ylabel('Operations per Symbol')
    plt.title(f'Bit-Level Operations per Decoded Symbol vs Accumulator Bit-Width (CMA), Num Taps={num_taps}')
    plt.grid(True)
    plt.show()
    return bits_per_symbol_filter, bits_per_symbol_word

if __name__ == "__main__":
    compare_bits_per_symbol_CMA()
    compare_io_width()
