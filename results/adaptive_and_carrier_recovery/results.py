import numpy as np
import matplotlib.pyplot as plt
from fxpmath import Fxp

from iib_project.adaptive_equalizer import Adaptive_Equalizer
from iib_project.channel import Channel
from iib_project.modulator import Modulator
from iib_project.plotting import plot_constellation
from iib_project.demodulator import Demodulator
from iib_project.carrier_recovery import Carrier_Recovery


def test_update_step():
    num_taps = 5
    step_size = 0.01
    DW_io = 16 
    DW_acc = 24
    num_symbols = 2048
    SNR = 25
    sps = 1
    symbol_rate = 32
    D = 16
    L = 80
    wavelength = 1550
    DGDSpec = 0.9
    N_pmd = 1
    M = 4

    modulator = Modulator(M, 2)
    channel = Channel(SNR=SNR, sps=sps, symbol_rate=symbol_rate, D=D, L=L, wavelength=wavelength, DGDSpec=DGDSpec, N_pmd=N_pmd)
    demodulator = Demodulator(M, 2)


    x = modulator.modulate(num_symbols)
    # 2 samples per symbol
    x_noisy = np.repeat(x, 2, axis=1)
    x_noisy = channel.add_AWGN(x_noisy)
    x_noisy = channel.add_phase_noise(x_noisy)
    plot_constellation(x_noisy[0,:], title=f"Received Symbols with AWGN and Phase Noise (SNR={SNR} dB) - Vertical Polarization")
    x_noisy = channel.add_pmd(x_noisy.T).T
    plot_constellation(x_noisy[0,:], title=f"Received Symbols with AWGN, Phase Noise and PMD (SNR={SNR} dB) - Vertical Polarization")

    scaling = np.percentile(np.abs(x_noisy), 95)
    x_noisy /= scaling

    # step sizes to test
    mu_steps = [1,2,4,8,16]
    theta_steps = [1,2,4,8,16]

    #SER results
    ser_results = {}

    for mu_step in mu_steps:
        for theta_step in theta_steps:
            adaptive_eq = Adaptive_Equalizer(num_taps=num_taps, step_size=step_size, DW_io=DW_io, DW_acc=DW_acc)
            carrier_recovery = Carrier_Recovery(symbol_rate=symbol_rate, sps=sps, DW_acc=DW_acc, DW_io=DW_io)

            x_noisy_fxp = Fxp(x_noisy).like(adaptive_eq.acc_t)
            eq_output = adaptive_eq.equalize(x_noisy_fxp[0,:], x_noisy_fxp[1,:], type='CMA', step=mu_step)

            #plot equalized constellation
            eq_output_np = np.array(eq_output)
            #plot_constellation(eq_output_np[0,:], title=f"Equalized Symbols (mu_step={mu_step}, theta_step={theta_step}) - Vertical Polarization")

            # Carrier recovery
            cr_output = carrier_recovery.viterbi_viterbi_fxp(x=eq_output_np, N=8, total_linewidth=100e3, snr=SNR, symbol_energy=1/scaling**2, step=theta_step)

            #plot recovered constellation
            cr_output_np = np.array(cr_output)
            plot_constellation(cr_output_np[0,:], title=f"Carrier Recovered Symbols (mu_step={mu_step}, theta_step={theta_step}) - Vertical Polarization")

            # compute SER
            demodulated_symbols = demodulator.decide(cr_output_np)

            num_rx_symbols = demodulated_symbols.shape[1]
            x_ref = x[:, -num_rx_symbols:]

            #shape check
            print(f"demodulated_symbols shape: {demodulated_symbols.shape}, x_shape: {x.shape}, ref shape: {x_ref.shape}")

            #correlation check for offset
            corr = np.correlate(demodulated_symbols[0,:], x_ref[0,:], mode='full')
            #plot
            plt.figure()
            plt.plot(corr)
            plt.title(f"Correlation between Demodulated and Reference Symbols (mu_step={mu_step}, theta_step={theta_step})")
            plt.xlabel("Lag")
            plt.ylabel("Correlation")
            plt.grid()
            plt.show()

            delay = corr.argmax() - (num_rx_symbols - 1)
            print(f"Calculated delay: {delay}")

            ser = np.sum(demodulated_symbols != x_ref) / (num_rx_symbols * 2)
            print(f"mu_step: {mu_step}, theta_step: {theta_step}, SER: {ser:.6f}")
            ser_results[(mu_step, theta_step)] = ser

    # Plot results: bar chart of SER vs mu_step. Different bar colours for different theta_steps
    fig, ax = plt.subplots()
    width = 0.1
    x_ticks = np.arange(len(mu_steps))
    for i, theta_step in enumerate(theta_steps):
        ser_values = [ser_results[(mu_step, theta_step)] for mu_step in mu_steps]
        ax.bar(x_ticks + i*width, ser_values, width=width, label=f'theta_step={theta_step}')
    ax.set_xticks(x_ticks + width*(len(theta_steps)-1)/2)
    ax.set_xticklabels(mu_steps)
    ax.set_xlabel('mu_step')
    ax.set_ylabel('Symbol Error Rate (SER)')
    ax.set_title('SER vs mu_step for different theta_steps')
    ax.legend()
    plt.yscale('log')
    plt.grid(True, which="both", ls="--")
    plt.show()

        
if __name__ == "__main__":
    test_update_step()





