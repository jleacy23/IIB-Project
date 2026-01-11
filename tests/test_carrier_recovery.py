import numpy as np
from fxpmath import Fxp
from iib_project.carrier_recovery import Carrier_Recovery
from iib_project.modulator import Modulator
from iib_project.channel import Channel
from iib_project.plotting import plot_constellation

def test_viterbit_ML_filter():
    # Test parameters
    sps = 8
    symbol_rate = 32
    total_linewidth = 1e6  # 1 MHz
    snr = 20  # 20 dB
    symbol_energy = 1.0
    N = 5  # Filter half-length

    carrier_recovery = Carrier_Recovery(symbol_rate, sps)
    w = carrier_recovery.viterbi_viterbi_ML(total_linewidth, snr, symbol_energy, N)
    print(w.shape)

def test_viterbi_viterbi():
    #Test parameters
    sps = 8
    symbol_rate = 32
    total_linewidth = 1e6
    snr = 20
    symbol_energy = 1
    N = 5
    DW_acc = 16
    DW_io = 8
    carrier_recovery = Carrier_Recovery(symbol_rate, sps, DW_acc, DW_io)
    num_symbols = 2**10
    M = 4
    
    #Test data, QPSK symbols with AWGN and Wiener phase noise
    modulator = Modulator(M)
    channel = Channel(SNR=snr, sps=sps, symbol_rate=symbol_rate, D = 1, L=1, wavelength=1, DGDSpec=0.1, N_pmd=1, total_linewidth=total_linewidth)

    xv = modulator.modulate(num_symbols)
    xh = modulator.modulate(num_symbols)
    #apply a constant phase shift
    xv = xv * np.exp(1j * np.pi/4)
    xh = xh * np.exp(1j * np.pi/4)
    x = np.vstack((xv, xh))
    x_noisy = channel.add_phase_noise(x)
    x_noisy = channel.add_AWGN(x_noisy)
    scaling = np.percentile(np.abs(x_noisy), 95)
    x_noisy = x_noisy / scaling

    #reference with numpy
    yv_ref = carrier_recovery.viterbi_viterbi_ref(x_noisy[0], N, total_linewidth, snr, symbol_energy/scaling**2)
    yh_ref = carrier_recovery.viterbi_viterbi_ref(x_noisy[1], N, total_linewidth, snr, symbol_energy/scaling**2)

    x_noisy = Fxp(x_noisy).like(carrier_recovery.acc_t)

    y = carrier_recovery.viterbi_viterbi_fxp(x_noisy, N, total_linewidth, snr, symbol_energy/scaling**2)


    #convert y to np for plotting
    yv = np.array(y)[0]
    yh = np.array(y)[1]
    x_noisy = np.array(x_noisy)

    plot_constellation(x_noisy[0], title="Noisy input constellation - Vertical Polarization")
    plot_constellation(yv, title="Viterbi-Viterbi output constellation - Vertical Polarization")
    plot_constellation(yv_ref, title="Viterbi-Viterbi reference output constellation - Vertical Polarization")
    plot_constellation(x_noisy[1], title="Noisy input constellation - Horizontal Polarization")
    plot_constellation(yh, title="Viterbi-Viterbi output constellation - Horizontal Polarization")
    plot_constellation(yh_ref, title="Viterbi-Viterbi reference output constellation - Horizontal Polarization")


if __name__ == "__main__":
    #test_viterbit_ML_filter()
    test_viterbi_viterbi()
    
