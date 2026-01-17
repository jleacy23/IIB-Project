import numpy as np
from fxpmath import Fxp
from iib_project.carrier_recovery import Carrier_Recovery
from iib_project.modulator import Modulator
from iib_project.channel import Channel
from iib_project.demodulator import Demodulator
from iib_project.plotting import plot_constellation

def test_viterbit_ML_filter():
    # Test parameters
    sps = 8
    symbol_rate = 32
    total_linewidth = 0  # 1 kHz
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
    total_linewidth = 1e5
    snr = 20
    symbol_energy = 1
    N = 5
    DW_acc = 16
    DW_io = 8
    num_symbols = 2**10
    M = 4
    pilot_interval = 20
    
    #Test data, QPSK symbols with AWGN and Wiener phase noise
    modulator = Modulator(M,2)
    channel = Channel(SNR=snr, sps=sps, symbol_rate=symbol_rate, D = 1, L=1, wavelength=1, DGDSpec=0.1, N_pmd=1, total_linewidth=total_linewidth)
    demodulator = Demodulator(M,2)
    carrier_recovery = Carrier_Recovery(symbol_rate, sps, DW_acc, pilot_interval)

    x = modulator.modulate(num_symbols)

    #store pilots
    pilots = np.angle(x[:, ::pilot_interval])

    #add noise
    theta = 0
    x_noisy = np.exp(1j*theta) * x
    x_noisy = channel.add_phase_noise(x_noisy)
    x_noisy = channel.add_AWGN(x_noisy)
    scaling = np.percentile(np.abs(x_noisy), 95)
    x_noisy = x_noisy / scaling

    #apply Viterbi-Viterbi carrier recovery
    x_fxp = Fxp(x_noisy).like(carrier_recovery.acc_t)
    y = carrier_recovery.viterbi_viterbi_fxp(x_fxp, N, total_linewidth, snr, symbol_energy/(scaling**2), 16, pilots)
    y = np.array(y).astype(np.complex64)


    plot_constellation(x_noisy[0], title="Noisy input constellation - Vertical Polarization")
    plot_constellation(y[0], title="Viterbi-Viterbi output constellation - Vertical Polarization")
    plot_constellation(x_noisy[1], title="Noisy input constellation - Horizontal Polarization")
    plot_constellation(y[1], title="Viterbi-Viterbi output constellation - Horizontal Polarization")

    # compute SER
    rx = demodulator.decide(y)
    ser_v = np.sum(rx[0] != x[0]) / rx[0].size
    ser_h = np.sum(rx[1] != x[1]) / rx[1].size

    print(f"Viterbi-Viterbi SER Vertical: {ser_v}")
    print(f"Viterbi-Viterbi SER Horizontal: {ser_h}")

def test_viterbi_viterbi_ref():
    sps = 8
    symbol_rate = 32
    total_linewidth = 1e8
    snr = 20
    symbol_energy = 1
    N = 5
    DW_acc = 16
    DW_io = 8
    num_symbols = 2**10
    M = 4
    pilot_interval = 20
    
    #Test data, QPSK symbols with AWGN and Wiener phase noise
    modulator = Modulator(M,1)
    channel = Channel(SNR=snr, sps=sps, symbol_rate=symbol_rate, D = 1, L=1, wavelength=1, DGDSpec=0.1, N_pmd=1, total_linewidth=total_linewidth)
    demodulator = Demodulator(M,1)
    carrier_recovery = Carrier_Recovery(symbol_rate, sps, DW_acc, DW_io, pilot_interval)

    x = modulator.modulate(num_symbols)

    #store pilots
    pilots = np.angle(x[:, ::pilot_interval])[0]

    #add noise
    theta = 0
    x_noisy = np.exp(1j*theta) * x
    x_noisy = channel.add_phase_noise(x_noisy)
    x_noisy = channel.add_AWGN(x_noisy)
    scaling = np.percentile(np.abs(x_noisy), 95)
    x_noisy = x_noisy / scaling
    x_noisy = x_noisy[0]

    y = carrier_recovery.viterbi_viterbi_ref(x_noisy, N, total_linewidth, snr, symbol_energy/(scaling**2), pilots)

    plot_constellation(x_noisy, title="Noisy input constellation")
    plot_constellation(y, title="Viterbi-Viterbi Reference output constellation")

    # compute SER
    rx = demodulator.decide(y[np.newaxis, :])
    ser = np.sum(rx[0] != x[0]) / rx[0].size

    #compute SER for pi/2 rotations
    for rotation in [1, 2, 3]:
        x_rotated = x * np.exp(1j * rotation * (np.pi/2))
        ser_rotated = np.sum(rx[0] != x_rotated) / rx[0].size
        if ser_rotated < ser:
            ser = ser_rotated

    print(f"Viterbi-Viterbi Reference SER: {ser}")



if __name__ == "__main__":
    #test_viterbit_ML_filter()
    test_viterbi_viterbi()
    #test_viterbi_viterbi_ref()
    
