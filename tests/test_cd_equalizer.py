import numpy as np
from fxpmath import Fxp
from iib_project.cd_equalizer import CD_Equalizer
from iib_project.modulator import Modulator
from iib_project.channel import Channel
from iib_project.demodulator import Demodulator
from iib_project.plotting import plot_tx_constellation

M_eq = 9 
DW_io = 16
DW_acc = 24
symbol_rate = 10  # 10 Gbps
L = 500
D = 16
wavelength = 1550
sps = 1
SNR = 15 

modulator = Modulator()
channel = Channel(SNR, sps, symbol_rate, D, L, wavelength)
#cd_equalizer = CD_Equalizer(D, L, symbol_rate, sps, wavelength, M_eq, DW_io, DW_acc)
demodulator = Demodulator()

def test_Hcd():
    cd_equalizer = CD_Equalizer(D, L, symbol_rate, sps, wavelength, M_eq, DW_io, DW_acc)
    assert len(cd_equalizer.H_cd) == cd_equalizer.N_fft, f"H_cd length of {len(cd_equalizer.H_cd)} does not match N_fft of {cd_equalizer.N_fft}"

def test_roll():
    test_array = [Fxp(i) for i in range(10)]
    cd_equalizer = CD_Equalizer(D, L, symbol_rate, sps, wavelength, M_eq, DW_io, DW_acc)
    shift = 4
    assert cd_equalizer.roll(cd_equalizer.roll(x_fxp, shift), -shift) == x_fxp, "Roll function failed to return original array after rolling back"
    print("roll test passed")

def test_pipeline():
    #gemerate test data between -1 and 1
    cd_equalizer = CD_Equalizer(D, L, symbol_rate, sps, wavelength, M_eq, DW_io, DW_acc)
    cd_equalizer.reset()
    cd_equalizer.pipeline(x_fxp)
    assert cd_equalizer.curr == x_fxp, "Current block not updated correctly in pipeline"
    assert cd_equalizer.prev == [Fxp(0).like(cd_equalizer.io_t) for _ in range(cd_equalizer.N_fft)], "Previous block not updated correctly in pipeline"
    assert len(cd_equalizer.overlap) == cd_equalizer.N_fft, "Overlap block size incorrect in pipeline"
    expected_overlap = cd_equalizer.prev[:cd_equalizer.N_cd] + cd_equalizer.curr[cd_equalizer.N_cd:]
    assert cd_equalizer.overlap == expected_overlap, "Overlap block not computed correctly in pipeline"
    print("pipeline test passed")


def test_equalize():
    cd_equalizer = CD_Equalizer(D, L, symbol_rate, sps, wavelength, M_eq, DW_io, DW_acc)
    cd_equalizer.reset()
    # test data
    num_symbols = 2**M_eq
    x = modulator.qpsk_symbols(num_symbols)
    x_noisy = channel.add_AWGN(x)
    x_noisy = channel.add_chromatic_dispersion(x_noisy)
    x_noisy = x_noisy / np.max(np.abs(x_noisy))  # Normalize to avoid overflow
    plot_tx_constellation(x_noisy, title="Received Signal Constellation Before CD Equalization")
    # convert to fixed-point
    x_fxp = [Fxp(val).like(cd_equalizer.io_t) for val in x_noisy]
    y = cd_equalizer.equalize(x_fxp)
    # convert to np for plotting
    y_np = np.array([val.get_val() for val in y])
    plot_tx_constellation(y_np, title="Received Signal Constellation After CD Equalization")
    decided = demodulator.qpsk_decide(y_np)
    ser = np.sum(decided != x) / len(x)
    print(f"Symbol Error Rate after CD Equalization: {ser}")

if __name__ == "__main__":
    test_equalize()



