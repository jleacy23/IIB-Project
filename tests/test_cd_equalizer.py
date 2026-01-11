import numpy as np
from scipy.signal import correlate
from fxpmath import Fxp
from iib_project.cd_equalizer import CD_Equalizer
from iib_project.modulator import Modulator
from iib_project.channel import Channel
from iib_project.demodulator import Demodulator
from iib_project.plotting import plot_constellation

M_eq = 7 
DW_io = 16
DW_acc = 24
symbol_rate = 10  # 10 Gbps
L = 500
D = 16
wavelength = 1550
sps = 1
SNR = 15 

modulator = Modulator(4, 1)
channel = Channel(SNR, sps, symbol_rate, D, L, wavelength)
#cd_equalizer = CD_Equalizer(D, L, symbol_rate, sps, wavelength, M_eq, DW_io, DW_acc)
demodulator = Demodulator(4, 1)

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
    cd_equalizer = CD_Equalizer(D, L, symbol_rate, sps, wavelength, 4, DW_io, DW_acc)
    cd_equalizer.reset()
    print(f'N_cd: {cd_equalizer.N_cd}')
    # test data
    num_symbols = cd_equalizer.N_fft * 2
    x= [i for i in range(num_symbols)]
    # reshape to blocks
    x_blocks = [x[i:i+cd_equalizer.N_fft] for i in range(0, len(x), cd_equalizer.N_fft)]

    # check first pipeline
    cd_equalizer.pipeline(x_blocks[0])
    print(f"Curr: {cd_equalizer.currck}")
    print(f"Prev: {cd_equalizer.prev_block}")
    print(f"Overlap: {cd_equalizer.overlap}")
    # check second pipeline
    print("Second block:")
    cd_equalizer.pipeline(x_blocks[1])
    print(f"Curr: {cd_equalizer.curr_block}")
    print(f"Prev: {cd_equalizer.prev_block}")
    print(f"Overlap: {cd_equalizer.overlap}")

    
        
def test_equalize():
    cd_equalizer = CD_Equalizer(D, L, symbol_rate, sps, wavelength, M_eq, DW_io, DW_acc)
    cd_equalizer.reset()
    # test data
    num_symbols = 2**(M_eq+2) + 3 # awkward length
    x = modulator.qpsk_symbols(num_symbols)
    x_noisy = channel.add_AWGN(x)
    x_noisy = channel.add_chromatic_dispersion(x_noisy)
    x_noisy = x_noisy / np.max(np.abs(x_noisy))  # Normalize to avoid overflow
    x_noisy = x_noisy[0]
    print(x_noisy.shape)
    plot_constellation(x_noisy, title="Received Signal Constellation Before CD Equalization")
    # convert to fixed-point
    x_fxp = [Fxp(val).like(cd_equalizer.io_t) for val in x_noisy]
    y = cd_equalizer.equalize(x_fxp)
    # convert to np for plotting
    y_np = np.array([val.get_val() for val in y])
    plot_constellation(y_np, title="Received Signal Constellation After CD Equalization")
    decided = demodulator.qpsk_decide(y_np)
    ser = np.sum(decided != x) / len(x)
    assert ser < 0.1, f"Equalization failed, SER={ser}"
    print(f"Equalization passed, SER={ser}")

if __name__ == "__main__":
    test_equalize()

