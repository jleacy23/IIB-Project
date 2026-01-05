import numpy as np
from iib_project.carrier_recovery import Carrier_Recovery
from iib_project.modulator import Modulator
from iib_project.channel import Channel

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
    channel = Channel(SNR=snr, sps=sps, symbol_rate=symbol_rate, D = 1, L=1, wavelength=1, DGDSpec=0.1, N_pmd=1)

    x = modulator.modulate(num_symbols)


    

if __name__ == "__main__":
    #test_viterbit_ML_filter()
    
