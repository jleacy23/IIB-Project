import numpy as np
from scipy.signal import upfirdn

class Modulator:
    """ Modulator class responsible for converting bitstream into a simulated optical signal, supports polarization multiplexing"""
    def __init__(self, M: int):
        self.M = M

    def gen_symbols(self, num_symbols: int) -> np.ndarray:
        """ Converts a randomly generated bitstream into QAM symbols with constellation size 4^M"""
        bits_per_symbol = 2 * self.M
        num_bits = num_symbols * bits_per_symbol
        bitstream = np.random.randint(0, 2, num_bits)

        # Reshape bitstream into symbols
        symbols = bitstream.reshape((num_symbols, bits_per_symbol))

        # Map bits to QAM symbols
        constellation_size = 4 ** self.M
        symbol_indices = np.zeros(num_symbols, dtype=int)

        for i in range(num_symbols):
            for j in range(bits_per_symbol):
                symbol_indices[i] += symbols[i, j] << (bits_per_symbol - j - 1)

        # Generate QAM constellation points
        real_parts = np.array([2 * (i % (2 ** self.M)) - (2 ** self.M - 1) for i in range(constellation_size)])
        imag_parts = np.array([2 * (i // (2 ** self.M)) - (2 ** self.M - 1) for i in range(constellation_size)])
        constellation = real_parts + 1j * imag_parts

        #Normalise
        normalisation = np.sqrt(3 / (2 * (4 ** self.M - 1)))

        qam_symbols = constellation[symbol_indices] * normalisation

        return qam_symbols
    
    def rrc_filter(self, span: int, sps: int, rolloff: float) -> np.ndarray:
        """ Root raised cosine filter"""
        N = span * sps
        t = np.arange(-N / 2, N / 2 + 1) / sps
        rrc = np.zeros_like(t)

        for i in range(len(t)):
            if t[i] == 0.0:
                rrc[i] = 1.0 - rolloff + (4 * rolloff / np.pi)
            elif abs(t[i]) == 1 / (4 * rolloff):
                rrc[i] = (rolloff / np.sqrt(2)) * ((1 + 2 / np.pi) * np.sin(np.pi / (4 * rolloff)) + (1 - 2 / np.pi) * np.cos(np.pi / (4 * rolloff)))
            else:
                rrc[i] = (np.sin(np.pi * t[i] * (1 - rolloff)) + 4 * rolloff * t[i] * np.cos(np.pi * t[i] * (1 + rolloff))) / (np.pi * t[i] * (1 - (4 * rolloff * t[i]) ** 2))

        rrc /= np.sqrt(np.sum(rrc ** 2))  # Normalize filter energy
        return rrc

    def gen_optical_signal(self, qam_symbols: np.ndarray, sps: int, rolloff: float, span: int) -> np.ndarray:
        """ Generates a simulated optical signal in one polarization direction"""
        rrc = self.rrc_filter(span=span, sps=sps, rolloff=rolloff)
        optical_signal = upfirdn(rrc, qam_symbols, up=sps)

        return optical_signal

