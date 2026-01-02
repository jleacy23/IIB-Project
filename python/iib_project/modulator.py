import numpy as np
from scipy.signal import upfirdn

class Modulator:
    """ Modulator class responsible for converting bitstream into a simulated optical signal, supports polarization multiplexing"""

    def __init__(self, M):
        self.M = M

    def qpsk_symbols(self, n: int) -> np.ndarray:
        """ Generate QPSK symbols """
        bits = np.random.randint(0, 2, n * 2)
        symbols = (2 * bits[0::2] - 1) + 1j * (2 * bits[1::2] - 1)
        symbols /= np.sqrt(2)  # Normalize power
        return symbols

    def qam16_symbols(self, n: int) -> np.ndarray:
        """ Generate 16-QAM symbols """
        bits = np.random.randint(0, 2, n * 4)
        mapping = {
            (0, 0): -3,
            (0, 1): -1,
            (1, 1): 1,
            (1, 0): 3
        }
        symbols = []
        for i in range(n):
            real_part = mapping[(bits[4*i], bits[4*i + 1])]
            imag_part = mapping[(bits[4*i + 2], bits[4*i + 3])]
            symbols.append(real_part + 1j * imag_part)
        symbols = np.array(symbols)
        symbols /= np.sqrt(10)  # Normalize power
        return symbols

    def qam64_symbols(self, n: int) -> np.ndarray:
        """ Generate 64-QAM symbols """
        bits = np.random.randint(0, 2, n * 6)
        mapping = {
            (0, 0, 0): -7,
            (0, 0, 1): -5,
            (0, 1, 1): -3,
            (0, 1, 0): -1,
            (1, 1, 0): 1,
            (1, 1, 1): 3,
            (1, 0, 1): 5,
            (1, 0, 0): 7
        }
        symbols = []
        for i in range(n):
            real_part = mapping[(bits[6*i], bits[6*i + 1], bits[6*i + 2])]
            imag_part = mapping[(bits[6*i + 3], bits[6*i + 4], bits[6*i + 5])]
            symbols.append(real_part + 1j * imag_part)
        symbols = np.array(symbols)
        symbols /= np.sqrt(42)  # Normalize power
        return symbols

    def modulate(self, num_symbols: int) -> np.ndarray:
        """ Modulate based on M-ary QAM """
        if self.M == 4:
            return self.qpsk_symbols(num_symbols)
        elif self.M == 16:
            return self.qam16_symbols(num_symbols)
        elif self.M == 64:
            return self.qam64_symbols(num_symbols)
        else:
            raise ValueError("Unsupported modulation order. Supported orders are 4, 16, and 64.")

    
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

