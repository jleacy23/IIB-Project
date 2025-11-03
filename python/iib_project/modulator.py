import numpy as np
from scipy.signal import upfirdn

class Modulator:
    """ Modulator class responsible for converting bitstream into a simulated optical signal, supports polarization multiplexing"""

    def qpsk_symbols(self, n: int) -> np.ndarray:
        """ Generate QPSK symbols """
        bits = np.random.randint(0, 2, n * 2)
        symbols = (2 * bits[0::2] - 1) + 1j * (2 * bits[1::2] - 1)
        symbols /= np.sqrt(2)  # Normalize power
        return symbols

    
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

