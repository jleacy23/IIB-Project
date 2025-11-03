import numpy as np

class Demodulator:
    def qpsk_decide(self, symbols):
        """ Convert received symbols to one of the sent QPSK symbols """
        decided_symbols = []
        for sym in symbols:
            real_part = np.real(sym)
            imag_part = np.imag(sym)
            if real_part >= 0 and imag_part >= 0:
                decided_symbols.append(1 + 1j)
            elif real_part < 0 and imag_part >= 0:
                decided_symbols.append(-1 + 1j)
            elif real_part < 0 and imag_part < 0:
                decided_symbols.append(-1 - 1j)
            else:
                decided_symbols.append(1 - 1j)
        return np.array(decided_symbols) / np.sqrt(2)  # Normalize power to 1k

    def qpsk_ser(self, tx_symbols, rx_symbols):
        tx_symbols_trimmed = tx_symbols[:len(rx_symbols)]
        # check signs of real and imaginary parts
        errors = np.sum((np.sign(np.real(tx_symbols_trimmed)) != np.sign(np.real(rx_symbols))) |
                        (np.sign(np.imag(tx_symbols_trimmed)) != np.sign(np.imag(rx_symbols))))
        ser = errors / len(tx_symbols_trimmed)
        return ser

