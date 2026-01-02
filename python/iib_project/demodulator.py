import numpy as np

class Demodulator:
    def __init__(self, M):
        self.M = M  # Modulation order

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

    def qam16_decide(self, symbols):
        #first normalize symbols to unit average power
        rx_energy = np.mean(np.abs(symbols)**2)
        print(rx_energy)
        symbols = symbols / np.sqrt(rx_energy)
        decided_symbols = []
        for sym in symbols:
            real_part = np.real(sym) * np.sqrt(10)
            imag_part = np.imag(sym) * np.sqrt(10)
            # Decide real part
            if real_part < -2:
                real_decided = -3
            elif real_part < 0:
                real_decided = -1
            elif real_part < 2:
                real_decided = 1
            else:
                real_decided = 3
            # Decide imaginary part
            if imag_part < -2:
                imag_decided = -3
            elif imag_part < 0:
                imag_decided = -1
            elif imag_part < 2:
                imag_decided = 1
            else:
                imag_decided = 3
            decided_symbols.append(real_decided + 1j * imag_decided)
        return np.array(decided_symbols) / np.sqrt(10)  # Normalize power

    def qam64_decide(self, symbols):
        #first normalize symbols to unit average power
        rx_energy = np.mean(np.abs(symbols)**2)
        print(rx_energy)
        symbols = symbols / np.sqrt(rx_energy)  # Normalize to unit power
        decided_symbols = []
        for sym in symbols:
            real_part = np.real(sym) * np.sqrt(42)
            imag_part = np.imag(sym) * np.sqrt(42)
            # Decide real part
            if real_part < -6:
                real_decided = -7
            elif real_part < -4:
                real_decided = -5
            elif real_part < -2:
                real_decided = -3
            elif real_part < 0:
                real_decided = -1
            elif real_part < 2:
                real_decided = 1
            elif real_part < 4:
                real_decided = 3
            elif real_part < 6:
                real_decided = 5
            else:
                real_decided = 7
            # Decide imaginary part
            if imag_part < -6:
                imag_decided = -7
            elif imag_part < -4:
                imag_decided = -5
            elif imag_part < -2:
                imag_decided = -3
            elif imag_part < 0:
                imag_decided = -1
            elif imag_part < 2:
                imag_decided = 1
            elif imag_part < 4:
                imag_decided = 3
            elif imag_part < 6:
                imag_decided = 5
            else:
                imag_decided = 7
            decided_symbols.append(real_decided + 1j * imag_decided)
        return np.array(decided_symbols) / np.sqrt(42)  # Normalize power

    def decide(self, symbols):
        if self.M == 4:
            return self.qpsk_decide(symbols)
        elif self.M == 16:
            return self.qam16_decide(symbols)
        elif self.M == 64:
            return self.qam64_decide(symbols)
        else:
            raise ValueError("Unsupported modulation order")

