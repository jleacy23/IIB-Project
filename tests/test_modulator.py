import numpy as np
from iib_project.modulator import Modulator
from iib_project.demodulator import Demodulator
from iib_project.plotting import plot_constellation

def test_mod_demod():
    num_symbols = 1000
    for M in [4, 16, 64]:
        modulator = Modulator(M,1)
        demodulator = Demodulator(M,1)

        symbols = modulator.modulate(num_symbols)
        print(symbols.shape)

        decided = demodulator.decide(symbols)

        ser = np.sum(decided != symbols) / num_symbols
        print(ser)

if __name__ == '__main__':
    test_mod_demod()
