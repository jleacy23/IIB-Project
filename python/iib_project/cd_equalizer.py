import numpy as np
from typing import List
from fxpmath import Fxp
from iib_project.fft_fp import FFT_fp
class CD_Equalizer:
    def __init__(self, D: float, L: float, symbol_rate: float, sps: int, wavelength: float, M_fft: int, DW_io: int, DW_fft: int, c: float = 3e8):
        self.N_fft = 2**M_fft  # Number of FFT points

        self.D = D * 1e-6  # Dispersion parameter in ps/(nm·km) converted to s/(m·m)
        self.L = L * 1e3  # Fiber length in km converted to m
        self.symbol_rate = symbol_rate * 1e9  # Symbol rate in GBd converted to Bd
        self.sps = sps  # Samples per symbol
        self.wavelength = wavelength * 1e-9  # Wavelength in nm converted to m
        self.nyq_freq = self.symbol_rate * self.sps / 2  # Nyquist frequency
        self.c = c  # Speed of light in m/s
        
        self.io_t = Fxp(dtype=f'fxp-s{DW_io}/{DW_io-1}-complex')
        self.fft_t = Fxp(dtype=f'fxp-s{DW_fft}/{DW_fft - M_fft - 1}-complex')
        self.curr = [Fxp(0).like(self.io_t) for _ in range(self.N_fft)]  # Current input block 
        self.prev = [Fxp(0).like(self.io_t) for _ in range(self.N_fft)]  # Previous input block 
        self.overlap = [Fxp(0).like(self.io_t) for _ in range(self.N_fft)]  # Overlapped block
        self.N_cd = self.get_Ncd() # Overlap length
        self.H_cd = self.get_Hcd() # Frequency response of CD equalizer
        self.H_cd_fxp = [Fxp(h).like(self.fft_t) for h in self.H_cd]  # Fixed-point H_cd

        self.fft = FFT_fp(M_fft, DW_io, DW_fft)

    def get_Ncd(self) -> int:
        N_cd = int(np.ceil(6.67 / (2 * np.pi * self.c) * self.D * self.L * (self.wavelength ** 2) * (self.symbol_rate ** 2) * self.sps))
        if N_cd >= self.N_fft:
            raise ValueError(f"Calculated overlap length N_cd={N_cd} exceeds or equals FFT size N_fft={self.N_fft}. Increase N_fft.")
        return N_cd

    def get_Hcd(self) -> np.ndarray:
        n = np.arange(-self.N_fft // 2, self.N_fft // 2)
        H_cd = np.exp(-1j * np.pi * self.wavelength ** 2 * self.D * self.L / self.c * (n * 2 * self.nyq_freq / self.N_fft) ** 2)
        return H_cd

    def pipeline(self, x: List[Fxp]) -> None:
        if len(x) != self.N_fft:
            raise ValueError(f"Input length {len(x)} does not match FFT size {self.N_fft}")
        self.prev = self.curr.copy()
        self.curr = x.copy()
        self.curr = [Fxp(val).like(self.io_t) for val in self.curr] # Ensure correct fixed-point format
        for i in range(self.N_cd):
            self.overlap[i] = self.prev[self.N_fft - self.N_cd + i] # Last N_cd samples from previous block
        for i in range(self.N_cd, self.N_fft):
            self.overlap[i] = self.curr[i - self.N_cd] # First N_fft - N_cd samples from current block

    def equalize(self, x: List[Fxp]) -> List[Fxp]:
        self.pipeline(x)
        X_fft = self.fft.fft(self.overlap, inverse=False)
        Y_fft = [Fxp(X_fft[i] * self.H_cd_fxp[i]).like(self.fft_t) for i in range(self.N_fft)]
        y_ifft = self.fft.fft(Y_fft, inverse=True)
        y_out = [Fxp(0).like(self.io_t) for _ in range(self.N_fft - self.N_cd)]
        for i in range(self.N_cd, self.N_fft):
            y_out[i - self.N_cd] = Fxp(y_ifft[i]).like(self.io_t)
        return y_out

    # Test
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from iib_project.channel import Channel
    from iib_project.modulator import Modulator
    from iib_project.plotting import plot_tx_constellation

    # Example parameters
    D = 17  # ps/(nm·km)
    L = 3   # km
    symbol_rate = 32  # GBd
    sps = 2  # Samples per symbol
    wavelength = 1550  # nm
    M_fft = 6 # FFT size exponent (N_fft = 2^M_fft)
    DW_io = 16  # Input/Output data width
    DW_fft = 24  # FFT data width
    SNR = 0

    modulator = Modulator(1)
    channel = Channel(SNR=SNR, sps=sps, symbol_rate=symbol_rate*1e9, D=D, L=L*1e3, wavelength=wavelength*1e-9)
    cd_equalizer = CD_Equalizer(D=D, L=L, symbol_rate=symbol_rate, sps=sps, wavelength=wavelength, M_fft=M_fft, DW_io=DW_io, DW_fft=DW_fft)
    print(cd_equalizer.N_cd)

    # Generate a test signal
    num_symbols = 2**6
    tx_symbols = modulator.gen_symbols(num_symbols)
    H_cd_channel = np.fft.fft(channel.add_chromatic_dispersion(tx_symbols)) / np.fft.fft(tx_symbols)

    # compare channel and equalizer frequency responses (check phases)
    plt.figure()
    freq = np.fft.fftfreq(len(H_cd_channel), d=1/(sps * symbol_rate * 1e9))
    plt.plot(freq, np.angle(H_cd_channel), label='Channel CD Phase Response')
    plt.plot(freq, np.angle(cd_equalizer.H_cd), label='Equalizer CD Phase Response', linestyle='--')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Phase (radians)')
    plt.title('Chromatic Dispersion Phase Response Comparison')
    plt.legend()
    plt.grid()
    plt.show()
