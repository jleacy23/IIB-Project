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
        self.overlap = [Fxp(0).like(self.io_t) for _ in range(self.N_fft)]  # Overlapped block
        self.N_cd = self.get_Ncd() # Overlap length
        self.H_cd = self.get_Hcd() # Frequency response of CD equalizer

        self.fft = FFT_fp(M_fft, DW_io, DW_fft)

    def get_Ncd(self) -> int:
        N_cd = int(np.ceil(6.67 / (2 * np.pi * self.c) * self.D * self.L * (self.wavelength ** 2) * (self.symbol_rate ** 2) * self.sps))
        if N_cd >= self.N_fft:
            raise ValueError(f"Calculated overlap length N_cd={N_cd} exceeds or equals FFT size N_fft={self.N_fft}. Increase N_fft.")
        if N_cd % 2 != 0:
            N_cd += 1
        return N_cd

    def get_Hcd(self) -> np.ndarray:
        n = np.arange(-self.N_fft // 2, self.N_fft // 2)
        H_cd = np.exp(-1j * np.pi * self.wavelength ** 2 * self.D * self.L / self.c *(n * 2 * self.nyq_freq / self.N_fft) ** 2) #filter is symmetric around 0 -> non-causal
        # convert to fixed-point
        H_cd_fxp = [Fxp(val).like(self.fft_t) for val in H_cd]
        # align frequency bins for FFT
        H_cd_fxp = H_cd_fxp[self.N_fft // 2:] + H_cd_fxp[:self.N_fft // 2]
        return H_cd_fxp
    
    def reset(self):
        self.curr = [Fxp(0).like(self.io_t) for _ in range(self.N_fft)]
        self.prev = [Fxp(0).like(self.io_t) for _ in range(self.N_fft)]
        self.overlap = [Fxp(0).like(self.io_t) for _ in range(self.N_fft)]

    def roll(self, x: List[Fxp], shift: int) -> List[Fxp]:
        N = len(x)
        shift = shift % N
        return x[-shift:] + x[:-shift]

    def pipeline(self, x: List[Fxp]):
        if len(x) != self.N_fft - self.N_cd:
            raise ValueError(f"Input block size {len(x)} does not match expected size {self.N_fft - self.N_cd}.")
        # last N_cd samples from previous block
        self.overlap = self.overlap[self.N_fft - self.N_cd:] + x

    def equalize_block(self) -> List[Fxp]:
        # Apply circular shift to input to accoount for non-causal filter
        overlap_shifted = self.roll(self.overlap, self.N_cd // 2)
        X = self.fft.fft(overlap_shifted, inverse=False)
        Y = [X[k] * self.H_cd[k] for k in range(self.N_fft)]
        y = self.fft.fft(Y, inverse=True)
        # Apply circular shift back to original order
        y_shifted = self.roll(y, -self.N_cd // 2)
        # Discard first N_cd samples
        return y_shifted[self.N_cd:]

    def pad_and_block(self, x: List[Fxp]) -> List[List[Fxp]]:
        step = self.N_fft - self.N_cd 
        x_len = len(x)
        num_blocks = int(np.ceil(x_len / step))
        padded_len = num_blocks * step # no need to prepend zeros, handled in pipeline
        x_padded = x + [Fxp(0).like(self.io_t) for _ in range(padded_len - x_len)]
        blocks = [x_padded[i * step : i * step + step] for i in range(num_blocks)]
        return blocks

    def equalize(self, x: List[Fxp]) -> List[Fxp]:
        blocks = self.pad_and_block(x)
        y = []
        self.reset()
        for block in blocks:
            self.pipeline(block)
            y_block = self.equalize_block()
            y.extend(y_block)
        return y[:len(x)]  # Trim to original length
