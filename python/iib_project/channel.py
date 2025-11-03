import numpy as np

class Channel:
    """ A simulation of the optical channel. Adds noise, dispersion and other impairments to the signal. """
    def __init__(self, SNR: float, sps: int, symbol_rate: float, D: float, L: float, wavelength: float, c: float = 3e8):
        self.SNR = SNR #dB
        self.sps = sps 
        self.symbol_rate = symbol_rate * 1e9 # GBd -> Bd
        self.D = D * 1e-6 # ps/(nm*km) -> s/(m*m)
        self.L = L * 1e3 # km -> m
        self.wavelength = wavelength * 1e-9 # nm -> m
        self.c = c # m/s

    def add_AWGN(self, signal: np.ndarray) -> np.ndarray:
        """ Add Additive White Gaussian Noise (AWGN) to the signal. """
        N = len(signal)
        signal_power = np.mean(np.abs(signal)**2)
        noise_power = signal_power / (10**(self.SNR / 10))
        noise = np.sqrt(noise_power / 2) * (np.random.randn(N) + 1j * np.random.randn(N))
        return signal + noise

    def add_chromatic_dispersion(self ,signal: np.ndarray) -> np.ndarray:
        """ Add chromatic dispersion to the signal. """
        N = len(signal)
        n = np.arange(-N//2, N//2)
        G = np.exp(1j * np.pi * self.wavelength ** 2 * self.D * self.L / self.c * (n * self.sps * self.symbol_rate / N) ** 2)
        signal_fft = np.fft.fftshift(np.fft.fft(signal))
        signal_cd_fft = signal_fft * G
        signal_cd = np.fft.ifft(np.fft.ifftshift(signal_cd_fft))
        return signal_cd
         

        
