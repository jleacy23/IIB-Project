import numpy as np
import scipy.fft

class Channel:
    """ A simulation of the optical channel. Adds noise, dispersion and other impairments to the signal. """
    def __init__(self, SNR: float, sps: int, symbol_rate: float, D: float, L: float, wavelength: float):
        """
        Initialize the Channel with given parameters.

        SNR: Signal-to-Noise Ratio in dB
        sps: Samples per symbol
        symbol_rate: Symbol rate in symbols/s
        D: Dispersion parameter in ps/(nm*km)
        L: Fiber length in m
        wavelength: Central wavelength in m
        """
        self.SNR = SNR
        self.sps = sps
        self.symbol_rate = symbol_rate
        self.D = D
        self.L = L
        self.wavelength = wavelength

    def add_AWGN(self, signal: np.ndarray) -> np.ndarray:
        """ Add Additive White Gaussian Noise (AWGN) to the signal. """
        N = len(signal)
        signal_power = np.mean(np.abs(signal)**2)
        noise_power = signal_power / (10**(self.SNR / 10))
        noise = np.sqrt(noise_power / 2) * (np.random.randn(N) + 1j * np.random.randn(N))
        return signal + noise

    def add_chromatic_dispersion(self ,signal: np.ndarray) -> np.ndarray:
        """ Add chromatic dispersion to the signal. """
        c = 299792458  # Speed of light in m/s
        D_conv = self.D * 1e-6 # Convert to seconds and meters
        
        beta2 = - (D_conv * (self.wavelength**2)) / (2 * np.pi * c)  # s^2/m

        N = len(signal)
        freq = scipy.fft.fftfreq(N, d=1/(self.sps * self.symbol_rate))
        H_cd = np.exp(-1j * (0.5 * beta2 * (2 * np.pi * freq)**2) * self.L)
        signal_freq = scipy.fft.fft(signal)
        signal_cd_freq = signal_freq * H_cd
        signal_cd = scipy.fft.ifft(signal_cd_freq)

        return signal_cd


