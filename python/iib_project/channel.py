import numpy as np

class Channel:
    """ A simulation of the optical channel. Adds noise, dispersion and other impairments to the signal. """
    def __init__(self, SNR: float, sps: int, symbol_rate: float, D: float, L: float, wavelength: float, DGDSpec: float = 0.3, N_pmd: int = 10, total_linewidth : float = 1e6, c: float = 3e8):
        self.SNR = SNR #dB
        self.sps = sps 
        self.symbol_rate = symbol_rate * 1e9 # GBd -> Bd
        self.D = D * 1e-6 # ps/(nm*km) -> s/(m*m)
        self.L = L * 1e3 # km -> m
        self.wavelength = wavelength * 1e-9 # nm -> m
        self.c = c # m/s
        self.DGDSpec = DGDSpec
        self.N_pmd = N_pmd
        self.total_linewidth = total_linewidth  


    def add_AWGN(self, signal: np.ndarray) -> np.ndarray:
        """ Add Additive White Gaussian Noise (AWGN) to the signal. """
        n_pols, n_samples = signal.shape
        signal_power = np.mean(np.abs(signal)**2)
        noise_power = signal_power / (10**(self.SNR / 10))
        noise = np.sqrt(noise_power / 2) * (np.random.randn(n_pols, n_samples) + 1j * np.random.randn(n_pols, n_samples)) 
        return signal + noise

    def add_chromatic_dispersion(self, signal: np.ndarray) -> np.ndarray:
        """ Add chromatic dispersion to the signal. """
        n_pols, n_samples = signal.shape
        signal_cd = np.zeros_like(signal, dtype=complex)
        n = np.arange(-n_samples // 2, n_samples // 2)
        G = np.exp(1j * np.pi * self.wavelength ** 2 * self.D * self.L / self.c * (n * self.sps * self.symbol_rate / n_samples) ** 2)
        for i in range(n_pols):
            signal_fft = np.fft.fftshift(np.fft.fft(signal[i]))
            signal_cd_fft = signal_fft * G
            signal_cd[i] = np.fft.ifft(np.fft.ifftshift(signal_cd_fft))
        return signal_cd

    def add_pmd(self, EInput: np.ndarray) -> np.ndarray:
        # Standard deviation of the Maxwellian distribution
        SDTau = np.sqrt(3 * np.pi / 8) * self.DGDSpec
        # DGD per section (in seconds)
        Tau = (SDTau * np.sqrt(self.L * 1e-3) / np.sqrt(self.N_pmd)) * 1e-12
        # Frequency vector (rad/s)
        nSamples = EInput.shape[0]
        w = 2 * np.pi * np.fft.fftshift(np.linspace(-0.5, 0.5 - 1/nSamples, nSamples)) * self.sps * self.symbol_rate 
    
        # Random unitary matrices for each section
        V = np.zeros((2, 2, self.N_pmd), dtype=complex)
        U = np.zeros((2, 2, self.N_pmd), dtype=complex)
        for i in range(self.N_pmd):
            rand_mat = np.random.randn(2, 2) + 1j * np.random.randn(2, 2)
            U_, _, Vh_ = np.linalg.svd(rand_mat)
            V[:, :, i] = U_
            U[:, :, i] = Vh_.conj().T
    
        # Input signals in frequency domain
        Freq_E_V = np.fft.fft(EInput[:, 0])
        Freq_E_H = np.fft.fft(EInput[:, 1])
    
        # Apply PMD model
        for i in range(self.N_pmd):
            UHerm = U[:, :, i].conj().T
    
            # Rotate input fields
            E1 = UHerm[0, 0] * Freq_E_V + UHerm[0, 1] * Freq_E_H
            E2 = UHerm[1, 0] * Freq_E_V + UHerm[1, 1] * Freq_E_H
    
            # Apply DGD phase shift
            E1 *= np.exp(1j * w * Tau / 2)
            E2 *= np.exp(-1j * w * Tau / 2)
    
            # Rotate using V
            Freq_E_V = V[0, 0, i] * E1 + V[0, 1, i] * E2
            Freq_E_H = V[1, 0, i] * E1 + V[1, 1, i] * E2
    
        # Convert back to time domain
        EOutput = np.zeros_like(EInput, dtype=complex)
        EOutput[:, 0] = np.fft.ifft(Freq_E_V)
        EOutput[:, 1] = np.fft.ifft(Freq_E_H)
    
        return EOutput

    def add_phase_noise(self, signal: np.ndarray) -> np.ndarray:
        """ Add Wiener Phase Noise to the signal """
        n_pols, n_samples = signal.shape
        phase_noise = np.zeros_like(signal, dtype=complex)
        delta_phi_std = np.sqrt(2 * np.pi * self.total_linewidth / self.symbol_rate)
        for i in range(n_pols):
            delta_phi = np.random.randn(n_samples) * delta_phi_std
            phi = np.cumsum(delta_phi)
            phase_noise[i] = signal[i] * np.exp(1j * phi)

        return phase_noise


         

        
