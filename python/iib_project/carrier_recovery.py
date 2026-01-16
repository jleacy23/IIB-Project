import numpy as np
from fxpmath import Fxp
class Carrier_Recovery:
    def __init__(self, symbol_rate: float, sps: int, DW_acc: int, DW_io: int, pilot_interval: int = 1):
        self.symbol_rate = symbol_rate * 1e9 # GHz -> Hz
        self.sps = sps
        self.acc_t = Fxp(dtype=f'fxp-s{DW_acc}/{DW_io-1}-complex')
        self.pilot_interval = pilot_interval

    def viterbi_viterbi_ML(self,
                           total_linewidth: float,
                           snr_db: float,
                           symbol_energy: float,
                           N: int):
        L = 2 * N + 1
        Ts = 1 / self.symbol_rate
    
        # Phase noise variance per symbol (Wiener process)
        var_phi = 2 * np.pi * total_linewidth * Ts
   
        # Es/N0
        snr_linear = 10 ** (snr_db / 10)
        var_eta = symbol_energy / (2 * snr_linear)
    
        # min(i,j) covariance kernel
        k_aux = np.fromfunction(lambda i, j: np.minimum(i + 1, j + 1), (N, N))
        k = np.zeros((L, L))
        k[:N, :N] = np.rot90(k_aux, 2)
        k[N + 1:, N + 1:] = k_aux
    
        # Covariance matrix (VV fourth-power statistics)
        C = (16 * symbol_energy**4 * var_phi * k +
             16 * symbol_energy**3 * var_eta * np.eye(L))
    
        # ML filter (C^{-1} 1) normalized
        ones = np.ones((L, 1))
        w = np.linalg.solve(C, ones)
        w /= np.sum(w)
    
        return w.flatten()



    def phase_unwrap(self, phase: np.ndarray, pilots: np.ndarray) -> np.ndarray:
        unwrapped_phase = phase.copy()
        for i in range(1, len(phase)):
            delta = phase[i] - phase[i - 1]
            unwrapped_phase[i] = phase[i] + np.floor(0.5 + delta / (2 * np.pi / 4)) * (2 * np.pi / 4)

        # apply pilot-based correction
        for i in range(len(pilots)):
            if i == 0:
                continue
            theta_pilot = pilots[i]
            theta_est = unwrapped_phase[i * self.pilot_interval]
            offset = self.pilot_phase_correction(theta_pilot, theta_est)
            unwrapped_phase[i * self.pilot_interval:] += offset

        return unwrapped_phase

    def pilot_phase_correction(self, theta_pilot: float, theta_est: float) -> float:
        delta_theta = theta_pilot - theta_est
        # wrap to -pi, pi
        delta_theta = (delta_theta + np.pi) % (2 * np.pi) - np.pi

        offset = np.round(delta_theta / (np.pi / 2)) * (np.pi / 2)

        if np.abs(offset) > 0:
            print(f'Cycle slip detected: {offset:.2f} rad')
        return offset

    def viterbi_viterbi_ref(self,
                            x: np.ndarray,
                            N: int,
                            total_linewidth: float,
                            snr_db: float,
                            symbol_energy: float,
                            pilots: np.ndarray) -> np.ndarray:
        K = len(x)
        L = 2 * N + 1
        w = self.viterbi_viterbi_ML(
            total_linewidth, snr_db, symbol_energy, N
        )
    
        # Build sliding window (ignore edges)
        z_blocks = np.zeros((L, K), dtype=complex)
        for k in range(N, K - N):
            z_blocks[:, k] = x[k - N:k + N + 1]
    
        # Fourth-power phase
        phi4 = np.angle(w.conj() @ (z_blocks ** 4))
    
        # Correct unwrapping (BEFORE dividing by 4)
        phi4 = np.unwrap(phi4)
    
        # Phase estimate
        theta = phi4 / 4 - np.pi / 4
    
        # Phase ambiguity and cycle slip correction with pilots
        current_amb = None

        for i, theta_pilot in enumerate(pilots):
            idx = i * self.pilot_interval
            if idx >= K:
                break
    
            theta_est = theta[idx]
            theta_ref = np.angle(x[idx]) - theta_pilot
            k_amb = int(np.round((theta_ref - theta_est) / (np.pi / 2))) 

            if current_amb is None:
                theta += k_amb * np.pi/2
                current_amb = k_amb
            elif k_amb != current_amb: #cycle slip has occurred
                delta = (k_amb - current_amb) * np.pi/2
                theta += delta
                current_amb = k_amb
    
        # Apply correction
        y = x * np.exp(-1j * theta)
    
        return y

    def viterbi_viterbi_fxp(self, x: Fxp, N: int, total_linewidth: float, snr: float, symbol_energy: float, pilots: np.ndarray, step: int = 1) -> Fxp:
        w = self.viterbi_viterbi_ML(total_linewidth, snr, symbol_energy, N)
        L = 2 * N + 1
        N_pol, K = x.shape

        y = Fxp(np.zeros((N_pol, K), dtype=complex)).like(self.acc_t)

        for pol in range(N_pol):
            x_pol = x[pol, :]
            pilots_pol = pilots[pol, :]

            #form overlapping blocks
            z_blocks = (np.zeros((L, K), dtype=complex))
            for k in range(K):
                for n in range(-N, N + 1):
                    idx = k + n
                    if idx < 0:
                        z_blocks[n + N, k] = Fxp(0).like(self.acc_t)
                    elif idx >= K:
                        z_blocks[n + N, k] = Fxp(0).like(self.acc_t)
                    else:
                        z_blocks[n + N, k] = x_pol[idx]

            #apply phase correction
            thetas = (1/4) * Fxp(np.angle(w.conj() @ (z_blocks ** 4))).like(self.acc_t) - Fxp(np.pi / 4).like(self.acc_t)
            #update thetas every 'step' samples
            thetas = thetas[::step]
            thetas = Fxp(np.repeat(thetas, step))[:K].like(self.acc_t)
            thetas_unwrapped = self.phase_unwrap(thetas, pilots_pol)
            thetas_unwrapped_fxp = Fxp(thetas_unwrapped).like(self.acc_t)
            corrections = Fxp(np.exp(-1j * thetas_unwrapped_fxp)).like(self.acc_t)
            y_pol = corrections * x_pol
            y[pol, :] = y_pol
        
        return y
def viterbi_viterbi_ref(self,
                        x: np.ndarray,
                        N: int,
                        total_linewidth: float,
                        snr_db: float,
                        symbol_energy: float,
                        pilots: np.ndarray) -> np.ndarray:
    """
    Viterbi–Viterbi carrier phase recovery with pilot-aided
    π/2 ambiguity resolution.
    """
    K = len(x)
    L = 2 * N + 1
    w = self.viterbi_viterbi_ML(
        total_linewidth, snr_db, symbol_energy, N
    )

    # Build sliding window (ignore edges)
    z_blocks = np.zeros((L, K), dtype=complex)
    for k in range(N, K - N):
        z_blocks[:, k] = x[k - N:k + N + 1]

    # Fourth-power phase
    phi4 = np.angle(w.conj() @ (z_blocks ** 4))

    # Correct unwrapping (BEFORE dividing by 4)
    phi4 = np.unwrap(phi4)

    # Phase estimate
    theta = phi4 / 4

    # --- Pilot-based π/2 ambiguity resolution ---
    for i, theta_pilot in enumerate(pilots):
        idx = i * self.pilot_interval
        if idx >= K:
            break

        theta_est = theta[idx]
        k_amb = np.round((theta_pilot - theta_est) / (np.pi / 2))
        theta += k_amb * (np.pi / 2)

    # Apply correction
    y = x * np.exp(-1j * theta)

    return y





