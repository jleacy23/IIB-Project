import numpy as np
from fxpmath import Fxp
from iib_project.cordic import Cordic

class Carrier_Recovery:
    def __init__(self, symbol_rate: float, sps: int, DW_acc: int, pilot_interval: int = 1):
        self.symbol_rate = symbol_rate * 1e9 # GHz -> Hz
        self.sps = sps
        self.acc_t = Fxp(dtype=f'fxp-s{DW_acc}/{DW_acc-3}-complex') 
        self.theta_t = Fxp(dtype=f'fxp-s{DW_acc}/{DW_acc-3}') # Account for unwrapping
        self.pilot_interval = pilot_interval
        self.cordic = Cordic(iterations=DW_acc-3, word_length=DW_acc)

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

    def viterbi_viterbi_ref(self,
                            x: np.ndarray,
                            N: int,
                            total_linewidth: float,
                            snr_db: float,
                            symbol_energy: float,
                            step: int = 1)-> np.ndarray:
        L = 2 * N + 1
        N_pol, K = x.shape
        w = self.viterbi_viterbi_ML(
            total_linewidth, snr_db, symbol_energy, N
        )
        y = np.zeros_like(x)
    
        # Build sliding window (ignore edges)
        for pol in range(N_pol):
            z_blocks = np.zeros((L, K), dtype=complex)
            for k in range(N, K - N):
                z_blocks[:, k] = x[pol, k - N:k + N + 1]
    
            # Fourth-power phase
            phi4 = np.angle(w.conj() @ (z_blocks ** 4))

            # Only use every 'step' samples to reduce complexity
            phi4 = np.repeat(phi4[::step], step)[:K]
    
            # Correct unwrapping (BEFORE dividing by 4)
            phi4 = np.unwrap(phi4)
    
            # Phase estimate
            theta = phi4 / 4 - np.pi / 4
    
            # Apply correction
            y[pol, :] = x[pol, :] * np.exp(-1j * theta)
    
        return y

    def viterbi_viterbi_fxp(self,
                            x: Fxp,
                            N: int,
                            total_linewidth: float,
                            snr_db: float,
                            symbol_energy: float,
                            cordic_its: int,
                            pilots: Fxp,
                            step: int = 1) -> Fxp:
        """ Implements Viterbi-Viterbi using fixed-point arithmetic and CORDIC """
        N_pol, K = x.shape
        L = 2 * N + 1
        P = pilots.shape[1]
        w = self.viterbi_viterbi_ML(
            total_linewidth, snr_db, symbol_energy, N
        )
        w_fxp = Fxp(w).like(self.acc_t)
        y = Fxp(np.zeros((N_pol, K), dtype=complex)).like(self.acc_t)
        for pol in range(N_pol):
            # Build sliding window (ignore edges)
            z_blocks = Fxp(np.zeros((L, K), dtype=complex)).like(self.acc_t)
            for k in range(N, K - N):
                z_blocks[:, k] = x[pol, k - N:k + N + 1]
    
            # Fourth-power phase
            z_blocks_4 = Fxp(z_blocks * z_blocks * z_blocks * z_blocks).like(self.acc_t)
            est = Fxp(w_fxp.conj().dot(z_blocks_4)).like(self.acc_t)
            phi4 = self.cordic.complex_phase(est)

            # Only use every 'step' samples to reduce complexity
            phi4 = np.repeat(phi4[::step], step)[:K]
    
            # Phase estimate
            theta = Fxp(phi4 / 4 - np.pi / 4).like(self.theta_t)
    
            # Phase ambiguity and cycle slip correction with pilots
            current_amb = None

            #for i in range(P):
             #   idx = i * self.pilot_interval
              #  if idx >= K:
               #     break
    
                #theta_est = theta[idx]
                #theta_ref = Fxp(np.angle(x[pol, idx]) - pilots[pol][i]).like(self.theta_t)
                #k_amb = int(np.round((theta_ref - theta_est) / (np.pi / 2))) 

                #if current_amb is None:
                    #theta += Fxp(k_amb * np.pi/2).like(self.theta_t)
                    #current_amb = k_amb
                #elif k_amb != current_amb: #cycle slip has occurred
                 #   delta = Fxp((k_amb - current_amb) * np.pi/2).like(self.theta_t)
                  #  theta[idx:] += delta
                   # current_amb = k_amb

            #wrap to [-pi, pi]
           # theta_np = theta.get_val()
            #theta_np = (theta_np + np.pi) % (2 * np.pi) - np.pi
            #theta = Fxp(theta_np).like(self.theta_t)
            
            # Apply correction using CORDIC
            y[pol, :] = self.cordic.complex_rotate(x[pol, :], -theta)

        return Fxp(y).like(self.acc_t)


