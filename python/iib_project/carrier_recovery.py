import numpy as np
from fxpmath import Fxp
class Carrier_Recovery:
    def __init__(self, symbol_rate: float, sps: int, DW_acc: int, DW_io: int):
        self.symbol_rate = symbol_rate * 1e9 # GHz -> Hz
        self.sps = sps
        self.acc_t = Fxp(dtype=f'fxp-s{DW_acc}/{DW_io-1}-complex')

    def viterbi_viterbi_ML(self, total_linewidth: float, snr: float, symbol_energy: float, N: int):
        """ Generate the ML filter for Viterbi-Viterbi carrier recovery algorithm. """
        L = 2 * N + 1 #filter length
        Ts = 1 / self.symbol_rate
        var_linewidth = 2 * np.pi * total_linewidth * Ts
        snr_linear = 10 ** (snr / 10) * (2 * 12.5e9) / self.symbol_rate #why is there 12.5e9 here??
        var_eta = symbol_energy / (2 * snr_linear)

        k_aux = np.zeros((N,N))
        k = np.zeros((L,L))

        for n in range(1, N + 1):
            for m in range(1, N + 1):
                k_aux[n-1,m-1] = min(n,m)

        k[:N, :N] = np.rot90(k_aux, 2)
        k[N+1:, N+1:] = k_aux

        # covariance matrix
        C = symbol_energy**4 * 16 * var_linewidth * k + symbol_energy**3 * 16 * var_eta * np.eye(L)

        # ML filter
        w = np.dot(np.ones((L, 1)).T, np.linalg.inv(C))
        w = w / np.max(w)

        return Fxp(w).like(self.acc_t)

    def phase_unwrap(self, phase: np.ndarray) -> np.ndarray:
        unwrapped_phase = phase.copy()
        for i in range(1, len(phase)):
            delta = phase[i] - phase[i - 1]
            unwrapped_phase[i] = phase[i] + np.floor(0.5 + delta / (2 * np.pi / 4)) * (2 * np.pi / 4)

        return unwrapped_phase

    def viterbi_viterbi_ref(self, x: np.ndarray, N: int, total_linewidth: float, snr: float, symbol_energy: float) -> np.ndarray:
        w = self.viterbi_viterbi_ML(total_linewidth, snr, symbol_energy, N)
        L = 2 * N + 1
        K = len(x)

        #form overlapping blocks
        z_blocks = np.zeros((L, K), dtype=complex)
        for k in range(K):
            for n in range(-N, N + 1):
                idx = k + n
                if idx < 0:
                    z_blocks[n + N, k] = 0
                elif idx >= K:
                    z_blocks[n + N, k] = 0
                else:
                    z_blocks[n + N, k] = x[idx]

        #apply phase correction
        thetas = (1/4) * np.angle(w.conj() @ (z_blocks ** 4)) - np.pi / 4
        thetas_unwrapped = self.phase_unwrap(thetas)
        corrections = np.exp(-1j * thetas_unwrapped)
        y = corrections * x
        
        return y

    def viterbi_viterbi_fxp(self, x: Fxp, N: int, total_linewidth: float, snr: float, symbol_energy: float, step: int = 1) -> Fxp:
        w = self.viterbi_viterbi_ML(total_linewidth, snr, symbol_energy, N)
        L = 2 * N + 1
        N_pol, K = x.shape

        y = Fxp(np.zeros((N_pol, K), dtype=complex)).like(self.acc_t)

        for pol in range(N_pol):
            x_pol = x[pol, :]

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
            thetas_unwrapped = self.phase_unwrap(thetas)
            thetas_unwrapped_fxp = Fxp(thetas_unwrapped).like(self.acc_t)
            corrections = Fxp(np.exp(-1j * thetas_unwrapped_fxp)).like(self.acc_t)
            y_pol = corrections * x_pol
            y[pol, :] = y_pol
        
        return y




