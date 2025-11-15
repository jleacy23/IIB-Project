import numpy as np
from fxpmath import Fxp
from iib_project.fft_fp import FFT_fp

class Adaptive_Equalizer:
    def __init__(self, num_taps: int, step_size: float, DW_io: int, DW_acc: int):
        self.num_taps = num_taps
        self.io_t = Fxp(dtype='fxp-s{DW_io}/{DW_io-1}-complex')
        self.acc_t = Fxp(dtype=f'fxp-s{DW_acc}/{DW_acc-1-int(np.ceil(np.log2(step_size)))}-complex')
        self.step_size = Fxp(step_size).like(self.acc_t)
        self.convergence_threshold = Fxp(1e-5).like(self.acc_t)
        self.w_1v = Fxp([0]*num_taps).like(self.acc_t)
        self.w_1h = Fxp([0]*num_taps).like(self.acc_t)
        self.w_2v = Fxp([0]*num_taps).like(self.acc_t)
        self.w_2h = Fxp([0]*num_taps).like(self.acc_t)

    def reset(self):
        self.w_1v = Fxp([0]*self.num_taps).like(self.acc_t)
        self.w_1h = Fxp([0]*self.num_taps).like(self.acc_t)
        self.w_2v = Fxp([0]*self.num_taps).like(self.acc_t)
        self.w_2h = Fxp([0]*self.num_taps).like(self.acc_t)

        self.w_1v[self.num_taps//2] = Fxp(1).like(self.acc_t) #avoids singularity

    def get_Rd(self, x: Fxp) -> Fxp:
        N = len(x)
        x_abs = [np.abs(xi.get_val()) for xi in x]
        R_d = sum([x_abs[i]**4 / x_abs[i]**2 for i in range(N)]) / N
        return Fxp(R_d).like(self.acc_t)

    def update_weights(self, xv_block: Fxp, xh_block: Fxp, e1: Fxp, e2: Fxp):
        "Update filters with stochastic gradient descent"
        self.w_1v = self.w_1v + self.step_size * e1.conj() * xv_block
        self.w_1h = self.w_1h + self.step_size * e1.conj() * xh_block
        self.w_2v = self.w_2v + self.step_size * e2.conj() * xv_block
        self.w_2h = self.w_2h + self.step_size * e2.conj() * xh_block

    def equalize_sample(self, xv_block: Fxp, xh_block: Fxp) -> (Fxp, Fxp):
        "Equalize single sample using current weights"
        y1 = self.w_1v.conj().dot(xv_block) + self.w_1h.conj().dot(xh_block)
        y2 = self.w_2v.conj().dot(xv_block) + self.w_2h.conj().dot(xh_block)
        return y1, y2

    def get_errors(self, y1: Fxp, y2: Fxp, xv: Fxp, xh: Fxp, type: str) -> (Fxp, Fxp):
        if type == 'CMA':
            R_d = self.get_Rd(xv.tolist() + xh.tolist())
            e1 = (R_d - y1 * y1.conj()) * y1
            e2 = (R_d - y2 * y2.conj()) * y2
        else:
            raise ValueError(f"Unsupported equalization type: {type}")
        return e1, e2

    def equalize(self, xv: Fxp, xh: Fxp, type: str) -> (Fxp, Fxp):
        "Equalize input signals xv and xh"""
        #pad input signals
        pad = self.num_taps - 1
        xv_padded = Fxp([0]*pad + xv.tolist() + [0]*pad).like(self.acc_t)
        xh_padded = Fxp([0]*pad + xh.tolist() + [0]*pad).like(self.acc_t)
        N = len(xv)
        y1_out = []
        y2_out = []
        # T/2 spaced inputs, T spaced outputs
        e1_converged = False
        for k in range(0, N, 2):
            xv_block = xv_padded[k:k+self.num_taps][::-1]
            xh_block = xh_padded[k:k+self.num_taps][::-1]
            y1, y2 = self.equalize_sample(xv_block, xh_block)
            e1, e2 = self.get_errors(y1, y2, xv, xh, type)
            self.update_weights(xv_block, xh_block, e1, e2)
            y1_out.append(y1)
            if e1_converged:
                y2_out.append(y2)
            if e1 < self.convergence_threshold and e1_converged is False:
                e1_converged = True
                self.w_2v = self.w_1v.conj()[::-1].like(self.acc_t)
                self.w_2h = -self.w_1h.conj()[::-1].like(self.acc_t)

        if e1_converged is False:
            raise Warning("Equalizer did not converge")
                
        return Fxp(y1_out.tolist()).like(self.acc_t), Fxp(y2_out.tolist()).like(self.acc_t)
