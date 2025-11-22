import numpy as np
from fxpmath import Fxp
from iib_project.fft_fp import FFT_fp

class Adaptive_Equalizer:
    def __init__(self, num_taps: int, step_size: float, DW_io: int, DW_acc: int):
        self.num_taps = num_taps
        self.io_t = Fxp(dtype=f'fxp-s{DW_io}/{DW_io-1}-complex')
        self.acc_t = Fxp(dtype=f'fxp-s{DW_acc}/{DW_io-1}-complex')
        self.acc_t.info(2)
        self.step_size = Fxp(step_size).like(self.acc_t)
        self.w_1v = Fxp([0]*num_taps).like(self.acc_t)
        self.w_1v[self.num_taps//2] = Fxp(1).like(self.acc_t) #avoids singularity
        self.w_1h = Fxp([0]*num_taps).like(self.acc_t)
        self.w_2v = Fxp([0]*num_taps).like(self.acc_t)
        self.w_2h = Fxp([0]*num_taps).like(self.acc_t)

    def get_Rd(self, x: Fxp) -> Fxp:
        R_d = np.mean(np.abs(x)**4) / np.mean(np.abs(x)**2)
        return Fxp(R_d).like(self.acc_t)

    def update_weights(self, xv_block: Fxp, xh_block: Fxp, e1: Fxp, e2: Fxp):
        "Update filters with stochastic gradient descent"
        self.w_1v = self.w_1v.like(self.acc_t) + self.step_size * e1.conj() * xv_block.like(self.acc_t)
        self.w_1h = self.w_1h.like(self.acc_t) + self.step_size * e1.conj() * xh_block.like(self.acc_t)
        self.w_2v = self.w_2v.like(self.acc_t) + self.step_size * e2.conj() * xv_block.like(self.acc_t)
        self.w_2h = self.w_2h.like(self.acc_t) + self.step_size * e2.conj() * xh_block.like(self.acc_t)
    def equalize_sample(self, xv_block: Fxp, xh_block: Fxp) -> (Fxp, Fxp):
        "Equalize single sample using current weights"
        y1 = xv_block.like(self.acc_t).dot(self.w_1v.conj().like(self.acc_t)).like(self.acc_t) + xh_block.like(self.acc_t).dot(self.w_1h.conj().like(self.acc_t)).like(self.acc_t)
        y2 = xv_block.like(self.acc_t).dot(self.w_2v.conj().like(self.acc_t)).like(self.acc_t) + xh_block.like(self.acc_t).dot(self.w_2h.conj().like(self.acc_t)).like(self.acc_t)
        return y1.like(self.acc_t), y2.like(self.acc_t)

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
            #print(f'y1: {y1}, y2: {y2}, e1: {e1}, e2: {e2}')
            #print(f'w_1v: {self.w_1v}')
            #print(f'w_1h: {self.w_1h}')
            #print(f'w_2v: {self.w_2v}')
            #print(f'w_2h: {self.w_2h}')
            y1_out.append(y1)
            if e1_converged:
                y2_out.append(y2)
            if k > 100 and not e1_converged:
                e1_converged = True
                self.w_2v = self.w_1v.conj()[::-1].like(self.acc_t)
                self.w_2h = -self.w_1h.conj()[::-1].like(self.acc_t)

        if e1_converged is False:
            raise Warning("Equalizer did not converge")

        print(len(y1_out), len(y2_out))
                
        return Fxp(y1_out).like(self.acc_t), Fxp(y2_out).like(self.acc_t)

    def bits_per_symbol_CMA(self) -> float:
        """ Estimate the the bit operations needed to equalize a symbol using CMA """
        adds = 0
        mults = 0

        # ops in equalization
        adds += self.num_taps # complex additions
        mults += 2 * self.num_taps #complex multiplications

        # ops in updating weights
        adds += 2 * (self.num_taps + 1)
        mults += 3

        # convert adds to bit operations
        add_bits = adds * 2 * 0.5 * 5 * self.acc_t.n_word # 0.5 switching probability, 5 gates in full adder, x2 for complex addition
        add_bits += mults * 5 * 0.5 * 5 * self.acc_t.n_word # 5 gates in full adder, x5 for complex multiplication
        mult_bits = mults * 3 * 0.5 * 6 * self.acc_t.n_word**2

        total_bits = add_bits + mult_bits
        return total_bits
        


