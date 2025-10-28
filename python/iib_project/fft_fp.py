import numpy as np
from fxpmath import Fxp
from typing import List
class FFT_fp:
    def __init__(self, M: int, DW_io: int, DW_acc: int):
        self.N = int(2**M)  # Number of FFT points
        self.io_t = Fxp(dtype=f'fxp-s{DW_io}/{DW_io-1}-complex')
        self.acc_t = Fxp(dtype=f'fxp-s{DW_acc}/{DW_acc - M - 1}-complex')

    def get_twiddle_factors(self, inverse: bool) -> List[Fxp]:
        sign = 1 if inverse else -1
        twiddles = [Fxp(np.exp(2j * sign * np.pi * k / self.N)).like(self.acc_t) for k in range(self.N // 2)]
        return twiddles

    def bit_reverse_indices(self) -> List[int]:
        n_bits = int(np.log2(self.N))
        indices = np.arange(self.N)
        reversed_indices = np.zeros(self.N, dtype=int)
        for i in range(self.N):
            b = '{:0{width}b}'.format(i, width=n_bits)
            reversed_indices[i] = int(b[::-1], 2)
        return reversed_indices

    def fft(self, x: List[Fxp], inverse: bool) -> List[Fxp]:
        if len(x) != self.N:
            raise ValueError(f"Input length {len(x)} does not match FFT size {self.N}")
        # initialize fixed-point vals
        X = [Fxp(val).like(self.acc_t) for val in x]
        indices = self.bit_reverse_indices()
        X = [X[i] for i in indices]
        twiddle_factors = self.get_twiddle_factors(inverse=inverse)

        stages = int(np.log2(self.N))

        size = 2
        while size <= self.N:
            halfsize = size // 2
            step = self.N // size
            for i in range(0, self.N, size):
                for j in range(halfsize):
                    index = j * step
                    t = Fxp(twiddle_factors[index] * X[i + j + halfsize]).like(self.acc_t)
                    X[i + j + halfsize] = Fxp(X[i + j] - t).like(self.acc_t)
                    X[i + j] = Fxp(X[i + j] + t).like(self.acc_t)
            size *= 2

        if inverse:
            for i in range(self.N):
                X[i] = Fxp(X[i] / self.N).like(self.io_t) 
 

        return X



