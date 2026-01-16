import numpy as np
from fxpmath import Fxp

class Cordic:
    def __init__(self, iterations=16, word_length=16):
        self.iterations = iterations
        self.acc_t = Fxp(dtype=f'fxp-s{word_length}/{word_length-3}')
        self.complex_t = Fxp(dtype=f'fxp-s{word_length}/{word_length-3}-complex')
        self.K = Fxp(1/1.6468).like(self.acc_t)
        self.atan_table = self.gen_atan_table()

    def gen_atan_table(self) -> Fxp:
        atan_table = Fxp(np.zeros(self.iterations)).like(self.acc_t)
        for i in range(self.iterations):
            atan_table[i] = Fxp(np.arctan(2 ** -i)).like(self.acc_t)

        return atan_table

    def rotate(self, x: Fxp | float, y: Fxp | float, theta: Fxp | float) -> (Fxp, Fxp):
        x = Fxp(x * self.K).like(self.acc_t)
        y = Fxp(y * self.K).like(self.acc_t)
        if theta > np.pi or theta < -np.pi:
            raise ValueError("Theta must be in the range [-pi, pi]")
        z = Fxp(theta).like(self.acc_t)

        for i in range(self.iterations):
            if z >= 0:
                x_new = x - (y >> i)
                y_new = y + (x >> i)
                z = z - self.atan_table[i]
            else:
                x_new = x + (y >> i)
                y_new = y - (x >> i)
                z = z + self.atan_table[i]

            x, y = x_new, y_new

        return x, y

    def complex_rotate(self, z_in: Fxp, theta: Fxp) -> Fxp:
        x = Fxp(np.real(z_in)).like(self.acc_t)
        y = Fxp(np.imag(z_in)).like(self.acc_t)
        x_out, y_out = self.rotate(x, y, theta)

        z_out = x_out + 1j * y_out

        return Fxp(z_out).like(self.complex_t)


