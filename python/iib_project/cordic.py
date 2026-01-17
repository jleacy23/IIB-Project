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

    def rotate(self, x: Fxp, y: Fxp, theta: Fxp) -> (Fxp, Fxp):
        """ cordic rotation, able to handle array inputs """
        assert x.shape == y.shape == theta.shape, f"Input shapes must match, got {x.shape}, {y.shape}, {theta.shape}"
        x = Fxp(x * self.K).like(self.acc_t)
        y = Fxp(y * self.K).like(self.acc_t)
        if np.sum(np.abs(theta) > np.pi) > 0:
            raise ValueError("Theta must be in the range [-pi, pi]")
        z = Fxp(theta).like(self.acc_t)

        # correct for angles beyond max
        mask = z < -np.pi/2
        if np.any(mask):
            x[mask], y[mask], z[mask] = Fxp(-x[mask]).like(self.acc_t), Fxp(-y[mask]).like(self.acc_t), Fxp(z[mask] + np.pi).like(self.acc_t)
        mask = z > np.pi/2
        if np.any(mask):
            x[mask], y[mask], z[mask] = Fxp(-x[mask]).like(self.acc_t), Fxp(-y[mask]).like(self.acc_t), Fxp(z[mask] - np.pi).like(self.acc_t)

        for i in range(self.iterations):
            dir = np.where(z < 0, -1, 1)
            x_new = Fxp(x - dir * (y >> i)).like(self.acc_t)
            y_new = Fxp(y + dir * (x >> i)).like(self.acc_t)
            z_new = Fxp(z - dir * self.atan_table[i]).like(self.acc_t)
            x, y, z = x_new, y_new, z_new

        return x, y

    def complex_rotate(self, z_in: Fxp, theta: Fxp) -> Fxp:
        x = Fxp(np.real(z_in)).like(self.acc_t)
        y = Fxp(np.imag(z_in)).like(self.acc_t)
        x_out, y_out = self.rotate(x, y, theta)

        z_out = x_out + 1j * y_out

        return Fxp(z_out).like(self.complex_t)

    def vectoring(self, x: Fxp, y: Fxp) -> (Fxp, Fxp):
        """ cordic vectoring, able to handle array inputs """
        assert x.shape == y.shape, f"Input shapes must match, got {x.shape}, {y.shape}"
        x = Fxp(x).like(self.acc_t)
        y = Fxp(y).like(self.acc_t)
        z = Fxp(np.zeros_like(x)).like(self.acc_t)

        # correct for negative x values
        mask = x < 0
        if np.any(mask):
            x[mask], y[mask], z[mask] = Fxp(-x[mask]).like(self.acc_t), Fxp(-y[mask]).like(self.acc_t), Fxp(np.pi * Fxp(np.sign(y[mask])).like(self.acc_t)).like(self.acc_t)


        for i in range(self.iterations):
            dir = np.where(y < 0, -1, 1)
            x_new = Fxp(x + dir * (y >> i)).like(self.acc_t)
            y_new = Fxp(y - dir * (x >> i)).like(self.acc_t)
            z_new = Fxp(z + dir * self.atan_table[i]).like(self.acc_t)
            x, y, z = x_new, y_new, z_new

        x = Fxp(x * self.K).like(self.acc_t)

        return x, z

    def complex_phase(self, z_in: Fxp) -> Fxp:
        x = Fxp(np.real(z_in)).like(self.acc_t)
        y = Fxp(np.imag(z_in)).like(self.acc_t)
        _, theta = self.vectoring(x, y)

        return Fxp(theta).like(self.acc_t)



