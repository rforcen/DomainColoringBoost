from numba import jit, jitclass, double, int32, int64, float32, types, complex64, uint8
from math import pi, e
from cmath import sin, cos, exp, log
import numpy as np
from timeit import default_timer as time
import matplotlib.pyplot as plt

DCDefs = [
    ('w', int32),
    ('h', int32),
    ('rmi', float32),
    ('rma', float32),
    ('imi', float32),
    ('ima', float32),
    ('ima_imi', float32),
    ('rma_rmi', float32),
    ('pi2', float32)
]


@jitclass(DCDefs)
class DomainColoring:
    def __init__(self, w, h):
        self.w = w
        self.h = h

        limit = pi
        self.rmi, self.rma, self.imi, self.ima = -limit, limit, -limit, limit

        self.ima_imi = self.ima - self.imi
        self.rma_rmi = self.rma - self.rmi

        self.pi2 = pi * 2

    def _zExpression(self, z):
        # if z == 0:
        #     return 0
        # else:
        return z * z * z * z * sin(z)  # sin(z) * sin(1 / z)

    def generate(self):
        def eval(x):  # x contains index

            def hsv_to_rgb(h, s, v):  # rgb in 0..1 range
                if s == 0:
                    return v, v, v

                i = int(h * 6)  # XXX assume int() truncates!
                f = (h * 6) - i
                p = v * (1 - s)
                q = v * (1 - s * f)
                t = v * (1 - s * (1 - f))
                i = i % 6

                if i == 0: return v, t, p
                if i == 1: return q, v, p
                if i == 2: return p, v, t
                if i == 3: return p, q, v
                if i == 4: return t, p, v
                if i == 5: return v, p, q

            def fmod(x, y):  # math.fmod not supported
                return x - int32(x / y) * y

            def pow3(x):
                return x * x * x

            def convByte(c):  # from c=[0,1] to 0..255
                return 255 if 1 <= c <= 0 else 255 * c

            def calcZinput(i, j):
                def imag(j):
                    return self.ima - (self.ima_imi) * j / (self.w - 1)

                def real(i):
                    return self.rma - (self.rma_rmi) * i / (self.h - 1)

                return complex(real(i), imag(j))

            i, j = x // self.w, x % self.w  # index to i,j coords

            z = self._zExpression(calcZinput(i, j))

            m, _minRange, _maxRange = abs(z), 0, 1
            while m > _maxRange:
                _minRange = _maxRange
                _maxRange *= e

            k = (m - _minRange) / (_maxRange - _minRange)
            kk = k * 2 if k < 0.5 else 1 - (k - 0.5) * 2

            hue = (self.pi2 - fmod(abs(z.real), self.pi2)) / self.pi2
            sat = 0.4 + (1 - pow3(1 - kk)) * 0.6
            val = 0.6 + (1 - pow3(1 - (1 - kk))) * 0.4

            r, g, b = hsv_to_rgb(hue, sat, val)

            return convByte(r), convByte(g), convByte(b)

        image = np.zeros(shape=(self.h * self.w, 3), dtype=np.uint8)  # faster than list generation
        for i in range(self.w * self.h):
            image[i] = eval(int32(i), self.w)

        return image


def showImage(image, w, h):
    fig = plt.figure('')
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)

    plt.imshow(image.reshape(h, w, 3))
    plt.show()


def test_class():
    mf = 2
    w, h = mf * 1920, mf * 1080
    dc = DomainColoring(w, h)

    lap = time()

    image = dc.generate()

    print('generated', w, 'x', h, '=', w * h, 'pixels, in', time() - lap, 'secs')

    showImage(image, w, h)


if __name__ == '__main__':
    test_class()
