from numba import jit, jitclass, double, int32, int64, float32, types, complex64
from math import pi, e
from cmath import sin, cos, exp, log
import numpy as np
from timeit import default_timer as time
import matplotlib.pyplot as plt


c0 = complex(0, 0)
pi2 = pi * 2

# base z expression to plot
@jit(complex64(complex64))
def _zExpression(z):
    if z == c0:
        return c0
    else:
        return sin(z) * sin(1 / z)


@jit((float32[:, :])(int64, int64))  # image(w,h)
def generate(w, h):
    def eval(x, w):  # x contains index

        def hsv_to_rgb(h, s, v):
            if s == 0:
                return v, v, v

            i = int(h * 6)  # XXX assume int() truncates!
            f = (h * 6) - i
            p = v * (1 - s)
            q = v * (1 - s * f)
            t = v * (1 - s * (1 - f))
            i = i % 6

            if i == 0:
                return v, t, p
            if i == 1:
                return q, v, p
            if i == 2:
                return p, v, t
            if i == 3:
                return p, q, v
            if i == 4:
                return t, p, v
            if i == 5:
                return v, p, q

        def fmod(x, y):  # math.fmod not supported
            return x - int32(x / y) * y

        def pow3(x):
            return x * x * x

        limit = pi

        rmi, rma, imi, ima = -limit, limit, -limit, limit
        ima_imi = ima - imi
        rma_rmi = rma - rmi

        i, j = x // w, x % w  # index to i,j coords

        zin = complex(rma - rma_rmi * i / (h - 1), ima - ima_imi * j / (w - 1))

        z = _zExpression(zin)

        m, _minRange, _maxRange = abs(z), 0, 1
        while m > _maxRange:
            _minRange = _maxRange
            _maxRange *= e

        k = (m - _minRange) / (_maxRange - _minRange)
        kk = k * 2 if k < 0.5 else 1 - (k - 0.5) * 2

        hue = (pi2 - fmod(abs(z.real), pi2)) / pi2
        sat = 0.4 + (1 - pow3(1 - (kk))) * 0.6
        val = 0.6 + (1 - pow3(1 - (1 - kk))) * 0.4

        return hsv_to_rgb(hue, sat, val)

    image = np.zeros(shape=(h * w, 3), dtype=np.float32)  # faster than list generation
    for i in range(w * h):
        image[i] = eval(int32(i), w)
        # if np.any(image[i] > 1):
        #     print(i, image[i])

    return image


def test_generate():
    w, h = 1920, 1080
    lap = time()

    image = generate(w, h)

    print('generated', w, 'x', h, '=', w * h, 'pixels, in', time() - lap, 'secs')

test_generate()