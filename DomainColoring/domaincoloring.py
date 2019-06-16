from colorsys import hsv_to_rgb
from math import pi, fmod, e
from multiprocessing import Pool, cpu_count

import matplotlib.pyplot as plt
import numpy as np

from timeit import default_timer as time

class DomainColoring:
    pi2 = pi * 2

    def __init__(self, _zExpression, w, h):
        self._zExpression, self.w, self.h = _zExpression, w, h
        self.set_limit(pi)

    def set_limit(self, limit):
        self.rmi, self.rma, self.imi, self.ima = -limit, limit, -limit, limit
        self.ima_imi = self.ima - self.imi
        self.rma_rmi = self.rma - self.rmi

    def imag(self, j):
        return self.ima - (self.ima_imi) * j / (self.w - 1)

    def real(self, i):
        return self.rma - (self.rma_rmi) * i / (self.h - 1)

    def eval(self, x):  # x contains index
        def pow3(x):
            return x * x * x

        try:
            i, j = x // self.w, x % self.w  # index to i,j coords

            z = self._zExpression(complex(self.real(i), self.imag(j)))

            hue = (self.pi2 - fmod(abs(z.real), self.pi2)) / self.pi2

            # _minRange=exp(int(log(m))) _maxRange=_minRange/e
            m, _minRange, _maxRange = abs(z), 0, 1
            while m > _maxRange:
                _minRange = _maxRange
                _maxRange *= e

            k = (m - _minRange) / (_maxRange - _minRange)
            kk = k * 2 if k < 0.5 else 1 - (k - 0.5) * 2
            sat = 0.4 + (1 - pow3(1 - (kk))) * 0.6
            val = 0.6 + (1 - pow3(1 - (1 - kk))) * 0.4

            return hsv_to_rgb(hue, sat, val)
        except:
            return 0., 0., 0.


def generateDomCol(w, h, _z):
    lap = time()

    func = DomainColoring(_zExpression=_z, w=w, h=h).eval
    img = list(map(float, range(w * h)))  # generate list | img[index]=float(index)

    with Pool(cpu_count()) as pool:  # map list
        img = pool.map(func, img, w)
        pool.close()
        pool.join()

    print('lap time:', time() - lap)

    return np.reshape(img, (h, w, 3))  # got a w*h,3 list -> reshape


def testDomCol(pdf, _z):
    fig = plt.figure(pdf)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)

    mf=2
    plt.imshow(generateDomCol(mf*1920, mf*1080, _z))
    plt.show()


if __name__ == '__main__':
    predefFuncs = ['sin(1/z)',
                   'acos((1+1j)*log(sin(z**3-1)/z))',
                   '(1+1j)*log(sin(z**3-1)/z)',
                   '(1+1j)*sin(z)',
                   'z + z**2/sin(z**4-1)',
                   'log(sin(z))',
                   'cos(z)/(sin(z**4-1))',
                   'z**6-1',
                   '(z**2-1) * (z-2-1j)**2 / (z**2+2*1j)',
                   'sin(z)*(1+2j)',
                   'sin(z)*sin(1/z)',
                   '1/sin(1/sin(z))',
                   'z',
                   '(z**2+1)/(z**2-1)',
                   '(z**2+1)/z',
                   '(z+3)*(z+1)**2',
                   '(z/2)**2*(z+1-2j)*(z+2+2j)/z**3',
                   '(z**2)-0.75-(0.2*(0+1j))']

    _zFunc = predefFuncs[11]
    exec(compile('''from cmath import sin, cos, acos, asin, tan, atan, log, log10
def _z(z): return %s''' % _zFunc, '<float>', 'exec'))  # define _z function


    testDomCol(_zFunc, _z)

