from colorsys import hsv_to_rgb
from math import sin, cos, pi, fmod, e
from multiprocessing import Pool, cpu_count
from time import time

import matplotlib.pyplot as plt
import numpy as np


def foo(x):
    return x * 2 + x / 2 + x * 0.23 + x / 90.12 + x + sin(x / 100 + pi) * cos(x * 2)


def fooxi(xi):
    x, i = xi[0], xi[1]
    return x + i + 2 * sin(x + i)


def test01():
    n = 50000000
    a = [i for i in range(n)]

    st = time()
    with Pool(cpu_count()) as pool:
        mtl = pool.map(foo, a)
        pool.close()
        pool.join()
    smt = sum(mtl)
    lMT = time() - st

    st = time()
    stl = map(foo, a)
    sst = sum(stl)
    lST = time() - st

    print('mt:', lMT, 'st:', lST, smt == sst)


def test02():
    a = [i for i in range(1000000)]

    st = time()
    with Pool(cpu_count()) as pool:
        b = pool.map(fooxi, enumerate(a))
        pool.close()
        pool.join()
    lMT = time() - st

    smt = sum(list(b))

    st = time()
    a = list(map(fooxi, enumerate(a)))
    lST = time() - st

    sst = sum(a)

    print('mt:', lMT, 'st:', lST, smt == sst)


def test03():  # numpy arrays map much slower than lists
    w, h = 800, 600
    wh = w * h

    img = np.asarray(range(wh), dtype=np.float32)

    ts = time()
    imgProcST = np.asarray(list(map(foo, img)))
    ts = time() - ts

    tm = time()
    with Pool(cpu_count() // 2) as pool:
        imgProcMT = np.asarray(list(pool.map(foo, img)))
        pool.close()
        pool.join()
    tm = time() - tm

    print(np.all(imgProcMT == imgProcST), ts, tm, ts / tm)


class DomainColoring:
    pi2 = pi * 2
    limit = pi

    rmi, rma, imi, ima = -limit, limit, -limit, limit
    ima_imi = ima - imi
    rma_rmi = rma - rmi

    def __init__(self, _zExpression, w, h, hIni, wDelta):
        self._zExpression, self.w, self.h, self.hIni, self.wDelta = _zExpression, w, h, hIni, wDelta

    def imag(self, j):
        return self.ima - (self.ima_imi) * j / (self.w - 1)

    def real(self, i):
        return self.rma - (self.rma_rmi) * i / (self.h - 1)

    def eval(self, x):  # x contains index
        def pow3(x):
            return x * x * x

        try:
            i, j = x // self.w, x % self.w
            z = self._zExpression(complex(self.real(i), self.imag(j)))

            hue = (self.pi2 - fmod(abs(z.real), self.pi2)) / self.pi2

            # _minRange=exp(int(log(m))) _maxRange=_minRange/e
            m, _minRange, _maxRange = abs(z), 0, 1
            while m > _maxRange:
                _minRange = _maxRange
                _maxRange *= e

            k = (m - _minRange) / (_maxRange - _minRange)
            kk = k * 2 if k < 0.5 else 1 - (k - 0.5) * 2
            sat = 0.4 + (1 - (1 - (kk)) ** 3) * 0.6
            val = 0.6 + (1 - (1 - (1 - kk)) ** 3) * 0.4

            return hsv_to_rgb(hue, sat, val)
        except:
            return (0., 0., 0.)


def test04():  # list map much faster t/numpy's
    def __z(z):
        return sin(1 / z)

    w, h = 400, 300
    wh = w * h
    ev = DomainColoring(_zExpression=__z, w=w, h=h, hIni=0, wDelta=1)
    func = ev.eval
    # func=foo

    img = list(map(float, range(wh)))  # generate list | img[index]=float(index)

    ts = time()
    imgProcST = list(map(func, img))
    ts = time() - ts

    tm = time()
    with Pool(cpu_count()) as pool:
        imgProcMT = pool.map(func, img, w)
        pool.close()
        pool.join()
    tm = time() - tm

    print('ok=', imgProcMT == imgProcST, 'single=', ts, 'multi=', tm, 'ratio=', ts / tm)

    fig = plt.figure(predefFuncs[0])
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)

    plt.imshow(imgProcMT)
    plt.show()


def testDomCol01(sfunc):
    w, h = 1920, 1080
    wh = w * h
    ev = DomainColoring(_zExpression=_z, w=w, h=h, hIni=0, wDelta=1)
    func = ev.eval

    t = time()
    img = list(map(float, range(wh)))  # generate list | img[index]=float(index)

    with Pool(cpu_count()) as pool:
        img = pool.map(func, img, w)
        pool.close()
        pool.join()

    print('lap:', time() - t)
    fig = plt.figure(predefFuncs[0])
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)

    plt.imshow(np.reshape(img, (h, w, 3)))
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
    exec(compile('''from cmath import sin, cos, acos, asin, tan, atan, log, log10
def _z(z): return %s''' % predefFuncs[4], '<float>', 'exec'))

    testDomCol01(predefFuncs[4])
