# learn topics
import threading
import time
from colorsys import hsv_to_rgb
from math import pi, e
from multiprocessing import cpu_count
from multiprocessing.dummy import Pool as ThreadPool

import matplotlib.pyplot as plt
import numpy as np


class Learn:
    # w, h = 1920, 1080
    # vect = np.zeros((w, h), dtype=np.complex)
    # nCores = cpu_count()
    # c0 = 2 + 3j

    def __init__(self, w, h):
        # self.w, self.h = w, h
        pass

    def _zExpression(self, c0):
        return c0 * c0

    def worker(self, l):  # reentrant as all vars are local
        # f, t = int(count * self.w / nCores), int((count + 1) * self.w / nCores)
        # self.vect[f:t] = np.asarray([[self._zExpression(self.c0) for i in range(0, self.h)] for c in range(f, t)])

        l.z = 0
        for l.i in range(0, 1920):
            for l.j in range(1080):
                l.z = (l.i + l.j) * (l.i + l.j)

    def threadOpt(self):
        # self.vect = np.zeros((self.w, self.h), dtype=np.complex)
        print('starting threads')
        self.l = [threading.local() for i in range(5)]
        st = time.time()
        threads = [threading.Thread(target=self.worker, args=(self.l[i],)) for i in range(5)]
        for t in threads: t.start()
        for t in threads: t.join()

        st = time.time() - st
        print('finished threads, time MT: %f' % st)

    def seqOpt(self):
        print('starting sequential')
        # self.vectTh = self.vect
        # self.vect = np.zeros((self.w, self.h), dtype=np.complex)

        st = time.time()
        for i in range(5): self.worker(self.l[i])
        st = time.time() - st
        print('time ST:%f' % st)

    def directOpt(self):
        # self.vect = np.zeros((self.w, self.h), dtype=np.complex)
        st = time.time()
        # self.vect = np.asarray([[self._zExpression(self.c0) for _ in range(0, self.h)] for _ in range(0, self.w)])
        st = time.time() - st
        print('direct calc: time ST:%f' % st)


class DomainColoring:
    def generateColors(self, h, w, deltaW=1):
        def pow3(x):
            return x * x * x

        pi2 = pi * 2
        limit = pi

        rmi, rma, imi, ima = -limit, limit, -limit, limit
        colors = np.zeros((w, h, 3), dtype=np.float32)

        for j in range(0, h, deltaW):
            im = ima - (ima - imi) * j / (h - 1)
            for i in range(0, w):
                re = rma - (rma - rmi) * i / (w - 1)

                v = _zExpression(complex(re, im))

                hue = v.real
                while hue < 0: hue += pi2
                hue /= pi2
                m, _ranges, _rangee = abs(v), 0, 1
                while m > _rangee:
                    _ranges = _rangee
                    _rangee *= e
                k = (m - _ranges) / (_rangee - _ranges)
                kk = k * 2 if k < 0.5 else 1 - (k - 0.5) * 2
                sat = 0.4 + (1 - pow3((1 - (kk)))) * 0.6
                val = 0.6 + (1 - pow3((1 - (1 - kk)))) * 0.4

                colors[i, j] = hsv_to_rgb(hue, sat, val)
        return colors


def test02():
    predefFuncs = ['acos((1+1j)*log(sin(z**3-1)/z))',
                   '(1+1j)*log(sin(z**3-1)/z)',
                   '(1+1j)*sin(z)',
                   'z + z**2/sin(z**4-1)',
                   'log(sin(z))',
                   'cos(z)/(sin(z**4-1)',
                   'z**6-1',
                   '(z**2-1) * (z-2-1j)**2 / (z**2+2*1j)',
                   'sin(z)*(1+2j)',
                   'sin(1/z)',
                   'sin(z)*sin(1/z)',
                   '1/sin(1/sin(z))',
                   'z',
                   '(z**2+1)/(z**2-1)',
                   '(z**2+1)/z',
                   '(z+3)*(z+1)**2',
                   '(z/2)**2*(z+1-2j)*(z+2+2j)/z**3',
                   '(z**2)-0.75-(0.2*j)']

    # compile/exec & check errors expressions
    def _z(z: complex) -> complex:
        return z * z

    _zFunc = predefFuncs[11]
    from cmath import sin, cos, acos, asin, tan, atan, log, log10
    exec(compile(source='''from cmath import sin, cos, acos, asin, tan, atan, log, log10
    
def _zExpression(z): return %s''' % _zFunc, filename='', mode='exec'))  # define _z function

#     code = compile('''from cmath import sin, cos, acos, asin, tan, atan, log, log10
# def _zExpression(z): return %s''' % predefFuncs[1], '<float>', 'exec')
#     exec(code)
    try:
        c0 = 1 + 1j
        _zExpression(c0)
    except:
        print('syntax error in function')
        exit(1)

    plt.imshow(DomainColoring().generateColors(400, 300))
    plt.show()


def test01():
    # threads
    nCores = cpu_count()
    c = hsv_to_rgb(1, 1, 1)

    l = Learn(1920, 1080)

    l.threadOpt()
    l.seqOpt()
    l.directOpt()


def testMT():
    def foo(x):
        return x + x

    a = np.arange(20000000)
    st = time.time()
    with ThreadPool(cpu_count()) as pool:
        results = pool.map(foo, a)
    lMT = time.time() - st

    st = time.time()
    results = [foo(item) for item in a]
    lST = time.time() - st
    print('lap mt:', lMT, 'lap st:', lST)


if __name__ == '__main__':
    predefFuncs = ['acos((1+1j)*log(sin(z**3-1)/z))',
                   '(1+1j)*log(sin(z**3-1)/z)',
                   '(1+1j)*sin(z)',
                   'z + z**2/sin(z**4-1)',
                   'log(sin(z))',
                   'cos(z)/(sin(z**4-1)',
                   'z**6-1',
                   '(z**2-1) * (z-2-1j)**2 / (z**2+2*1j)',
                   'sin(z)*(1+2j)',
                   'sin(1/z)',
                   'sin(z)*sin(1/z)',
                   '1/sin(1/sin(z))',
                   'z',
                   '(z**2+1)/(z**2-1)',
                   '(z**2+1)/z',
                   '(z+3)*(z+1)**2',
                   '(z/2)**2*(z+1-2j)*(z+2+2j)/z**3',
                   '(z**2)-0.75-(0.2*j)']


    # compile/exec & check errors expressions
    def _z(z: complex) -> complex:
        return z * z


    _zFunc = predefFuncs[11]
    from cmath import sin, cos, acos, asin, tan, atan, log, log10

    exec(compile(source='''from cmath import sin, cos, acos, asin, tan, atan, log, log10

def _zExpression(z): return %s''' % _zFunc, filename='', mode='exec'))  # define _z function

    #     code = compile('''from cmath import sin, cos, acos, asin, tan, atan, log, log10
    # def _zExpression(z): return %s''' % predefFuncs[1], '<float>', 'exec')
    #     exec(code)
    try:
        c0 = 1 + 1j
        _zExpression(c0)
    except:
        print('syntax error in function')
        exit(1)

    plt.imshow(DomainColoring().generateColors(400, 300))
    plt.show()