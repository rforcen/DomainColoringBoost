/*
    Domain coloring python interface
    usage:

import numpy as np
from DomainColoring import DomainColoring
import matplotlib.pyplot as plt
from color import Color

w, h = 1920, 1080
dc = DomainColoring(w, h)

plt.imshow(dc.get_image_np())
plt.show()


*/

#include <boost/python.hpp>
#include <boost/python/numpy.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>

#include <complex>
#include <cmath>

#include "Thread.h"

namespace p = boost::python;
namespace np = boost::python::numpy;

typedef unsigned char byte;
typedef std::complex<float>ComplexFloat;

class DomainColoring {
public:
    DomainColoring(int w, int h) : w(w), h(h), szBytes(w*h*3) {
        Py_Initialize(); // init boost & numpy boost
        np::initialize();

        image = new byte[szBytes];
        generateMT();
    }
    ~DomainColoring() {
        delete[]image;
    }

    np::ndarray get_image_np() { // return numpy array to direct plot, image->numpy
        p::tuple shape = p::make_tuple(h, w, 3); // w x h x 3
        np::dtype dtype = np::dtype::get_builtin<byte>(); // of bytes

        np::ndarray result = np::zeros(shape, dtype); // np array

        memcpy(result.get_data(), image, szBytes); // copy data from image
        return result;
    }

private:

    inline ComplexFloat zFunc(ComplexFloat z) { // the complex func to evaluate
        return z*z*z*z*sin(z);
    }

    inline float pow3(float x) { return x*x*x; }

    void generate() {
        float PI = M_PI, PI2 = PI * 2;
        float E = M_E;

        float limit = PI;
        float rmi, rma, imi, ima;
        rmi = -limit; rma = limit; imi = -limit; ima = limit;
        int icol = 0; // image index

        try {

            for (int j = 0; j < h; j++) {
                float im = ima - (ima - imi) * j / (h - 1);

                for (int i = 0; i < w; i++, icol+=3) { // next RGB +3
                    float re = rma - (rma - rmi) * i / (w - 1);

                    ComplexFloat v = zFunc(ComplexFloat(re, im)); // fun(c); // evaluate here

                    float hue = arg(v);
                    while (hue < 0) hue += PI2;
                    hue /= PI2;

                    float m = abs(v), ranges = 0, rangee = 1;
                    while (m > rangee) {
                        ranges = rangee;
                        rangee *= E;
                    }

                    float k = (m - ranges) / (rangee - ranges);
                    float kk = (k < 0.5 ? k * 2 : 1 - (k - 0.5) * 2);

                    float sat = 0.4 + (1 - pow3(1 - kk))     * 0.6;
                    float val = 0.6 + (1 - pow3(1 - (1 - kk))) * 0.4;

                    setColorRGB(icol, hue, sat, val);
                }
            }
        }
        catch (...) {
        }
    }


    void generateMT() { // multitheaded version
        float PI2 = M_PI * 2;

        float limit = M_PI;
        float rmi, rma, imi, ima;
        rmi = -limit; rma = limit; imi = -limit; ima = limit;

        Thread(h).run([this, ima, imi, rma, rmi, PI2](int t, int from, int to) {
            for (int j=from; j<to; j++) {
                float im = ima - (ima - imi) * j / (h - 1);

                for (int i = 0, index=j*w*3; i < w; i++, index+=3) {
                    float re = rma - (rma - rmi) * i / (w - 1);

                    ComplexFloat v = zFunc(ComplexFloat(re, im)); // fun(c); // evaluate here

                    float hue = arg(v); // calc hue
                    while (hue < 0) hue += PI2;
                    hue /= PI2;

                    float m = abs(v), ranges = 0, rangee = 1;
                    while (m > rangee) {
                        ranges = rangee;
                        rangee *= M_E;
                    }

                    float k = (m - ranges) / (rangee - ranges);
                    float kk = (k < 0.5 ? k * 2 : 1 - (k - 0.5) * 2);

                    float sat = 0.4 + (1 - pow3(1 - kk))     * 0.6;
                    float val = 0.6 + (1 - pow3(1 - (1 - kk))) * 0.4;

                    setColorRGB(index, hue, sat, val);
                }
            }
        });

    }
    void setColorRGB(int iCol, float h, float s, float v) { // convert hsv to int with alpha 0xff00000
        float r = 0, g = 0, b = 0;
        if (s == 0)
            r = g = b = v;
        else {
            if (h == 1)
                h = 0;
            float z = floor(h * 6);
            int i = (int)(z);
            float f = h * 6 - z,
            p = v * (1 - s),
            q = v * (1 - s * f),
            t = v * (1 - s * (1 - f));

            switch (i) {
                case 0:
                    r = v;
                    g = t;
                    b = p;
                    break;
                case 1:
                    r = q;
                    g = v;
                    b = p;
                    break;
                case 2:
                    r = p;
                    g = v;
                    b = t;
                    break;
                case 3:
                    r = p;
                    g = q;
                    b = v;
                    break;
                case 4:
                    r = t;
                    g = p;
                    b = v;
                    break;
                case 5:
                    r = v;
                    g = p;
                    b = q;
                    break;
            }
        }
        image[iCol+0]=r*255; // assign RGB to current index
        image[iCol+1]=g*255;
        image[iCol+2]=b*255;
    }

    int HSV2int(float h, float s, float v) { // convert hsv to int with alpha 0xff00000
            float r = 0, g = 0, b = 0;
            if (s == 0)
                r = g = b = v;
            else {
                if (h == 1)
                    h = 0;
                float z = floor(h * 6);
                int i = (int)(z);
                float f = h * 6 - z,
                p = v * (1 - s),
                q = v * (1 - s * f),
                t = v * (1 - s * (1 - f));

                switch (i) {
                    case 0:
                        r = v;
                        g = t;
                        b = p;
                        break;
                    case 1:
                        r = q;
                        g = v;
                        b = p;
                        break;
                    case 2:
                        r = p;
                        g = v;
                        b = t;
                        break;
                    case 3:
                        r = p;
                        g = q;
                        b = v;
                        break;
                    case 4:
                        r = t;
                        g = p;
                        b = v;
                        break;
                    case 5:
                        r = v;
                        g = p;
                        b = q;
                        break;
                }
            }
            int c, color = 0xff000000;
            // alpha = 0xff
            c = (int)(255. * r) & 0xff;
            color |= c;
            c = (int)(255. * g) & 0xff;
            color |= (c << 8);
            c = (int)(255. * b) & 0xff;
            color |= (c << 16);
            return color;
        }
        

public:
    int w,h, szBytes;
    byte* image;
};


// python interface

BOOST_PYTHON_MODULE(DomainColoring)
{
    p::class_<DomainColoring>("DomainColoring", p::init<int, int>())
        .def_readwrite("w", &DomainColoring::w)
        .def_readwrite("h", &DomainColoring::h)
        .def("get_image_np", &DomainColoring::get_image_np);
    ;
}