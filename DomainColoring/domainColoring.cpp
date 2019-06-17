/*
    Domain coloring python interface
    usage:

from DomainColoring import DomainColoring

import matplotlib.pyplot as plt

w, h = 1920, 1080
dc = DomainColoring(w, h, 'z^5 + sin(z)^8')

print('z compiler error for formula:', dc.formula, '->', dc.error)

if not dc.error:
    print('domain coloring', dc.w, 'x', dc.h, 'image, generation time:', time()-lap, 'secs')

    plt.imshow( dc.getimage_np() )
    plt.show()


*/

#include <boost/python.hpp>
#include <boost/python/numpy.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>

#include <complex>
#include <cmath>

#include "Thread.h"
#include "zCompiler.h"

#include <Accelerate.h>

namespace p = boost::python;
namespace np = boost::python::numpy;

typedef unsigned char byte;
typedef std::complex<float>ComplexFloat;
using std::string;

class DomainColoring {
public:
    DomainColoring(int w, int h, string formula) : w(w), h(h), szBytes(w*h*3), formula(formula) {
        Py_Initialize(); // init boost & numpy boost
        np::initialize();

        compErr = zComp.compile(formula);
        image = new byte[szBytes];

        if(! compErr)  generate();
    }

    ~DomainColoring() {
        delete[]image;
    }

    np::ndarray getimage_np() { // return numpy array to direct plot, image->numpy
        return np::from_data(image,             // data -> image
            np::dtype::get_builtin<byte>(),     // dtype -> byte
            p::make_tuple(h, w, 3),             // shape -> h x w x 3
            p::make_tuple(w*3, 3, 1), own);     // stride in bytes [1,1,1] (3) each row = w x 3
    }

    void testAccelerate(int n) {
        float *array1=new float[n];
        float value = 42.195;
        vDSP_vfill(&value, array1, 1, n);
        delete[]array1;
    }

    float inline sqr(float x) {return x*x;}
    void testFFTAccelerate() {

        int log2n=20, n=1<<log2n, n2=n/2;

        float*inputdata=new float[n];
        for (int i=0; i<n; i++) inputdata[i]=(float)rand()/RAND_MAX;

        COMPLEX_SPLIT cxfft;
        cxfft.realp = new float[n2];
        cxfft.imagp = new float[n2];

        // prepare the fft algo (you want to reuse the setup across fft calculations)
        FFTSetup setup = vDSP_create_fftsetup(log2n, kFFTRadix2);

        vDSP_ctoz((COMPLEX *)inputdata, 2, &cxfft, 1, n2); // copy the input to the packed complex array

        vDSP_fft_zrip(setup, &cxfft, 1, log2n, FFT_FORWARD); // calculate the fft

        float *fftdata=new float[n2];
        for (int i = 0; i < n2; ++i) // return modulus
            fftdata[i] = sqrtf(sqr(cxfft.realp[i]) + sqr(cxfft.imagp[i]));

        vDSP_destroy_fftsetup(setup); // release resources

        delete[]cxfft.realp; delete[]cxfft.imagp;
        delete[]fftdata;
    }

private:

    inline float pow3(float x) { return x*x*x; }

    void generate() { // multitheaded version
        float PI2 = M_PI * 2;

        float limit = M_PI;
        float rmi, rma, imi, ima;
        rmi = -limit; rma = limit; imi = -limit; ima = limit;

        Thread(h).run([this, ima, imi, rma, rmi, PI2](int t, int from, int to) {
            for (int j=from; j<to; j++) {
                float im = ima - (ima - imi) * j / (h - 1);

                for (int i = 0, index=j*w*3; i < w; i++, index+=3) {
                    float re = rma - (rma - rmi) * i / (w - 1);

                    ComplexFloat v = zComp.execute(ComplexFloat(re, im)); // fun(c); // evaluate here

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

public:
    int w,h, szBytes;
    byte* image;
    string formula;
    zCompiler zComp;
    bool compErr;
    p::object own;
};


// python interface

BOOST_PYTHON_MODULE(DomainColoring)
{
    p::class_<DomainColoring>("DomainColoring", p::init<int, int, string>())
        .def_readwrite("w",  &DomainColoring::w)
        .def_readwrite("h",  &DomainColoring::h)
        .def_readwrite("formula",  &DomainColoring::formula)
        .def_readwrite("error",  &DomainColoring::compErr)

        .def("getimage_np",  &DomainColoring::getimage_np);
    ;
}