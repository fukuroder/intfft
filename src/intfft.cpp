#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/complex.h>
namespace py = pybind11;

static const double PI = 3.14159265358979323846;

/*
inline std::complex<double> lift_(const std::complex<double>& x, const std::complex<double>& w)
{
    if(w.imag() == 0.0){
        return x;
    }

    const double c = w.real();
    const double s = w.imag();
    double xr = x.real();
    double xi = x.imag();
    
    if (c >= 0.0){ // (-0.5pi, 0.5pi)
        xr += static_cast<int>(xi*(c-1)/s);
        xi += static_cast<int>(xr*s);
        xr += static_cast<int>(xi*(c-1)/s);
    }
    else{ // (0.5pi, 1.5pi)
        xr += static_cast<int>(xi*(c+1)/s);
        xi += static_cast<int>(xr*(-s));
        xr += static_cast<int>(xi*(c+1)/s);
        xr = -xr; xi = -xi; 
    }
    return std::complex<double>(xr, xi);
}

inline std::complex<double> ilift_(const std::complex<double>& x, const std::complex<double>& w)
{
    if(w.imag() == 0.0){
        return x;
    }

    const double c = w.real();
    const double s = w.imag();
    double xr = x.real();
    double xi = x.imag();
    
    if (c >= 0.0){ // (-0.5pi, 0.5pi)
        xr -= static_cast<int>(xi*(c-1)/s);
        xi -= static_cast<int>(xr*s);
        xr -= static_cast<int>(xi*(c-1)/s);
    }
    else{ // (0.5pi, 1.5pi)
        xr = -xr; xi = -xi; 
        xr -= static_cast<int>(xi*(c+1)/s);
        xi -= static_cast<int>(xr*(-s));
        xr -= static_cast<int>(xi*(c+1)/s);
    }
    return std::complex<double>(xr, xi);
}
*/

inline std::complex<double> lift_(const std::complex<double>& x, const std::complex<double>& w)
{
    if(w.imag() == 0.0){
        return x;
    }

    const double c = w.real();
    const double s = w.imag();
    double xr = x.real();
    double xi = x.imag();
    
    if(s > c){
        if (s > -c) {// (0.25pi, 0.75pi)
            const double t = xr; xr = xi; xi = t;
            xr += static_cast<int>(xi*(s-1)/c);
            xi += static_cast<int>(xr*c);
            xr += static_cast<int>(xi*(s-1)/c);
            xr = -xr;
        }
        else{ // (0.75pi, 1.25pi)
            xi = -xi;
            xr += static_cast<int>(xi*(-c-1)/s);
            xi += static_cast<int>(xr*s);
            xr += static_cast<int>(xi*(-c-1)/s);
            xr = -xr;
        }
    }
    else{
        if (s < -c){ // (-0.75pi, -0.25pi)
            xr += static_cast<int>(xi*(-s-1)/c);
            xi += static_cast<int>(xr*c);
            xr += static_cast<int>(xi*(-s-1)/c);
            const double t = xr; xr = xi; xi = -t;
        }
        else{ // (-0.25pi, 0.25pi)
            xr += static_cast<int>(xi*(c-1)/s);
            xi += static_cast<int>(xr*s);
            xr += static_cast<int>(xi*(c-1)/s);
        }
    }
    return std::complex<double>(xr, xi);
}

inline std::complex<double> ilift_(const std::complex<double>& x, const std::complex<double>& w)
{
    if(w.imag() == 0.0){
        return x;
    }
    
    const double c = w.real();
    const double s = w.imag();
    double xr = x.real();
    double xi = x.imag();

    if(s > c){
        if(s > -c){ // (0.25pi, 0.75pi)
            xr = -xr;
            xr -= static_cast<int>(xi*(s-1)/c);
            xi -= static_cast<int>(xr*c);
            xr -= static_cast<int>(xi*(s-1)/c);
            const double t = xr; xr = xi, xi = t;
        }
        else{ // (0.75pi, 1.25pi)
            xr = -xr;
            xr -= static_cast<int>(xi*(-c-1)/s);
            xi -= static_cast<int>(xr*s);
            xr -= static_cast<int>(xi*(-c-1)/s);
            xi = -xi;
        }
    }
    else{
        if(s < -c){ // (-0.75pi, -0.25pi)
            const double t = xr; xr = -xi; xi = t;
            xr -= static_cast<int>(xi*(-s-1)/c);
            xi -= static_cast<int>(xr*c);
            xr -= static_cast<int>(xi*(-s-1)/c);
        }
        else{ // (-0.25pi, 0.25pi)
            xr -= static_cast<int>(xi*(c-1)/s);
            xi -= static_cast<int>(xr*s);
            xr -= static_cast<int>(xi*(c-1)/s);
        }
    }
    return std::complex<double>(xr, xi);
}

// Created by referring to http://www.kurims.kyoto-u.ac.jp/~ooura/fftman/ftmn1_24.html#sec1_2_4
void fft_(int n, std::complex<double>* a)
{   
    using namespace std::literals::complex_literals;

    // L shaped butterflies
    for (int m = n; m > 2; m >>= 1) {
        const double theta = -2 * PI / m;
        const int mq = m >> 2;
        for (int i = 0; i < mq; i++) {
            const std::complex<double> w1 = std::polar(1.0, theta * i);
            const std::complex<double> w3 = std::polar(1.0, theta * 3 * i);
            for (int k = m; k <= n; k <<= 2) {
                for (int j0 = k - m + i; j0 < n; j0 += 2 * k) {
                    const int j1 = j0 + mq;
                    const int j2 = j1 + mq;
                    const int j3 = j2 + mq;
                    const std::complex<double> x1 = a[j0] - a[j2];
                    a[j0] += a[j2];
                    const std::complex<double> x3 = a[j1] - a[j3];
                    a[j1] += a[j3];
                    a[j2] = lift_(x1 - x3*1.0i, w1); //a[j2] = (x1 - x3*1.0i) * w1;
                    a[j3] = lift_(x1 + x3*1.0i, w3); //a[j3] = (x1 + x3*1.0i) * w3;
                }
            }
        }

    }

    // radix 2 butterflies
    for (int k = 2; k <= n; k <<= 2) {
        for (int j = k - 2; j < n; j += 2 * k) {
            const std::complex<double> x0 = a[j] - a[j + 1];
            a[j] += a[j + 1];
            a[j + 1] = x0;
        }
    }
    // unscrambler
    for (int i = 0, j = 1; j < n - 1; j++) {
        for (int k = n >> 1; k > (i ^= k); k >>= 1);
        if (j < i) {
            const std::complex<double> x0 = a[j];
            a[j] = a[i];
            a[i] = x0;
        }
    }
}

// Created by referring to http://www.kurims.kyoto-u.ac.jp/~ooura/fftman/ftmn1_24.html#sec1_2_4
void ifft_(int n, std::complex<double>* a)
{
    using namespace std::literals::complex_literals;

    // scrambler
    for (int i = 0, j = 1; j < n - 1; j++) {
        for (int k = n >> 1; k > (i ^= k); k >>= 1);
        if (j < i) {
            const std::complex<double> x0 = a[j];
            a[j] = a[i];
            a[i] = x0;
        }
    }

    // radix 2 butterflies
    for (int k = 2; k <= n; k <<= 2) {
        for (int j = k - 2; j < n; j += 2 * k) {
            const std::complex<double> x0 = a[j];
            a[j] = (x0 + a[j + 1]) / 2.0;
            a[j + 1] = x0 - a[j];
        }
    }

    // L shaped butterflies
    for (int m = 4; m <= n; m <<= 1) {
        const double theta =  - 2 * PI / m;
        const int mq = m >> 2;
        for (int i = 0; i < mq; i++) {
            const std::complex<double> w1 = std::polar(1.0, theta * i);
            const std::complex<double> w3 = std::polar(1.0, theta * 3 * i);
            for (int k = m; k <= n; k <<= 2) {
                for (int j0 = k - m + i; j0 < n; j0 += 2 * k) {
                    const int j1 = j0 + mq;
                    const int j2 = j1 + mq;
                    const int j3 = j2 + mq;

                    const std::complex<double> x0 = a[j0];
                    const std::complex<double> x1 = a[j1];
                    const std::complex<double> x2 = ilift_(a[j2], w1);
                    const std::complex<double> x3 = ilift_(a[j3], w3);
                    const std::complex<double> x2_ = (x2 + x3)/2.0;
                    const std::complex<double> x3_ = (x2 - x2_)*1.0i;
                    a[j0] = (x0 + x2_)/2.0;
                    a[j1] = (x1 + x3_)/2.0;
                    a[j2] = (x0 - a[j0]);
                    a[j3] = (x1 - a[j1]);
                }
            }
        }
    }
}

py::array_t<std::complex<double>> fft(py::array_t<std::complex<double>> x) {
    fft_(static_cast<int>(x.shape(0)), static_cast<std::complex<double>*>(x.request().ptr));
    return x;
}

py::array_t<std::complex<double>> ifft(py::array_t<std::complex<double>> x) {
    ifft_(static_cast<int>(x.shape(0)), static_cast<std::complex<double>*>(x.request().ptr));
    return x;
}

PYBIND11_MODULE(intfft, m) {
    m.doc() = "Integer Fast Fourier Transform in Python";
    m.def("fft", &fft, "");
    m.def("ifft", &ifft, "");
}