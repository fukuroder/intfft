#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
namespace py = pybind11;
static const double PI = 3.14159265358979323846;

/*
std::tuple<int32_t, int32_t> lift_(int32_t xr, int32_t xi, double c, double s)
{
    if(s == 0.0){
        return {xr, xi};
    }
    
    if (c >= 0.0){ // (-0.5pi, 0.5pi)
        xr += static_cast<int32_t>(xi*(c-1)/s);
        xi += static_cast<int32_t>(xr*s);
        xr += static_cast<int32_t>(xi*(c-1)/s);
    }
    else{ // (0.5pi, 1.5pi)
        xr += static_cast<int32_t>(xi*(c+1)/s);
        xi += static_cast<int32_t>(xr*(-s));
        xr += static_cast<int32_t>(xi*(c+1)/s);
        xr = -xr; xi = -xi; 
    }
    return {xr, xi};
}

std::tuple<int32_t, int32_t> ilift_(int32_t xr, int32_t xi, double c, double s)
{
    if(s == 0.0){
        return {xr, xi};
    }
    
    if (c >= 0.0){ // (-0.5pi, 0.5pi)
        xr -= static_cast<int32_t>(xi*(c-1)/s);
        xi -= static_cast<int32_t>(xr*s);
        xr -= static_cast<int32_t>(xi*(c-1)/s);
    }
    else{ // (0.5pi, 1.5pi)
        xr = -xr; xi = -xi; 
        xr -= static_cast<int32_t>(xi*(c+1)/s);
        xi -= static_cast<int32_t>(xr*(-s));
        xr -= static_cast<int32_t>(xi*(c+1)/s);
    }
    return {xr, xi};
}
*/

std::tuple<int32_t, int32_t> lift_(int32_t xr, int32_t xi, double c, double s)
{
    if(s == 0.0){
        return {xr, xi};
    }
    
    if(s > c){
        if (s > -c) {// (0.25pi, 0.75pi)
            const int32_t t = xr; xr = xi; xi = t;
            xr += static_cast<int32_t>(xi*(s-1)/c);
            xi += static_cast<int32_t>(xr*c);
            xr += static_cast<int32_t>(xi*(s-1)/c);
            xr = -xr;
        }
        else{ // (0.75pi, 1.25pi)
            xi = -xi;
            xr += static_cast<int32_t>(xi*(-c-1)/s);
            xi += static_cast<int32_t>(xr*s);
            xr += static_cast<int32_t>(xi*(-c-1)/s);
            xr = -xr;
        }
    }
    else{
        if (s < -c){ // (-0.75pi, -0.25pi)
            xr += static_cast<int32_t>(xi*(-s-1)/c);
            xi += static_cast<int32_t>(xr*c);
            xr += static_cast<int32_t>(xi*(-s-1)/c);
            const int32_t t = xr; xr = xi; xi = -t;
        }
        else{ // (-0.25pi, 0.25pi)
            xr += static_cast<int32_t>(xi*(c-1)/s);
            xi += static_cast<int32_t>(xr*s);
            xr += static_cast<int32_t>(xi*(c-1)/s);
        }
    }
    return {xr, xi};
}

std::tuple<int32_t, int32_t> ilift_(int32_t xr, int32_t xi, double c, double s)
{
    if(s == 0.0){
        return {xr, xi};
    }

    if(s > c){
        if(s > -c){ // (0.25pi, 0.75pi)
            xr = -xr;
            xr -= static_cast<int32_t>(xi*(s-1)/c);
            xi -= static_cast<int32_t>(xr*c);
            xr -= static_cast<int32_t>(xi*(s-1)/c);
            const int32_t t = xr; xr = xi, xi = t;
        }
        else{ // (0.75pi, 1.25pi)
            xr = -xr;
            xr -= static_cast<int32_t>(xi*(-c-1)/s);
            xi -= static_cast<int32_t>(xr*s);
            xr -= static_cast<int32_t>(xi*(-c-1)/s);
            xi = -xi;
        }
    }
    else{
        if(s < -c){ // (-0.75pi, -0.25pi)
            const int32_t t = xr; xr = -xi; xi = t;
            xr -= static_cast<int32_t>(xi*(-s-1)/c);
            xi -= static_cast<int32_t>(xr*c);
            xr -= static_cast<int32_t>(xi*(-s-1)/c);
        }
        else{ // (-0.25pi, 0.25pi)
            xr -= static_cast<int32_t>(xi*(c-1)/s);
            xi -= static_cast<int32_t>(xr*s);
            xr -= static_cast<int32_t>(xi*(c-1)/s);
        }
    }
    return {xr, xi};
}

// Created by referring to http://www.kurims.kyoto-u.ac.jp/~ooura/fftman/ftmn1_24.html#sec1_2_4
void fft_(int n, int32_t* ar, int32_t* ai)
{   
    // L shaped butterflies
    for (int m = n; m > 2; m >>= 1) {
        const double theta = -2 * PI / m;
        const int mq = m >> 2;
        for (int i = 0; i < mq; i++) {
            const double s1 = std::sin(theta * i);
            const double c1 = std::cos(theta * i);
            const double s3 = std::sin(theta * 3 * i);
            const double c3 = std::cos(theta * 3 * i);
            for (int k = m; k <= n; k <<= 2) {
                for (int j0 = k - m + i; j0 < n; j0 += 2 * k) {
                    const int j1 = j0 + mq;
                    const int j2 = j1 + mq;
                    const int j3 = j2 + mq;
                    const int32_t x1r = ar[j0] - ar[j2];
                    const int32_t x1i = ai[j0] - ai[j2];
                    ar[j0] += ar[j2];
                    ai[j0] += ai[j2];
                    const int32_t x3r = ar[j1] - ar[j3];
                    const int32_t x3i = ai[j1] - ai[j3];
                    ar[j1] += ar[j3];
                    ai[j1] += ai[j3];
                    std::tie(ar[j2], ai[j2]) = lift_(x1r + x3i, x1i - x3r, c1, s1);
                    std::tie(ar[j3], ai[j3]) = lift_(x1r - x3i, x1i + x3r, c3, s3);
                }
            }
        }

    }

    // radix 2 butterflies
    for (int k = 2; k <= n; k <<= 2) {
        for (int j = k - 2; j < n; j += 2 * k) {
            const int32_t x0r = ar[j] - ar[j + 1];
            const int32_t x0i = ai[j] - ai[j + 1];
            ar[j] += ar[j + 1];
            ai[j] += ai[j + 1];
            ar[j + 1] = x0r;
            ai[j + 1] = x0i;
        }
    }
    // unscrambler
    for (int i = 0, j = 1; j < n - 1; j++) {
        for (int k = n >> 1; k > (i ^= k); k >>= 1);
        if (j < i) {
            const int32_t x0r = ar[j];
            const int32_t x0i = ai[j];
            ar[j] = ar[i];
            ai[j] = ai[i];
            ar[i] = x0r;
            ai[i] = x0i;
        }
    }
}

// Created by referring to https://www.kurims.kyoto-u.ac.jp/~ooura/fftman/ftmn2_12.html#sec2_1_2
void rfft_(int n, int32_t* a)
{
    /* ---- scrambler ---- */
    for (int i = 0, j = 1; j < n - 1; j++) {
        for (int k = n >> 1; k > (i ^= k); k >>= 1);
        if (j < i) {
            int32_t xr = a[j];
            a[j] = a[i];
            a[i] = xr;
        }
    }
    double theta = -8 * atan(1.0);  /* -2*pi */
    for (int mh = 1, m; (m = mh << 1) <= n; mh = m) {
        int mq = mh >> 1;
        theta *= 0.5;
        /* ---- real to real butterflies (W == 1) ---- */
        for (int jr = 0; jr < n; jr += m) {
            int kr = jr + mh;
            int32_t xr = a[kr];
            a[kr] = a[jr] - xr;
            a[jr] += xr;
        }
        /* ---- complex to complex butterflies (W != 1) ---- */
        for (int i = 1; i < mq; i++) {
            const double wr = std::cos(theta * i);
            const double wi = std::sin(theta * i);
            for (int j = 0; j < n; j += m) {
                int jr = j + i;
                int ji = j + mh - i;
                int kr = j + mh + i;
                int ki = j + m - i;

                //double xr = wr * a[kr] + wi * a[ki];
                //double xi = -(wr * a[ki] - wi * a[kr]);
                auto[xr, xi] = lift_(a[kr], -a[ki], wr, wi);
                a[kr] = -a[ji] - xi;
                a[ki] = a[ji] - xi;
                a[ji] = a[jr] - xr;
                a[jr] = a[jr] + xr;
            }
        }
    }
}

// Created by referring to https://www.kurims.kyoto-u.ac.jp/~ooura/fftman/ftmn2_12.html#sec2_1_2
void irfft_(int n, int32_t* a)
{
    double theta = -8 * atan(1.0) / n;  /* 2*pi/n */
    for (int m = n, mh; (mh = m >> 1) >= 1; m = mh) {
        int mq = mh >> 1;

        /* ---- complex to complex butterflies (W != 1) ---- */
        for (int i = 1; i < mq; i++) {
            const double wr = cos(theta * i);
            const double wi = sin(theta * i);
            for (int j = 0; j < n; j += m) {
                int jr = j + i;
                int ji = j + mh - i;
                int kr = j + mh + i;
                int ki = j + m - i;
                
                int32_t xr = -(a[ji] - a[jr])/2;
                int32_t xi = -(a[kr] + a[ki])/2;
                a[jr] = (a[ji] + a[jr])/2;
                a[ji] = -(a[kr] - a[ki])/2;

                std::tie(a[kr], a[ki]) = ilift_(xr, xi, wr, wi);
                a[ki] = -a[ki];
            }
        }

        /* ---- real to real butterflies (W == 1) ---- */
        for (int jr = 0; jr < n; jr += m) {
            int kr = jr + mh;
            int32_t xr = a[jr];
            a[jr] = (xr + a[kr]) / 2;
            a[kr] = (xr - a[kr]) / 2;
        }
        theta *= 2;
    }
    
    /* ---- unscrambler ---- */
    for (int i = 0, j = 1; j < n - 1; j++) {
        for (int k = n >> 1; k > (i ^= k); k >>= 1);
        if (j < i) {
            int32_t xr = a[j];
            a[j] = a[i];
            a[i] = xr;
        }
    }
}

// Created by referring to http://www.kurims.kyoto-u.ac.jp/~ooura/fftman/ftmn1_24.html#sec1_2_4
void ifft_(int n, int32_t* ar, int32_t* ai)
{
    // scrambler
    for (int i = 0, j = 1; j < n - 1; j++) {
        for (int k = n >> 1; k > (i ^= k); k >>= 1);
        if (j < i) {
            const int32_t x0r = ar[j];
            const int32_t x0i = ai[j];
            ar[j] = ar[i];
            ai[j] = ai[i];
            ar[i] = x0r;
            ai[i] = x0i;
        }
    }

    // radix 2 butterflies
    for (int k = 2; k <= n; k <<= 2) {
        for (int j = k - 2; j < n; j += 2 * k) {
            const int32_t x0r = ar[j];
            const int32_t x0i = ai[j];
            ar[j] = (x0r + ar[j + 1]) / 2;
            ai[j] = (x0i + ai[j + 1]) / 2;
            ar[j + 1] = x0r - ar[j];
            ai[j + 1] = x0i - ai[j];
        }
    }

    // L shaped butterflies
    for (int m = 4; m <= n; m <<= 1) {
        const double theta =  - 2 * PI / m;
        const int mq = m >> 2;
        for (int i = 0; i < mq; i++) {
            const double s1 = std::sin(theta * i);
            const double c1 = std::cos(theta * i);
            const double s3 = std::sin(theta * 3 * i);
            const double c3 = std::cos(theta * 3 * i);
            for (int k = m; k <= n; k <<= 2) {
                for (int j0 = k - m + i; j0 < n; j0 += 2 * k) {
                    const int j1 = j0 + mq;
                    const int j2 = j1 + mq;
                    const int j3 = j2 + mq;
                    const int32_t x0r = ar[j0];
                    const int32_t x0i = ai[j0];
                    const int32_t x1r = ar[j1];
                    const int32_t x1i = ai[j1];
                    auto [x2r, x2i] = ilift_(ar[j2], ai[j2], c1, s1);
                    auto [x3r, x3i] = ilift_(ar[j3], ai[j3], c3, s3);
                    const int32_t x2r_ = (x2r + x3r)/2;
                    const int32_t x2i_ = (x2i + x3i)/2;
                    const int32_t x3r_ = -(x2i - x2i_);
                    const int32_t x3i_ = (x2r - x2r_);
                    ar[j0] = (x0r + x2r_)/2;
                    ai[j0] = (x0i + x2i_)/2;
                    ar[j1] = (x1r + x3r_)/2;
                    ai[j1] = (x1i + x3i_)/2;
                    ar[j2] = (x0r - ar[j0]);
                    ai[j2] = (x0i - ai[j0]);
                    ar[j3] = (x1r - ar[j1]);
                    ai[j3] = (x1i - ai[j1]);
                }
            }
        }
    }
}

bool check_pow2(size_t x)
{
    if(x == 0){
        return false;
    }
    return (x & (x-1))==0;
}

void check_range(py::array_t<int32_t,0>& a)
{
    ssize_t n = a.shape(0);
    if(n >= 2){
        int32_t min_value = - static_cast<int32_t>(0x80000000U / n);
        int32_t max_value = static_cast<int32_t>(0x80000000U / n) - 1;

        int32_t* ptr = static_cast<int32_t*>(a.request().ptr);
        bool result = std::all_of(ptr, ptr+n, [min_value, max_value](int32_t x){
            return (min_value <= x) && (x <= max_value);
        });

        if (result==false){
            throw std::runtime_error("value range is assumed to be ["
             + std::to_string(min_value)
             + ", " 
             + std::to_string(max_value)
             + "]");
        }
    }
}

void check_args(py::array_t<int32_t, 0> ar, py::array_t<int32_t, 0> ai)
{
    if(ar.ndim() != 1){
        throw std::runtime_error("ar.ndim != 1");
    }
    if(ai.ndim() != 1){
        throw std::runtime_error("ai.ndim != 1");
    }
    if(ar.shape(0) != ai.shape(0)){
        throw std::runtime_error("ar.shape(0) != ai.shape(0)");
    }
    if(check_pow2(ar.shape(0)) == false){
        throw std::runtime_error("ar.shape(0) is not a power of 2");
    }
}

std::tuple<py::array_t<int32_t>, py::array_t<int32_t>> fft(py::array_t<int32_t, 0> ar, py::array_t<int32_t, 0> ai) {
    check_args(ar, ai);
    if (ar.ref_count()>1){
        ar = py::array_t<int32_t, 0>(ar.request());
    }
    if (ai.ref_count()>1){
        ai = py::array_t<int32_t, 0>(ai.request());
    }
    check_range(ar);
    check_range(ai);
    fft_(static_cast<int>(ar.shape(0)), static_cast<int32_t*>(ar.request().ptr), static_cast<int32_t*>(ai.request().ptr));
    return {ar, ai};
}

std::tuple<py::array_t<int32_t>, py::array_t<int32_t>> ifft(py::array_t<int32_t, 0> ar, py::array_t<int32_t, 0> ai) {
    check_args(ar, ai);
    if (ar.ref_count()>1){
        ar = py::array_t<int32_t, 0>(ar.request());
    }
    if (ai.ref_count()>1){
        ai = py::array_t<int32_t, 0>(ai.request());
    }
    ifft_(static_cast<int>(ar.shape(0)), static_cast<int32_t*>(ar.request().ptr), static_cast<int32_t*>(ai.request().ptr));
    return {ar, ai};
}

py::array_t<int32_t, 0> rfft(py::array_t<int32_t, 0> a) {
    if (a.ref_count()>1){
        a = py::array_t<int32_t, 0>(a.request());
    }
    rfft_(static_cast<int>(a.shape(0)), static_cast<int32_t*>(a.request().ptr));

    return a;
}

py::array_t<int32_t, 0> irfft(py::array_t<int32_t, 0> a) {
    if (a.ref_count()>1){
        a = py::array_t<int32_t, 0>(a.request());
    }
    irfft_(static_cast<int>(a.shape(0)), static_cast<int32_t*>(a.request().ptr));

    return a;
}

PYBIND11_MODULE(intfft, m) {
    m.doc() = "Integer Fast Fourier Transform in Python";
    m.def("fft", &fft);
    m.def("ifft", &ifft);
    m.def("rfft", &rfft);
    m.def("irfft", &irfft);
 }

 /*
void fft_(int n, int32_t* ar, int32_t* ai)
{  
    // unscrambler
    for (int i = 0, j = 1; j < n - 1; j++) {
        for (int k = n >> 1; k > (i ^= k); k >>= 1);
        if (j < i) {
            const int32_t x0r = ar[j];
            const int32_t x0i = ai[j];
            ar[j] = ar[i];
            ai[j] = ai[i];
            ar[i] = x0r;
            ai[i] = x0i;
        }
    }

    const double theta = -PI / 2 / n;

    // L shaped butterflies
    for (int m = 4; m <= n; m <<= 1) { 
        const int mq = m >> 2;
        int irev = 0;
        for (int i = 0; i < n; i += m) {
            const double s1 = std::sin(theta * irev);
            const double c1 = std::cos(theta * irev);
            const double s3 = std::sin(theta * 3 * irev);
            const double c3 = std::cos(theta * 3 * irev);
            for (int k = n >> 1; k > (irev ^= k); k >>= 1);
            for (int k = mq; k >= 1; k >>= 2) {
                for (int j0 = i + mq - k; j0 < i + mq - (k >> 1); j0++) {
                    const int j1 = j0 + mq;
                    const int j2 = j1 + mq;
                    const int j3 = j2 + mq;
                    const int32_t x1r = ar[j0] - ar[j1];
                    const int32_t x1i = ai[j0] - ai[j1];
                    ar[j0] += ar[j1];
                    ai[j0] += ai[j1];
                    const int32_t x3r = ar[j3] - ar[j2];
                    const int32_t x3i = ai[j3] - ai[j2];
                    ar[j2] += ar[j3];
                    ai[j2] += ai[j3];
                    std::tie(ar[j1], ai[j1]) = lift_(x1r - x3i, x1i + x3r, c1, s1);
                    std::tie(ar[j3], ai[j3]) = lift_(x1r + x3i, x1i - x3r, c3, s3);
                }
            }
        }
    }

    // radix 2 butterflies
    const int mq = n >> 1;
    for (int k = mq; k >= 1; k >>= 2) {
        for (int j = mq - k; j < mq - (k >> 1); j++) {
            int j1 = mq + j;
            const int32_t x0r = ar[j] - ar[j1];
            const int32_t x0i = ai[j] - ai[j1];
            ar[j] += ar[j1];
            ai[j] += ai[j1];
            ar[j1] = x0r;
            ai[j1] = x0i;
        }
    }
}
*/