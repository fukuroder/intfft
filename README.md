# intfft
Integer FFT(Fast Fourier Transform) in Python

Function list
------------
|Name|Description|
|:---|:---|
|fft|Discrete Fourier transform|
|ifft|Inverse discrete Fourier transform|

Installation
------------
```
pip install git+https://github.com/fukuroder/intfft
```

Tests
------------
```py
>>> import intfft
>>> xr = [1, 2, 3, 4, 5, 6, 7, 8]
>>> xi = [0, 0, 0, 0, 0, 0, 0, 0]
>>> wr, wi = intfft.fft(xr, xi)
>>> wr
array([36, -5, -4, -5, -4, -3, -4, -3])
>>> wi
array([ 0, 10,  4,  1,  0, -2, -4, -9])
>>> xr_, xi_ = intfft.ifft(wr, wi)
>>> xr_
array([1, 2, 3, 4, 5, 6, 7, 8])
>>> xi_
array([0, 0, 0, 0, 0, 0, 0, 0])
```

Tasks
------------
- [ ] Add real FFT/IFFT
- [ ] Add DCT/IDCT
- [ ] Add MDCT/IMDCT
- [x] Implement with C++ extension
- [x] Implement with in-place algorithm
- [ ] Register to PyPI

Reference
------------
- Soontorn Oraintara, Ying-Jui Chen, Truong Q.Nguyen: Integer Fast Fourier Transform. IEEE Transactions on Signal Processing, Vol. 50, 2002.
- FFT (高速フーリエ・コサイン・サイン変換) の概略と設計法 https://www.kurims.kyoto-u.ac.jp/~ooura/fftman/
