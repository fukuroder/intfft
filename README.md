# intfft
Integer FFT(Fast Fourier Transform) in Python

Function list
------------
|Name|Description|
|:---|:---|
|fft|Discrete Fourier transform|
|ifft|Inverse discrete Fourier transform|

Installation(Ubuntu)
------------
```
sudo python3 setup.py install
```

Tests
------------
```py
>>> import intfft
>>> a=intfft.fft([1, 2, 3, 4, 5, 6, 7, 8])
>>> a
array([36. +0.j, -5.+10.j, -4. +4.j, -5. +1.j, -4. +0.j, -3. -2.j, -4. -4.j, -3. -9.j])

>>> intfft.ifft(a)
array([1.+0.j, 2.+0.j, 3.+0.j, 4.+0.j, 5.+0.j, 6.+0.j, 7.+0.j, 8.+0.j])
```

Tasks
------------
- [ ] Add real FFT/IFFT
- [ ] Add DCT/IDCT
- [ ] Add MDCT/IMDCT
- [ ] Implement with C++ extension
- [ ] Implement with in-place algorithm
- [ ] Register to PyPI

Reference
------------
- Soontorn Oraintara, Ying-Jui Chen, Truong Q.Nguyen: Integer Fast Fourier Transform. IEEE Transactions on Signal Processing, Vol. 50, 2002.
