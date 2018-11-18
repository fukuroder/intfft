# pyintfft
Integer FFT(Fast Fourier Transform) in Python

Function list
------------
|Name|Description|
|:---|:---|
|fft|Discrete Fourier transform|
|ifft|Inverse discrete Fourier transform|
|rfft|Discrete Fourier transform of a real sequence|
|irfft|Inverse discrete Fourier transform of a real sequence|


Installation
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
array([36. +0.j, -5.+10.j, -4. +4.j, -3. +2.j, -4. +0.j, -3. -2.j, -4. -4.j, -5.-10.j])
>>> intfft.ifft(a)
array([1.+0.j, 2.+0.j, 3.+0.j, 4.+0.j, 5.+0.j, 6.+0.j, 7.+0.j, 8.+0.j])
```

Tasks
------------
- [ ] Add DCT/IDCT
- [ ] Implement with C ++ extension

Reference
------------
- Soontorn Oraintara, Ying-Jui Chen, Truong Q.Nguyen: Integer Fast Fourier Transform. IEEE Transactions on Signal Processing, Vol. 50, 2002.
