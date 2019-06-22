import unittest
import numpy as np
from intfft import fft, ifft, rfft, irfft

class TestIntfft(unittest.TestCase):
    def test_fft(self):
        x = list(range(2**5))
        expect = \
[496.  +0.j, -20.+158.j, -18. +79.j, -16. +51.j, -17. +39.j, -15. +34.j, \
 -17. +25.j, -16. +18.j, -16. +16.j, -17. +15.j, -15. +12.j, -17.  +2.j, \
 -17.  +6.j, -15.  +6.j, -14.  +2.j, -16.  +3.j, -16.  +0.j, -16.  +0.j, \
 -16.  -1.j, -18.  -1.j, -15.  -7.j, -15. -10.j, -17. -13.j, -12. -14.j, \
 -16. -16.j, -15. -17.j, -15. -26.j, -17. -28.j, -15. -38.j, -15. -58.j, \
 -16. -78.j, -16.-159.j,]

        self.assertTrue(np.all(fft(x)==np.array(expect)))

    def test_ifft(self):
        x = \
[496.  +0.j, -20.+158.j, -18. +79.j, -16. +51.j, -17. +39.j, -15. +34.j, \
 -17. +25.j, -16. +18.j, -16. +16.j, -17. +15.j, -15. +12.j, -17.  +2.j, \
 -17.  +6.j, -15.  +6.j, -14.  +2.j, -16.  +3.j, -16.  +0.j, -16.  +0.j, \
 -16.  -1.j, -18.  -1.j, -15.  -7.j, -15. -10.j, -17. -13.j, -12. -14.j, \
 -16. -16.j, -15. -17.j, -15. -26.j, -17. -28.j, -15. -38.j, -15. -58.j, \
 -16. -78.j, -16.-159.j,]
        expect = list(range(2**5))

        self.assertTrue(np.all(ifft(x)==np.array(expect, dtype=np.float)))

if __name__ == '__main__':
    unittest.main()
