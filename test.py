import unittest
import numpy as np
from intfft import fft, ifft

class TestIntfft(unittest.TestCase):

    def _test_001_fft_input_int(self, dtype):
        xr1 = np.arange(2**7, dtype=dtype)
        xi1 = np.arange(2**7, dtype=dtype)
        xr1_ = xr1.copy()
        xi1_ = xi1.copy()
        _, _ = fft(xr1, xi1)
        self.assertTrue(np.all(xr1_==xr1))
        self.assertTrue(np.all(xi1_==xi1))
    
    def test_001_fft_input_int_i8(self):self._test_001_fft_input_int(np.int8)
    def test_001_fft_input_int_i16(self):self._test_001_fft_input_int(np.int16)
    def test_001_fft_input_int_i32(self):self._test_001_fft_input_int(np.int32)
    def test_001_fft_input_int_u8(self):self._test_001_fft_input_int(np.uint8)
    def test_001_fft_input_int_u16(self):self._test_001_fft_input_int(np.uint16)

    def _test_002_ifft_input_int(self, dtype):
        xr1 = np.arange(2**7, dtype=dtype)
        xi1 = np.arange(2**7, dtype=dtype)
        xr1_ = xr1.copy()
        xi1_ = xi1.copy()
        _, _ = ifft(xr1, xi1)
        self.assertTrue(np.all(xr1_==xr1))
        self.assertTrue(np.all(xi1_==xi1))
    
    def test_002_ifft_input_int_i8(self):self._test_002_ifft_input_int(np.int8)
    def test_002_ifft_input_int_i18(self):self._test_002_ifft_input_int(np.int16)
    def test_002_ifft_input_int_i32(self):self._test_002_ifft_input_int(np.int32)
    def test_002_ifft_input_int_u8(self):self._test_002_ifft_input_int(np.uint8)
    def test_002_ifft_input_int_u16(self):self._test_002_ifft_input_int(np.uint16)
    
    def _test_003_fft_input_error(self, dtype):
        f = np.arange(2**7, dtype=dtype)
        i = np.arange(2**7, dtype=np.int32)
        with self.assertRaises(Exception):
            _, _ = fft(f, i)
        with self.assertRaises(Exception):
            _, _ = fft(i, f)
        with self.assertRaises(Exception):
            _, _ = fft(f, f)
    
    def test_003_fft_input_error_i64(self):self._test_003_fft_input_error(np.int64)
    def test_003_fft_input_error_u32(self):self._test_003_fft_input_error(np.uint32)
    def test_003_fft_input_error_u64(self):self._test_003_fft_input_error(np.uint64)
    def test_003_fft_input_error_f32(self):self._test_003_fft_input_error(np.float32)
    def test_003_fft_input_error_f64(self):self._test_003_fft_input_error(np.float64)
    def test_003_fft_input_error_c64(self):self._test_003_fft_input_error(np.complex64)
    def test_003_fft_input_error_c128(self):self._test_003_fft_input_error(np.complex128)

    def _test_004_ifft_input_error(self, dtype):
        f = np.arange(2**7, dtype=dtype)
        i = np.arange(2**7, dtype=np.int32)
        with self.assertRaises(Exception):
            _, _ = ifft(f, i)
        with self.assertRaises(Exception):
            _, _ = ifft(i, f)
        with self.assertRaises(Exception):
            _, _ = ifft(f, f)

    def test_004_ifft_input_error_i64(self):self._test_004_ifft_input_error(np.int64)
    def test_004_ifft_input_error_u32(self):self._test_004_ifft_input_error(np.uint32)
    def test_004_ifft_input_error_u64(self):self._test_004_ifft_input_error(np.uint64)
    def test_004_ifft_input_error_f32(self):self._test_004_ifft_input_error(np.float32)
    def test_004_ifft_input_error_f64(self):self._test_004_ifft_input_error(np.float64)
    def test_004_ifft_input_error_c64(self):self._test_004_ifft_input_error(np.complex64)
    def test_004_ifft_input_error_c128(self):self._test_004_ifft_input_error(np.complex128)

    def _test_005_fft_output_int(self, dtype):
        xr1 = np.arange(2**7, dtype=dtype)
        xi1 = np.arange(2**7, dtype=dtype)
        xr2, xi2 = fft(xr1, xi1)
        self.assertTrue(xr2.dtype==np.int32)
        self.assertTrue(xi2.dtype==np.int32)
        
    def test_005_fft_output_int_i8(self):self._test_005_fft_output_int(np.int8)
    def test_005_fft_output_int_i16(self):self._test_005_fft_output_int(np.int16)
    def test_005_fft_output_int_i32(self):self._test_005_fft_output_int(np.int32)
    def test_005_fft_output_int_u8(self):self._test_005_fft_output_int(np.uint8)
    def test_005_fft_output_int_u16(self):self._test_005_fft_output_int(np.uint16)

    def _test_006_fft_invertible_int(self, dtype):
        xr1 = np.arange(2**7, dtype=dtype)
        xi1 = np.arange(2**7, dtype=dtype)
        xr2, xi2 = fft(xr1, xi1)
        xr3, xi3 = ifft(xr2, xi2)
        self.assertTrue(np.all(xr3==xr1))
        self.assertTrue(np.all(xi3==xi1))
        
    def test_006_fft_invertible_int_i8(self):self._test_006_fft_invertible_int(np.int8)
    def test_006_fft_invertible_int_i16(self):self._test_006_fft_invertible_int(np.int16)
    def test_006_fft_invertible_int_i32(self):self._test_006_fft_invertible_int(np.int32)
    def test_006_fft_invertible_int_u8(self):self._test_006_fft_invertible_int(np.uint8)
    def test_006_fft_invertible_int_u16(self):self._test_006_fft_invertible_int(np.uint16)

    def _test_007_max_n_2(self, n):
        xr1 = np.array([(2**31)//n-1]*n, dtype=np.int32)
        xi1 = np.array([(2**31)//n-1]*n, dtype=np.int32)
        xr2, xi2 = fft(xr1, xi1)
        xr3, xi3 = ifft(xr2, xi2)
        self.assertTrue(np.all(xr3==xr1))
        self.assertTrue(np.all(xi3==xi1))
    
    def test_007_max_n_2_00(self):self._test_007_max_n_2(2**0)
    def test_007_max_n_2_01(self):self._test_007_max_n_2(2**1)
    def test_007_max_n_2_02(self):self._test_007_max_n_2(2**2)
    def test_007_max_n_2_03(self):self._test_007_max_n_2(2**3)
    def test_007_max_n_2_04(self):self._test_007_max_n_2(2**4)
    def test_007_max_n_2_05(self):self._test_007_max_n_2(2**5)
    def test_007_max_n_2_06(self):self._test_007_max_n_2(2**6)
    def test_007_max_n_2_07(self):self._test_007_max_n_2(2**7)
    def test_007_max_n_2_08(self):self._test_007_max_n_2(2**8)
    def test_007_max_n_2_09(self):self._test_007_max_n_2(2**9)
    def test_007_max_n_2_10(self):self._test_007_max_n_2(2**10)
    def test_007_max_n_2_11(self):self._test_007_max_n_2(2**11)
    def test_007_max_n_2_12(self):self._test_007_max_n_2(2**12)
    def test_007_max_n_2_13(self):self._test_007_max_n_2(2**13)
    def test_007_max_n_2_14(self):self._test_007_max_n_2(2**14)
    def test_007_max_n_2_15(self):self._test_007_max_n_2(2**15)
    def test_007_max_n_2_16(self):self._test_007_max_n_2(2**16)
    def test_007_max_n_2_17(self):self._test_007_max_n_2(2**17)
    def test_007_max_n_2_18(self):self._test_007_max_n_2(2**18)
    def test_007_max_n_2_19(self):self._test_007_max_n_2(2**19)
    def test_007_max_n_2_20(self):self._test_007_max_n_2(2**20)
    
    def _test_008_min_n_2(self, n):
        xr1 = np.array([-(2**31)//n]*n, dtype=np.int32)
        xi1 = np.array([-(2**31)//n]*n, dtype=np.int32)
        xr2, xi2 = fft(xr1, xi1)
        xr3, xi3 = ifft(xr2, xi2)
        self.assertTrue(np.all(xr3==xr1))
        self.assertTrue(np.all(xi3==xi1))
    
    def test_008_min_n_2_00(self):self._test_008_min_n_2(2**0)
    def test_008_min_n_2_01(self):self._test_008_min_n_2(2**1)
    def test_008_min_n_2_02(self):self._test_008_min_n_2(2**2)
    def test_008_min_n_2_03(self):self._test_008_min_n_2(2**3)
    def test_008_min_n_2_04(self):self._test_008_min_n_2(2**4)
    def test_008_min_n_2_05(self):self._test_008_min_n_2(2**5)
    def test_008_min_n_2_06(self):self._test_008_min_n_2(2**6)
    def test_008_min_n_2_07(self):self._test_008_min_n_2(2**7)
    def test_008_min_n_2_08(self):self._test_008_min_n_2(2**8)
    def test_008_min_n_2_09(self):self._test_008_min_n_2(2**9)
    def test_008_min_n_2_10(self):self._test_008_min_n_2(2**10)
    def test_008_min_n_2_11(self):self._test_008_min_n_2(2**11)
    def test_008_min_n_2_12(self):self._test_008_min_n_2(2**12)
    def test_008_min_n_2_13(self):self._test_008_min_n_2(2**13)
    def test_008_min_n_2_14(self):self._test_008_min_n_2(2**14)
    def test_008_min_n_2_15(self):self._test_008_min_n_2(2**15)
    def test_008_min_n_2_16(self):self._test_008_min_n_2(2**16)
    def test_008_min_n_2_17(self):self._test_008_min_n_2(2**17)
    def test_008_min_n_2_18(self):self._test_008_min_n_2(2**18)
    def test_008_min_n_2_19(self):self._test_008_min_n_2(2**19)
    def test_008_min_n_2_20(self):self._test_008_min_n_2(2**20)
            
    def _test_009_random_n_2(self, n):
        for _ in range(10):
            xr1 = np.random.randint(-(2**31)//n, (2**31)//n, n, dtype=np.int32)
            xi1 = np.random.randint(-(2**31)//n, (2**31)//n, n, dtype=np.int32)
            xr2, xi2 = fft(xr1, xi1)
            xr3, xi3 = ifft(xr2, xi2)
            self.assertTrue(np.all(xr3==xr1))
            self.assertTrue(np.all(xi3==xi1))
    
    def test_009_random_n_2_00(self):self._test_009_random_n_2(2**0)
    def test_009_random_n_2_01(self):self._test_009_random_n_2(2**1)
    def test_009_random_n_2_02(self):self._test_009_random_n_2(2**2)
    def test_009_random_n_2_03(self):self._test_009_random_n_2(2**3)
    def test_009_random_n_2_04(self):self._test_009_random_n_2(2**4)
    def test_009_random_n_2_05(self):self._test_009_random_n_2(2**5)
    def test_009_random_n_2_06(self):self._test_009_random_n_2(2**6)
    def test_009_random_n_2_07(self):self._test_009_random_n_2(2**7)
    def test_009_random_n_2_08(self):self._test_009_random_n_2(2**8)
    def test_009_random_n_2_09(self):self._test_009_random_n_2(2**9)
    def test_009_random_n_2_10(self):self._test_009_random_n_2(2**10)
    def test_009_random_n_2_11(self):self._test_009_random_n_2(2**11)
    def test_009_random_n_2_12(self):self._test_009_random_n_2(2**12)
    def test_009_random_n_2_13(self):self._test_009_random_n_2(2**13)
    def test_009_random_n_2_14(self):self._test_009_random_n_2(2**14)
    def test_009_random_n_2_15(self):self._test_009_random_n_2(2**15)
    def test_009_random_n_2_16(self):self._test_009_random_n_2(2**16)
    def test_009_random_n_2_17(self):self._test_009_random_n_2(2**17)
    def test_009_random_n_2_18(self):self._test_009_random_n_2(2**18)
    def test_009_random_n_2_19(self):self._test_009_random_n_2(2**19)
    def test_009_random_n_2_20(self):self._test_009_random_n_2(2**20)

    def test_010_fft_output_value(self):
        xr1 = np.arange(2**5, dtype=np.int32)
        xi1 = np.zeros(2**5, dtype=np.int32)
        xr2, xi2 = fft(xr1, xi1)
        xr2_ = [496, -20, -18, -16, -17, -15, -17, -16, -16, -17, -15, -17, -17, -15, -14, -16, -16, -16, -16, -18, -15, -15, -17, -12, -16, -15, -15, -17, -15, -15, -16, -16]
        xi2_ = [0, 158, 79, 51, 39, 34, 25, 18, 16, 15, 12, 2, 6, 6, 2, 3, 0, 0, -1, -1, -7, -10, -13, -14, -16, -17, -26, -28, -38, -58, -78, -159]
        self.assertTrue(np.all(xr2 == xr2_))
        self.assertTrue(np.all(xi2 == xi2_))

if __name__ == '__main__':
    unittest.main(verbosity=2)
