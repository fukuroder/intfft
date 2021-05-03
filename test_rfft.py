import unittest
import numpy as np
from intfft import rfft, irfft

class TestRFFT(unittest.TestCase):
    # confirm the input arguments .
    def _test_fft_input_type(self, dtype):
        xr1 = np.arange(2**7, dtype=dtype)
        _ = rfft(xr1)
    def test_fft_input_type_i8(self):self._test_fft_input_type(np.int8)
    def test_fft_input_type_i16(self):self._test_fft_input_type(np.int16)
    def test_fft_input_type_i32(self):self._test_fft_input_type(np.int32)
    def test_fft_input_type_u8(self):self._test_fft_input_type(np.uint8)
    def test_fft_input_type_u16(self):self._test_fft_input_type(np.uint16)

    # confirm the input arguments .
    def _test_ifft_input_type(self, dtype):
        xr1 = np.arange(2**7, dtype=dtype)
        _ = irfft(xr1)
    def test_ifft_input_type_i8(self):self._test_ifft_input_type(np.int8)
    def test_ifft_input_type_i16(self):self._test_ifft_input_type(np.int16)
    def test_ifft_input_type_i32(self):self._test_ifft_input_type(np.int32)
    def test_ifft_input_type_u8(self):self._test_ifft_input_type(np.uint8)
    def test_ifft_input_type_u16(self):self._test_ifft_input_type(np.uint16)

    # confirm the input arguments type(error)
    def _test_fft_input_ar_type_error(self, dtype):
        ar = np.arange(2**7, dtype=dtype)
        with self.assertRaisesRegex(Exception, "incompatible function arguments. The following argument types are supported:"):
            _ = rfft(ar)
    def test_fft_input_ar_type_error_i64(self):self._test_fft_input_ar_type_error(np.int64)
    def test_fft_input_ar_type_error_u32(self):self._test_fft_input_ar_type_error(np.uint32)
    def test_fft_input_ar_type_error_u64(self):self._test_fft_input_ar_type_error(np.uint64)
    def test_fft_input_ar_type_error_f32(self):self._test_fft_input_ar_type_error(np.float32)
    def test_fft_input_ar_type_error_f64(self):self._test_fft_input_ar_type_error(np.float64)
    def test_fft_input_ar_type_error_c64(self):self._test_fft_input_ar_type_error(np.complex64)
    def test_fft_input_ar_type_error_c128(self):self._test_fft_input_ar_type_error(np.complex128)

    # confirm the input arguments type(error)
    def _test_ifft_input_ar_error(self, dtype):
        ar = np.arange(2**7, dtype=dtype)
        ai = np.arange(2**7, dtype=np.int32)
        with self.assertRaisesRegex(Exception, "incompatible function arguments. The following argument types are supported:"):
            _ = irfft(ar)
    def test_ifft_input_ar_error_i64(self):self._test_ifft_input_ar_error(np.int64)
    def test_ifft_input_ar_error_u32(self):self._test_ifft_input_ar_error(np.uint32)
    def test_ifft_input_ar_error_u64(self):self._test_ifft_input_ar_error(np.uint64)
    def test_ifft_input_ar_error_f32(self):self._test_ifft_input_ar_error(np.float32)
    def test_ifft_input_ar_error_f64(self):self._test_ifft_input_ar_error(np.float64)
    def test_ifft_input_ar_error_c64(self):self._test_ifft_input_ar_error(np.complex64)
    def test_ifft_input_ar_error_c128(self):self._test_ifft_input_ar_error(np.complex128)

    # confirm the output type is "int32"
    def _test_fft_output_type(self, dtype):
        xr1 = np.arange(2**7, dtype=dtype)
        xr2 = rfft(xr1)
        self.assertTrue(xr2.dtype==np.int32)
    def test_fft_output_type_i8(self):self._test_fft_output_type(np.int8)
    def test_fft_output_type_i16(self):self._test_fft_output_type(np.int16)
    def test_fft_output_type_i32(self):self._test_fft_output_type(np.int32)
    def test_fft_output_type_u8(self):self._test_fft_output_type(np.uint8)
    def test_fft_output_type_u16(self):self._test_fft_output_type(np.uint16)

    # confirm the input arguments is immutable.
    def _test_fft_input_immutable(self, dtype):
        xr1 = np.arange(2**7, dtype=dtype)
        xr1_ = xr1.copy()
        _ = rfft(xr1)
        self.assertTrue(np.all(xr1_==xr1))
    def test_fft_input_immutable_i8(self):self._test_fft_input_immutable(np.int8)
    def test_fft_input_immutable_i16(self):self._test_fft_input_immutable(np.int16)
    def test_fft_input_immutable_i32(self):self._test_fft_input_immutable(np.int32)
    def test_fft_input_immutable_u8(self):self._test_fft_input_immutable(np.uint8)
    def test_fft_input_immutable_u16(self):self._test_fft_input_immutable(np.uint16)

    # confirm the input arguments is immutable.
    def _test_ifft_input_immutable(self, dtype):
        xr1 = np.arange(2**7, dtype=dtype)
        xr1_ = xr1.copy()
        self.assertTrue(np.all(xr1_==xr1))
    def test_ifft_input_immutable_i8(self):self._test_ifft_input_immutable(np.int8)
    def test_ifft_input_immutable_i16(self):self._test_ifft_input_immutable(np.int16)
    def test_ifft_input_immutable_i32(self):self._test_ifft_input_immutable(np.int32)
    def test_ifft_input_immutable_u8(self):self._test_ifft_input_immutable(np.uint8)
    def test_ifft_input_immutable_u16(self):self._test_ifft_input_immutable(np.uint16)

    # confirm invertible
    def _test_fft_invertible(self, dtype):
        xr1 = np.arange(2**7, dtype=dtype)
        xr2 = rfft(xr1)
        xr3 = irfft(xr2)
        self.assertTrue(np.all(xr3==xr1))
    def test_fft_invertible_i8(self):self._test_fft_invertible(np.int8)
    def test_fft_invertible_i16(self):self._test_fft_invertible(np.int16)
    def test_fft_invertible_i32(self):self._test_fft_invertible(np.int32)
    def test_fft_invertible_u8(self):self._test_fft_invertible(np.uint8)
    def test_fft_invertible_u16(self):self._test_fft_invertible(np.uint16)

    # confirm input max values
    def _test_fft_input_max(self, n):
        xr1 = np.array([(2**31)//n-1]*n, dtype=np.int32)
        xr2 = rfft(xr1)
        xr3 = irfft(xr2)
        self.assertTrue(np.all(xr3==xr1))
    def test_fft_input_max_01(self):self._test_fft_input_max(2**1)
    def test_fft_input_max_02(self):self._test_fft_input_max(2**2)
    def test_fft_input_max_03(self):self._test_fft_input_max(2**3)
    def test_fft_input_max_04(self):self._test_fft_input_max(2**4)
    def test_fft_input_max_05(self):self._test_fft_input_max(2**5)
    def test_fft_input_max_06(self):self._test_fft_input_max(2**6)
    def test_fft_input_max_07(self):self._test_fft_input_max(2**7)
    def test_fft_input_max_08(self):self._test_fft_input_max(2**8)
    def test_fft_input_max_09(self):self._test_fft_input_max(2**9)
    def test_fft_input_max_10(self):self._test_fft_input_max(2**10)
    def test_fft_input_max_11(self):self._test_fft_input_max(2**11)
    def test_fft_input_max_12(self):self._test_fft_input_max(2**12)
    def test_fft_input_max_13(self):self._test_fft_input_max(2**13)
    def test_fft_input_max_14(self):self._test_fft_input_max(2**14)
    def test_fft_input_max_15(self):self._test_fft_input_max(2**15)
    def test_fft_input_max_16(self):self._test_fft_input_max(2**16)
    def test_fft_input_max_17(self):self._test_fft_input_max(2**17)
    def test_fft_input_max_18(self):self._test_fft_input_max(2**18)
    def test_fft_input_max_19(self):self._test_fft_input_max(2**19)
    def test_fft_input_max_20(self):self._test_fft_input_max(2**20)

    # confirm input max values(error)
    def _test_fft_input_ar_max_error(self, n):
        xr1 = np.array([(2**31)//n]*n, dtype=np.int32)
        with self.assertRaisesRegex(Exception, "value range is assumed to be \[-\d+, \d+\]"):
            xr2 = rfft(xr1)
    def test_fft_input_ar_max_error_01(self):self._test_fft_input_ar_max_error(2**1)
    def test_fft_input_ar_max_error_02(self):self._test_fft_input_ar_max_error(2**2)
    def test_fft_input_ar_max_error_03(self):self._test_fft_input_ar_max_error(2**3)
    def test_fft_input_ar_max_error_04(self):self._test_fft_input_ar_max_error(2**4)
    def test_fft_input_ar_max_error_05(self):self._test_fft_input_ar_max_error(2**5)
    def test_fft_input_ar_max_error_06(self):self._test_fft_input_ar_max_error(2**6)
    def test_fft_input_ar_max_error_07(self):self._test_fft_input_ar_max_error(2**7)
    def test_fft_input_ar_max_error_08(self):self._test_fft_input_ar_max_error(2**8)
    def test_fft_input_ar_max_error_09(self):self._test_fft_input_ar_max_error(2**9)
    def test_fft_input_ar_max_error_10(self):self._test_fft_input_ar_max_error(2**10)
    def test_fft_input_ar_max_error_11(self):self._test_fft_input_ar_max_error(2**11)
    def test_fft_input_ar_max_error_12(self):self._test_fft_input_ar_max_error(2**12)
    def test_fft_input_ar_max_error_13(self):self._test_fft_input_ar_max_error(2**13)
    def test_fft_input_ar_max_error_14(self):self._test_fft_input_ar_max_error(2**14)
    def test_fft_input_ar_max_error_15(self):self._test_fft_input_ar_max_error(2**15)
    def test_fft_input_ar_max_error_16(self):self._test_fft_input_ar_max_error(2**16)
    def test_fft_input_ar_max_error_17(self):self._test_fft_input_ar_max_error(2**17)
    def test_fft_input_ar_max_error_18(self):self._test_fft_input_ar_max_error(2**18)
    def test_fft_input_ar_max_error_19(self):self._test_fft_input_ar_max_error(2**19)
    def test_fft_input_ar_max_error_20(self):self._test_fft_input_ar_max_error(2**20)
    
    # confirm input min values
    def _test_fft_input_min(self, n):
        xr1 = np.array([-(2**31)//n]*n, dtype=np.int32)
        xr2 = rfft(xr1)
        xr3 = irfft(xr2)
        self.assertTrue(np.all(xr3==xr1))
    def test_fft_input_min_01(self):self._test_fft_input_min(2**1)
    def test_fft_input_min_02(self):self._test_fft_input_min(2**2)
    def test_fft_input_min_03(self):self._test_fft_input_min(2**3)
    def test_fft_input_min_04(self):self._test_fft_input_min(2**4)
    def test_fft_input_min_05(self):self._test_fft_input_min(2**5)
    def test_fft_input_min_06(self):self._test_fft_input_min(2**6)
    def test_fft_input_min_07(self):self._test_fft_input_min(2**7)
    def test_fft_input_min_08(self):self._test_fft_input_min(2**8)
    def test_fft_input_min_09(self):self._test_fft_input_min(2**9)
    def test_fft_input_min_10(self):self._test_fft_input_min(2**10)
    def test_fft_input_min_11(self):self._test_fft_input_min(2**11)
    def test_fft_input_min_12(self):self._test_fft_input_min(2**12)
    def test_fft_input_min_13(self):self._test_fft_input_min(2**13)
    def test_fft_input_min_14(self):self._test_fft_input_min(2**14)
    def test_fft_input_min_15(self):self._test_fft_input_min(2**15)
    def test_fft_input_min_16(self):self._test_fft_input_min(2**16)
    def test_fft_input_min_17(self):self._test_fft_input_min(2**17)
    def test_fft_input_min_18(self):self._test_fft_input_min(2**18)
    def test_fft_input_min_19(self):self._test_fft_input_min(2**19)
    def test_fft_input_min_20(self):self._test_fft_input_min(2**20)

    # confirm input max values(error)
    def _test_fft_input_ar_min_error(self, n):
        xr1 = np.array([-(2**31)//n-1]*n, dtype=np.int32)
        with self.assertRaisesRegex(Exception, "value range is assumed to be \[-\d+, \d+\]"):
            xr2 = rfft(xr1)
    def test_fft_input_ar_min_error_01(self):self._test_fft_input_ar_min_error(2**1)
    def test_fft_input_ar_min_error_02(self):self._test_fft_input_ar_min_error(2**2)
    def test_fft_input_ar_min_error_03(self):self._test_fft_input_ar_min_error(2**3)
    def test_fft_input_ar_min_error_04(self):self._test_fft_input_ar_min_error(2**4)
    def test_fft_input_ar_min_error_05(self):self._test_fft_input_ar_min_error(2**5)
    def test_fft_input_ar_min_error_06(self):self._test_fft_input_ar_min_error(2**6)
    def test_fft_input_ar_min_error_07(self):self._test_fft_input_ar_min_error(2**7)
    def test_fft_input_ar_min_error_08(self):self._test_fft_input_ar_min_error(2**8)
    def test_fft_input_ar_min_error_09(self):self._test_fft_input_ar_min_error(2**9)
    def test_fft_input_ar_min_error_10(self):self._test_fft_input_ar_min_error(2**10)
    def test_fft_input_ar_min_error_11(self):self._test_fft_input_ar_min_error(2**11)
    def test_fft_input_ar_min_error_12(self):self._test_fft_input_ar_min_error(2**12)
    def test_fft_input_ar_min_error_13(self):self._test_fft_input_ar_min_error(2**13)
    def test_fft_input_ar_min_error_14(self):self._test_fft_input_ar_min_error(2**14)
    def test_fft_input_ar_min_error_15(self):self._test_fft_input_ar_min_error(2**15)
    def test_fft_input_ar_min_error_16(self):self._test_fft_input_ar_min_error(2**16)
    def test_fft_input_ar_min_error_17(self):self._test_fft_input_ar_min_error(2**17)
    def test_fft_input_ar_min_error_18(self):self._test_fft_input_ar_min_error(2**18)
    def test_fft_input_ar_min_error_19(self):self._test_fft_input_ar_min_error(2**19)
    def test_fft_input_ar_min_error_20(self):self._test_fft_input_ar_min_error(2**20)

    # confirm invertible(random input)       
    def _test_fft_invertible_random(self, n):
        for _ in range(10):
            xr1 = np.random.randint(-(2**31)//n, (2**31)//n, n, dtype=np.int32)
            xr2 = rfft(xr1)
            xr3 = irfft(xr2)
            self.assertTrue(np.all(xr3==xr1))
    def test_fft_invertible_random_00(self):self._test_fft_invertible_random(2**0)
    def test_fft_invertible_random_01(self):self._test_fft_invertible_random(2**1)
    def test_fft_invertible_random_02(self):self._test_fft_invertible_random(2**2)
    def test_fft_invertible_random_03(self):self._test_fft_invertible_random(2**3)
    def test_fft_invertible_random_04(self):self._test_fft_invertible_random(2**4)
    def test_fft_invertible_random_05(self):self._test_fft_invertible_random(2**5)
    def test_fft_invertible_random_06(self):self._test_fft_invertible_random(2**6)
    def test_fft_invertible_random_07(self):self._test_fft_invertible_random(2**7)
    def test_fft_invertible_random_08(self):self._test_fft_invertible_random(2**8)
    def test_fft_invertible_random_09(self):self._test_fft_invertible_random(2**9)
    def test_fft_invertible_random_10(self):self._test_fft_invertible_random(2**10)
    def test_fft_invertible_random_11(self):self._test_fft_invertible_random(2**11)
    def test_fft_invertible_random_12(self):self._test_fft_invertible_random(2**12)
    def test_fft_invertible_random_13(self):self._test_fft_invertible_random(2**13)
    def test_fft_invertible_random_14(self):self._test_fft_invertible_random(2**14)
    def test_fft_invertible_random_15(self):self._test_fft_invertible_random(2**15)
    def test_fft_invertible_random_16(self):self._test_fft_invertible_random(2**16)
    def test_fft_invertible_random_17(self):self._test_fft_invertible_random(2**17)
    def test_fft_invertible_random_18(self):self._test_fft_invertible_random(2**18)
    def test_fft_invertible_random_19(self):self._test_fft_invertible_random(2**19)
    def test_fft_invertible_random_20(self):self._test_fft_invertible_random(2**20)

    # confirm output value
    def test_fft_output_value(self):
        xr1 = np.arange(2**5, dtype=np.int32)
        xr2 = rfft(xr1)
        xr2_ = [496,  -23,  -19,  -14,  -17,  -17,  -14,  -14,  -16,  -16,  -16, -15,  -15,  -14,  -15,  -15,  -16,   -2,   -3,   -4,   -7,   -9,  -9,  -12,  -16,  -18,  -23,  -27,  -39,  -50,  -81, -164]
        self.assertTrue(np.all(xr2 == xr2_))

    # confirm intput type list
    def test_fft_input_list(self):
        xr1 = list(range(2**5))
        xr2 = rfft(xr1)
        xr2_ = [496,  -23,  -19,  -14,  -17,  -17,  -14,  -14,  -16,  -16,  -16, -15,  -15,  -14,  -15,  -15,  -16,   -2,   -3,   -4,   -7,   -9,  -9,  -12,  -16,  -18,  -23,  -27,  -39,  -50,  -81, -164]
        self.assertTrue(np.all(xr2 == xr2_))

    # confirm intput type list
    def test_ifft_input_list(self):
        xr1 = [496,  -23,  -19,  -14,  -17,  -17,  -14,  -14,  -16,  -16,  -16, -15,  -15,  -14,  -15,  -15,  -16,   -2,   -3,   -4,   -7,   -9,  -9,  -12,  -16,  -18,  -23,  -27,  -39,  -50,  -81, -164]
        xr2 = irfft(xr1)
        xr2_ = np.arange(2**5, dtype=np.int32)
        self.assertTrue(np.all(xr2 == xr2_))

    # confirm input with strides
    def test_fft_input_with_strides(self):
        xr1 = np.c_[np.arange(2**5, dtype=np.int32), np.arange(2**5, dtype=np.int32)].flatten()
        xr2 = rfft(xr1[::2])
        xr2_ = [496,  -23,  -19,  -14,  -17,  -17,  -14,  -14,  -16,  -16,  -16, -15,  -15,  -14,  -15,  -15,  -16,   -2,   -3,   -4,   -7,   -9,  -9,  -12,  -16,  -18,  -23,  -27,  -39,  -50,  -81, -164]
        self.assertTrue(np.all(xr2 == xr2_))

    # confirm input shape(error)
    def _test_fft_input_shape_error(self, shape1, regex):
        x1 = np.zeros(shape1, dtype=np.int32)
        with self.assertRaisesRegex(Exception, regex):
            _ = rfft(x1)
    def test_fft_input_error_unexpected_ndim_ar(self):self._test_fft_input_shape_error((2**10,1), "a\.ndim != 1")
    def test_fft_input_error_not_pow2(self):self._test_fft_input_shape_error((2**10+1,), "a.shape\(0\) is not a power of 2")

    # confirm input shape(error)
    def _test_ifft_input_shape_error(self, shape1, regex):
        x1 = np.zeros(shape1, dtype=np.int32)
        with self.assertRaisesRegex(Exception, regex):
            _ = irfft(x1)
    def test_ifft_input_error_unexpected_ndim_ar(self):self._test_ifft_input_shape_error((2**10,1), "a\.ndim != 1")
    def test_ifft_input_error_not_pow2(self):self._test_ifft_input_shape_error((2**10+1,), "a.shape\(0\) is not a power of 2")

if __name__ == '__main__':
    unittest.main(verbosity=2)
