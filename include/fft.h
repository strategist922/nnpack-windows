#pragma once

#ifdef __cplusplus
#include <cstddef>
#include <cstdint>
#include <cstdbool>
#else
#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>
#endif

/* Reference versions */

typedef void (*nnp_strided_fft_function)(const float*, size_t, float*, size_t);
typedef void (*nnp_fft_function)(const float*, float*);

#ifdef __cplusplus  
extern "C" {  // only need to export C interface if  
			  // used by C++ source code  
#endif 
/* Forward FFT within rows with SOA layout: used in the horizontal phase of 2D FFT */
void nnp_fft8_soa__avx2(const float* t, float* f);
void nnp_fft16_soa__avx2(const float* t, float* f);

/* Inverse FFT within rows with SOA layout: used in the horizontal phase of 2D IFFT */
void nnp_ifft8_soa__avx2(const float* f, float* t);
void nnp_ifft16_soa__avx2(const float* f, float* t);

/* Forward FFT across rows with SIMD AOS layout: used in the vertical phase of 2D FFT */
void nnp_fft4_8aos__fma3(const float* t, float* f);
void nnp_fft8_8aos__fma3(const float* t, float* f);

/* Inverse FFT across rows with SIMD AOS layout: used in the vertical phase of 2D IFFT */
void nnp_ifft8_8aos__fma3(const float* f, float* t);

/* Forward real-to-complex FFT across rows with SIMD layout: used in the vertical phase of 2D FFT */
void nnp_fft8_8real__fma3(const float* t, float* f);
void nnp_fft16_8real__fma3(const float* t, float* f);

/* Inverse complex-to-real FFT across rows with SIMD layout: used in the vertical phase of 2D IFFT */
void nnp_ifft8_8real__fma3(const float* f, float* t);
void nnp_ifft16_8real__fma3(const float* f, float* t);

void nnp_fft8_dualreal__avx2(const float* t, float* f);
void nnp_fft16_dualreal__avx2(const float* t, float* f);

void nnp_ifft8_dualreal__avx2(const float* f, float* t);
void nnp_ifft16_dualreal__avx2(const float* f, float* t);

#ifdef __cplusplus  
}
#endif 