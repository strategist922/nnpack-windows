#pragma once

#if defined(__cplusplus) && (__cplusplus >= 201103L)
#include <cstddef>
#include <cstdint>
#include <cstdbool>
#else
#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

	// AVX2

	void nnp_fft8x8_with_offset_and_store__avx2(const float t[], float f[], size_t stride_t, size_t stride_f, uint32_t row_count, uint32_t column_count, uint32_t row_offset, uint32_t column_offset);
	void nnp_fft8x8_with_offset_and_stream__avx2(const float t[], float f[], size_t stride_t, size_t stride_f, uint32_t row_count, uint32_t column_count, uint32_t row_offset, uint32_t column_offset);
	void nnp_ifft8x8_with_offset__avx2(const float f[], float t[], size_t stride_f, size_t stride_t, uint32_t row_count, uint32_t column_count, uint32_t row_offset, uint32_t column_offset);
	void nnp_ifft8x8__avx2(const float f[], float t[], size_t stride_f, size_t stride_t, uint32_t row_count, uint32_t column_count);
	void nnp_ifft8x8_with_relu__avx2(const float f[], float t[], size_t stride_f, size_t stride_t, uint32_t row_count, uint32_t column_count);
	void nnp_ifft8x8_with_bias__avx2(const float f[], float t[], const float bias[], size_t stride_f, size_t stride_t, uint32_t row_count, uint32_t column_count);
	void nnp_ifft8x8_with_bias_with_relu__avx2(const float f[], float t[], const float bias[], size_t stride_f, size_t stride_t, uint32_t row_count, uint32_t column_count);

	void nnp_fft16x16_with_offset_and_store__avx2(const float t[], float f[], size_t stride_t, size_t stride_f, uint32_t row_count, uint32_t column_count, uint32_t row_offset, uint32_t column_offset);
	void nnp_fft16x16_with_offset_and_stream__avx2(const float t[], float f[], size_t stride_t, size_t stride_f, uint32_t row_count, uint32_t column_count, uint32_t row_offset, uint32_t column_offset);
	void nnp_ifft16x16_with_offset__avx2(const float f[], float t[], size_t stride_f, size_t stride_t, uint32_t row_count, uint32_t column_count, uint32_t row_offset, uint32_t column_offset);
	void nnp_ifft16x16__avx2(const float f[], float t[], size_t stride_f, size_t stride_t, uint32_t row_count, uint32_t column_count);
	void nnp_ifft16x16_with_relu__avx2(const float f[], float t[], size_t stride_f, size_t stride_t, uint32_t row_count, uint32_t column_count);
	void nnp_ifft16x16_with_bias__avx2(const float f[], float t[], const float bias[], size_t stride_f, size_t stride_t, uint32_t row_count, uint32_t column_count);
	void nnp_ifft16x16_with_bias_with_relu__avx2(const float f[], float t[], const float bias[], size_t stride_f, size_t stride_t, uint32_t row_count, uint32_t column_count);

	void nnp_iwt8x8_3x3_with_offset_and_store__avx2(const float d[], float wd[], size_t stride_d, size_t stride_wd, uint32_t row_count, uint32_t column_count, uint32_t row_offset, uint32_t column_offset);
	void nnp_iwt8x8_3x3_with_offset_and_stream__avx2(const float d[], float wd[], size_t stride_d, size_t stride_wd, uint32_t row_count, uint32_t column_count, uint32_t row_offset, uint32_t column_offset);
	void nnp_kwt8x8_3x3_and_store__avx2(const float g[], float wg[], size_t stride_g, size_t stride_wg, uint32_t, uint32_t, uint32_t, uint32_t);
	void nnp_kwt8x8_3x3_and_stream__avx2(const float g[], float wg[], size_t stride_g, size_t stride_wg, uint32_t, uint32_t, uint32_t, uint32_t);
	void nnp_kwt8x8_3Rx3R_and_store__avx2(const float g[], float wg[], size_t stride_g, size_t stride_wg, uint32_t, uint32_t, uint32_t, uint32_t);
	void nnp_kwt8x8_3Rx3R_and_stream__avx2(const float g[], float wg[], size_t stride_g, size_t stride_wg, uint32_t, uint32_t, uint32_t, uint32_t);
	void nnp_owt8x8_3x3__avx2(const float m[], float s[], size_t stride_m, size_t stride_s, uint32_t row_count, uint32_t column_count, uint32_t, uint32_t);
	void nnp_owt8x8_3x3_with_relu__avx2(const float m[], float s[], size_t stride_m, size_t stride_s, uint32_t row_count, uint32_t column_count, uint32_t, uint32_t);
	void nnp_owt8x8_3x3_with_bias__avx2(const float m[], float s[], const float bias[], size_t stride_m, size_t stride_s, uint32_t row_count, uint32_t column_count);
	void nnp_owt8x8_3x3_with_bias_with_relu__avx2(const float m[], float s[], const float bias[], size_t stride_m, size_t stride_s, uint32_t row_count, uint32_t column_count);


	// Scalar

	void nnp_fft8x8_with_offset__scalar(const float t[], float f[], size_t stride_t, size_t stride_f, uint32_t row_count, uint32_t column_count, uint32_t row_offset, uint32_t column_offset);
	void nnp_ifft8x8_with_offset__scalar(const float t[], float f[], size_t stride_t, size_t stride_f, uint32_t row_count, uint32_t column_count, uint32_t row_offset, uint32_t column_offset);
	void nnp_ifft8x8_with_bias__scalar(const float f[], float t[], const float bias[], size_t stride_f, size_t stride_t, uint32_t row_count, uint32_t column_count);
	void nnp_ifft8x8_with_bias_with_relu__scalar(const float f[], float t[], const float bias[], size_t stride_f, size_t stride_t, uint32_t row_count, uint32_t column_count);

	void nnp_fft16x16_with_offset__scalar(const float t[], float f[], size_t stride_t, size_t stride_f, uint32_t row_count, uint32_t column_count, uint32_t row_offset, uint32_t column_offset);
	void nnp_ifft16x16_with_offset__scalar(const float f[], float t[], size_t stride_f, size_t stride_t, uint32_t row_count, uint32_t column_count, uint32_t row_offset, uint32_t column_offset);
	void nnp_ifft16x16_with_bias__scalar(const float f[], float t[], const float bias[], size_t stride_f, size_t stride_t, uint32_t row_count, uint32_t column_count);
	void nnp_ifft16x16_with_bias_with_relu__scalar(const float f[], float t[], const float bias[], size_t stride_f, size_t stride_t, uint32_t row_count, uint32_t column_count);

	void nnp_iwt8x8_3x3_with_offset__scalar(const float d[], float wd[], size_t stride_d, size_t stride_wd, uint32_t row_count, uint32_t column_count, uint32_t row_offset, uint32_t column_offset);
	void nnp_kwt8x8_3x3__scalar(const float g[], float wg[], size_t stride_g, size_t stride_wg, uint32_t, uint32_t, uint32_t, uint32_t);
	void nnp_kwt8x8_3Rx3R__scalar(const float g[], float wg[], size_t stride_g, size_t stride_wg, uint32_t, uint32_t, uint32_t, uint32_t);
	void nnp_owt8x8_3x3__scalar(const float m[], float s[], size_t stride_m, size_t stride_s, uint32_t row_count, uint32_t column_count, uint32_t, uint32_t);
	void nnp_owt8x8_3x3_with_bias__scalar(const float m[], float s[], const float bias[], size_t stride_m, size_t stride_s, uint32_t row_count, uint32_t column_count);
	void nnp_owt8x8_3x3_with_bias_with_relu__scalar(const float m[], float s[], const float bias[], size_t stride_m, size_t stride_s, uint32_t row_count, uint32_t column_count);

#ifdef __cplusplus
} /* extern "C" */
#endif


