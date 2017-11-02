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

#include <macros.h>

#ifdef __cplusplus
extern "C" {
#endif

#define bit_OSXSAVE (1 << 27)
#define bit_AVX		(1 << 28)
#define bit_FMA		(1 << 12)
#define bit_AVX2	0x00000020

struct isa_info 
{
	bool has_avx;
	bool has_fma3;
	bool has_avx2;
};

struct cache_info 
{
	uint32_t size;
	uint32_t associativity;
	uint32_t threads;
	bool inclusive;
};

struct cache_hierarchy_info 
{
	cache_info l1;
	cache_info l2;
	cache_info l3;
	cache_info l4;
};

struct cache_blocking_info 
{
	size_t l1;
	size_t l2;
	size_t l3;
	size_t l4;
};


#define NNP_COMPLEX_TUPLE_INDEX 1


typedef void (*nnp_transform_2d)(const float*, float*, size_t, size_t, uint32_t, uint32_t);
typedef void (*nnp_transform_2d_with_bias)(const float*, float*, const float*, size_t, size_t, uint32_t, uint32_t);
typedef void (*nnp_transform_2d_with_offset)(const float*, float*, size_t, size_t, uint32_t, uint32_t, uint32_t, uint32_t);

typedef void (*nnp_fast_sgemm_function)(size_t, size_t, const float*, const float*, float*, size_t);
typedef void (*nnp_full_sgemm_function)(uint32_t, uint32_t, size_t, size_t, const float*, const float*, float*, size_t);


typedef void (*nnp_fast_conv_function)(size_t, size_t, const float*, const float*, float*);
typedef void (*nnp_full_conv_function)(uint32_t, uint32_t, size_t, size_t, const float*, const float*, float*);

typedef void (*nnp_fast_tuple_gemm_function)(size_t, size_t, const float*, const float*, float*, size_t);
typedef void (*nnp_full_tuple_gemm_function)(uint32_t, uint32_t, size_t, size_t, const float*, const float*, float*, size_t);

typedef void (*nnp_sdotxf_function)(const float*, const float*, size_t, float*, size_t);
typedef void (*nnp_shdotxf_function)(const float*, const float*, size_t, float*, size_t);

typedef void (*nnp_relu_function)(const float*, float*, size_t, float);
typedef void (*nnp_inplace_relu_function)(float*, size_t, float);
typedef void (*nnp_grad_relu_function)(const float*, const float*, float*, size_t, float);

typedef void (*nnp_softmax_function)(size_t, const float*, float*);
typedef void (*nnp_inplace_softmax_function)(size_t, float*);

struct transforms 
{
	nnp_transform_2d_with_offset fft8x8_with_offset_and_store;
	nnp_transform_2d_with_offset fft8x8_with_offset_and_stream;
	nnp_transform_2d_with_offset ifft8x8_with_offset;
	nnp_transform_2d_with_bias ifft8x8_with_bias;
	nnp_transform_2d_with_bias ifft8x8_with_bias_with_relu;
	nnp_transform_2d_with_offset fft16x16_with_offset_and_store;
	nnp_transform_2d_with_offset fft16x16_with_offset_and_stream;
	nnp_transform_2d_with_offset ifft16x16_with_offset;
	nnp_transform_2d_with_bias ifft16x16_with_bias;
	nnp_transform_2d_with_bias ifft16x16_with_bias_with_relu;
	nnp_transform_2d_with_offset iwt_f6x6_3x3_with_offset_and_store;
	nnp_transform_2d_with_offset iwt_f6x6_3x3_with_offset_and_stream;
	nnp_transform_2d_with_offset kwt_f6x6_3x3;
	nnp_transform_2d_with_offset kwt_f6x6_3Rx3R;
	nnp_transform_2d_with_offset owt_f6x6_3x3;
	nnp_transform_2d_with_bias owt_f6x6_3x3_with_bias;
	nnp_transform_2d_with_bias owt_f6x6_3x3_with_bias_with_relu;
};

struct activations 
{
	nnp_relu_function relu;
	nnp_inplace_relu_function inplace_relu;
	nnp_grad_relu_function grad_relu;
	nnp_softmax_function softmax;
	nnp_inplace_softmax_function inplace_softmax;
};

struct convolution 
{
	nnp_fast_conv_function only_mr_x_nr;
	nnp_full_conv_function upto_mr_x_nr;
	uint32_t mr;
	uint32_t nr;
};

struct sgemm 
{
	nnp_fast_sgemm_function only_mr_x_nr;
	nnp_full_sgemm_function upto_mr_x_nr;
	uint32_t mr;
	uint32_t nr;
};

struct sxgemm 
{
	nnp_fast_tuple_gemm_function only_mr_x_nr;
	nnp_full_tuple_gemm_function upto_mr_x_nr;
	uint32_t mr;
	uint32_t nr;
};

struct cxgemm 
{
	nnp_fast_tuple_gemm_function s4cX_only_mr_x_nr;
	nnp_full_tuple_gemm_function s4cX_upto_mr_x_nr;
	nnp_fast_tuple_gemm_function cX_only_mr_x_nr;
	nnp_full_tuple_gemm_function cX_upto_mr_x_nr;
	nnp_fast_tuple_gemm_function s4cX_conjb_only_mr_x_nr;
	nnp_full_tuple_gemm_function s4cX_conjb_upto_mr_x_nr;
	nnp_fast_tuple_gemm_function cX_conjb_only_mr_x_nr;
	nnp_full_tuple_gemm_function cX_conjb_upto_mr_x_nr;
	nnp_fast_tuple_gemm_function s4cX_conjb_transc_only_mr_x_nr;
	nnp_full_tuple_gemm_function s4cX_conjb_transc_upto_mr_x_nr;
	nnp_fast_tuple_gemm_function cX_conjb_transc_only_mr_x_nr;
	nnp_full_tuple_gemm_function cX_conjb_transc_upto_mr_x_nr;
	uint32_t mr;
	uint32_t nr;
};

struct sdotxf 
{
	const nnp_sdotxf_function* functions;
	uint32_t fusion;
};

struct shdotxf 
{
	const nnp_shdotxf_function* functions;
	uint32_t fusion;
};

struct hardware_info 
{
	bool initialized;
	bool supported;
	uint32_t simd_width;

	cache_hierarchy_info cache;
	cache_blocking_info blocking;

	transforms transforms;
	activations activations;
	convolution conv1x1;
	sgemm sgemm;
	sxgemm sxgemm;
	cxgemm cxgemm;
	sdotxf sdotxf;
	shdotxf shdotxf;

	isa_info isa;
};

extern hardware_info nnp_hwinfo;

#ifdef __cplusplus
} /* extern "C" */
#endif
