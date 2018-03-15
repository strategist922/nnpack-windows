#pragma once
#ifndef PSIMD_H
#define PSIMD_H

#include "vectorclass.h"

using psimd_s8 = Vec16c;
using psimd_u8 = Vec16uc;
using psimd_s16 = Vec8s;
using psimd_u16 = Vec8us;
using psimd_s32 = Vec4i;
using psimd_u32 = Vec4ui;
using psimd_f32 = Vec4f;

#if defined(__CUDA_ARCH__)
	/* CUDA compiler */
	#define PSIMD_INTRINSIC __forceinline__ __device__
#elif defined(__OPENCL_VERSION__)
	/* OpenCL compiler */
	#define PSIMD_INTRINSIC inline static
#elif defined(__INTEL_COMPILER)
	/* Intel compiler, even on Windows */
	#define PSIMD_INTRINSIC inline static __attribute__((__always_inline__))
#elif defined(__GNUC__)
	/* GCC-compatible compiler (gcc/clang/icc) */
	#define PSIMD_INTRINSIC inline static __attribute__((__always_inline__))
#elif defined(_MSC_VER)
	/* MSVC-compatible compiler (cl/icl/clang-cl) */
	#define PSIMD_INTRINSIC __forceinline static
#elif defined(__cplusplus)
	/* Generic C++ compiler */
	#define PSIMD_INTRINSIC inline static
#elif defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 199901L)
	/* Generic C99 compiler */
	#define PSIMD_INTRINSIC inline static
#else
	/* Generic C compiler */
	#define PSIMD_INTRINSIC static
#endif

#if defined(__ARM_NEON__)
	#include <arm_neon.h>
#endif

#if defined(__cplusplus)
	#define PSIMD_CXX_SYNTAX
#elif defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 201112L)
	#define PSIMD_C11_SYNTAX
#elif defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 199901L)
	#define PSIMD_C99_SYNTAX
#else
	#define PSIMD_C89_SYNTAX
#endif

#if defined(__cplusplus) && (__cplusplus >= 201103L)
	#include <cstddef>
	#include <cstdint>
#elif !defined(__OPENCL_VERSION__)
	#include <stddef.h>
	#include <stdint.h>
#endif

#if defined(__GNUC__) || defined(_MSC_VER)
	#define PSIMD_HAVE_F64 0
	#define PSIMD_HAVE_F32 1
	#define PSIMD_HAVE_U8 1
	#define PSIMD_HAVE_S8 1
	#define PSIMD_HAVE_U16 1
	#define PSIMD_HAVE_S16 1
	#define PSIMD_HAVE_U32 1
	#define PSIMD_HAVE_S32 1
	#define PSIMD_HAVE_U64 0
	#define PSIMD_HAVE_S64 0
    
	typedef struct {
		psimd_s8 lo;
		psimd_s8 hi;
	} psimd_s8x2;

	typedef struct {
		psimd_u8 lo;
		psimd_u8 hi;
	} psimd_u8x2;

	typedef struct {
		psimd_s16 lo;
		psimd_s16 hi;
	} psimd_s16x2;

	typedef struct {
		psimd_u16 lo;
		psimd_u16 hi;
	} psimd_u16x2;

	typedef struct {
		psimd_s32 lo;
		psimd_s32 hi;
	} psimd_s32x2;

	typedef struct {
		psimd_u32 lo;
		psimd_u32 hi;
	} psimd_u32x2;

	typedef struct {
		psimd_f32 lo;
		psimd_f32 hi;
	} psimd_f32x2;

	/* Bit casts */
	PSIMD_INTRINSIC psimd_u32x2 psimd_cast_s32x2_u32x2(psimd_s32x2 v) {
		return psimd_u32x2 { psimd_u32(v.lo), psimd_u32(v.hi) };
	}

	PSIMD_INTRINSIC psimd_f32x2 psimd_cast_s32x2_f32x2(psimd_s32x2 v) {
		return psimd_f32x2 { to_float(v.lo), to_float(v.hi) };
	}

	PSIMD_INTRINSIC psimd_s32x2 psimd_cast_u32x2_s32x2(psimd_u32x2 v) {
		return psimd_s32x2 { psimd_s32(v.lo), psimd_s32(v.hi) };
	}

	PSIMD_INTRINSIC psimd_f32x2 psimd_cast_u32x2_f32x2(psimd_u32x2 v) {
		return psimd_f32x2 { to_float(v.lo), to_float(v.hi) };
	}

	PSIMD_INTRINSIC psimd_s32x2 psimd_cast_f32x2_s32x2(psimd_f32x2 v) {
		return psimd_s32x2 { truncate_to_int(v.lo), truncate_to_int(v.hi) };
	}

	PSIMD_INTRINSIC psimd_u32x2 psimd_cast_f32x2_u32x2(psimd_f32x2 v) {
		return psimd_u32x2 { psimd_u32(truncate_to_int(v.lo)), psimd_u32(truncate_to_int(v.hi)) };
	}

	/* Swap */
	PSIMD_INTRINSIC void psimd_swap_s8(psimd_s8* a, psimd_s8* b) {
		const psimd_s8 new_a = *b;
		const psimd_s8 new_b = *a;
		*a = new_a;
		*b = new_b;
	}

	PSIMD_INTRINSIC void psimd_swap_u8(psimd_u8* a, psimd_u8* b) {
		const psimd_u8 new_a = *b;
		const psimd_u8 new_b = *a;
		*a = new_a;
		*b = new_b;
	}

	PSIMD_INTRINSIC void psimd_swap_s16(psimd_s16* a, psimd_s16* b) {
		const psimd_s16 new_a = *b;
		const psimd_s16 new_b = *a;
		*a = new_a;
		*b = new_b;
	}

	PSIMD_INTRINSIC void psimd_swap_u16(psimd_u16* a, psimd_u16* b) {
		const psimd_u16 new_a = *b;
		const psimd_u16 new_b = *a;
		*a = new_a;
		*b = new_b;
	}

	PSIMD_INTRINSIC void psimd_swap_s32(psimd_s32* a, psimd_s32* b) {
		const psimd_s32 new_a = *b;
		const psimd_s32 new_b = *a;
		*a = new_a;
		*b = new_b;
	}

	PSIMD_INTRINSIC void psimd_swap_u32(psimd_u32* a, psimd_u32* b) {
		const psimd_u32 new_a = *b;
		const psimd_u32 new_b = *a;
		*a = new_a;
		*b = new_b;
	}

	PSIMD_INTRINSIC void psimd_swap_f32(psimd_f32* a, psimd_f32* b) {
		const psimd_f32 new_a = *b;
		const psimd_f32 new_b = *a;
		*a = new_a;
		*b = new_b;
	}

	/* Zero-initialization */
	PSIMD_INTRINSIC psimd_s8 psimd_zero_s8() {
		return psimd_s8(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
	}

	PSIMD_INTRINSIC psimd_u8 psimd_zero_u8() {
		return psimd_u8(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
	}

	PSIMD_INTRINSIC psimd_s16 psimd_zero_s16() {
		return psimd_s16(0, 0, 0, 0, 0, 0, 0, 0);
	}

	PSIMD_INTRINSIC psimd_u16 psimd_zero_u16() {
		return psimd_u16(0, 0, 0, 0, 0, 0, 0, 0);
	}

	PSIMD_INTRINSIC psimd_s32 psimd_zero_s32() {
		return psimd_s32(0, 0, 0, 0);
	}

	PSIMD_INTRINSIC psimd_u32 psimd_zero_u32() {
		return psimd_u32(0, 0, 0, 0);
	}

	PSIMD_INTRINSIC psimd_f32 psimd_zero_f32() {
		return psimd_f32(0.0f, 0.0f, 0.0f, 0.0f);
	}

	/* Initialization to the same constant */
	PSIMD_INTRINSIC psimd_s8 psimd_splat_s8(int8_t c) {
		return psimd_s8(c, c, c, c, c, c, c, c, c, c, c, c, c, c, c, c);
	}

	PSIMD_INTRINSIC psimd_u8 psimd_splat_u8(uint8_t c) {
		return psimd_u8(c, c, c, c, c, c, c, c, c, c, c, c, c, c, c, c);
	}

	PSIMD_INTRINSIC psimd_s16 psimd_splat_s16(int16_t c) {
		return psimd_s16(c, c, c, c, c, c, c, c);
	}

	PSIMD_INTRINSIC psimd_u16 psimd_splat_u16(uint16_t c) {
		return psimd_u16(c, c, c, c, c, c, c, c);
	}

	PSIMD_INTRINSIC psimd_s32 psimd_splat_s32(int32_t c) {
		return psimd_s32(c, c, c, c);
	}

	PSIMD_INTRINSIC psimd_u32 psimd_splat_u32(uint32_t c) {
		return psimd_u32(c, c, c, c);
	}

	PSIMD_INTRINSIC psimd_f32 psimd_splat_f32(float c) {
		return psimd_f32(c, c, c, c);
	}

	/* Load vector */
	PSIMD_INTRINSIC psimd_s8 psimd_load_s8(const void* address) {
		return *((const psimd_s8*) address);
	}

	PSIMD_INTRINSIC psimd_u8 psimd_load_u8(const void* address) {
		return *((const psimd_u8*) address);
	}

	PSIMD_INTRINSIC psimd_s16 psimd_load_s16(const void* address) {
		return *((const psimd_s16*) address);
	}

	PSIMD_INTRINSIC psimd_u16 psimd_load_u16(const void* address) {
		return *((const psimd_u16*) address);
	}

	PSIMD_INTRINSIC psimd_s32 psimd_load_s32(const void* address) {
		return *((const psimd_s32*) address);
	}

	PSIMD_INTRINSIC psimd_u32 psimd_load_u32(const void* address) {
		return *((const psimd_u32*) address);
	}

	PSIMD_INTRINSIC psimd_f32 psimd_load_f32(const void* address) {
		return *((const psimd_f32*) address);
	}

	PSIMD_INTRINSIC psimd_f32 psimd_load1_f32(const void* address) {
		return psimd_f32(*((const float*) address), 0.0f, 0.0f, 0.0f);
	}

	PSIMD_INTRINSIC psimd_f32 psimd_load2_f32(const void* address) {
		const float* address_f32 = (const float*) address;
		return psimd_f32(address_f32[0], address_f32[1], 0.0f, 0.0f);
	}

	PSIMD_INTRINSIC psimd_f32 psimd_load3_f32(const void* address) {
		const float* address_f32 = (const float*) address;
		return psimd_f32(address_f32[0], address_f32[1], address_f32[2], 0.0f);
	}

	PSIMD_INTRINSIC psimd_f32 psimd_load4_f32(const void* address) {
		return psimd_load_f32(address);
	}

	PSIMD_INTRINSIC psimd_f32 psimd_load_stride2_f32(const void* address) {
		const psimd_f32 v0x1x = psimd_load_f32(address);
		const psimd_f32 vx2x3 = psimd_load_f32((const float*) address + 3);
		#if defined(__clang__)
			return __builtin_shufflevector(v0x1x, vx2x3, 0, 2, 5, 7);
		#else
			return _mm_shuffle_ps(v0x1x, vx2x3, _MM_SHUFFLE(0,2,5,7));
		#endif
	}

	PSIMD_INTRINSIC psimd_f32 psimd_load1_stride2_f32(const void* address) {
		return psimd_load_f32(address);
	}

	PSIMD_INTRINSIC psimd_f32 psimd_load2_stride2_f32(const void* address) {
		const float* address_f32 = (const float*) address;
		return psimd_f32(address_f32[0], address_f32[2], 0.0f, 0.0f);
	}

	PSIMD_INTRINSIC psimd_f32 psimd_load3_stride2_f32(const void* address) {
		const psimd_f32 v0x1x = psimd_load_f32(address);
		const psimd_f32 v2zzz = psimd_load1_f32((const float*) address + 2);
		#if defined(__clang__)
			return __builtin_shufflevector(v0x1x, v2zzz, 0, 2, 4, 6);
		#else
			return _mm_shuffle_ps(v0x1x, v2zzz, _MM_SHUFFLE(0,2,4,6));
		#endif
	}

	PSIMD_INTRINSIC psimd_f32 psimd_load4_stride2_f32(const void* address) {
		return psimd_load_stride2_f32(address);
	}

	PSIMD_INTRINSIC psimd_f32 psimd_load_stride_f32(const void* address, size_t stride) {
		const float* address0_f32 = (const float*) address;
		const float* address1_f32 = address0_f32 + stride;
		const float* address2_f32 = address1_f32 + stride;
		const float* address3_f32 = address2_f32 + stride;
		return psimd_f32(*address0_f32, *address1_f32, *address2_f32, *address3_f32);
	}

	PSIMD_INTRINSIC psimd_f32 psimd_load1_stride_f32(const void* address, size_t stride) {
		return psimd_load1_f32(address);
	}

	PSIMD_INTRINSIC psimd_f32 psimd_load2_stride_f32(const void* address, size_t stride) {
		const float* address_f32 = (const float*) address;
		return psimd_f32(address_f32[0], address_f32[stride], 0.0f, 0.0f);
	}

	PSIMD_INTRINSIC psimd_f32 psimd_load3_stride_f32(const void* address, size_t stride) {
		const float* address0_f32 = (const float*) address;
		const float* address1_f32 = address0_f32 + stride;
		const float* address2_f32 = address1_f32 + stride;
		return psimd_f32(*address0_f32, *address1_f32, *address2_f32, 0.0f);
	}

	PSIMD_INTRINSIC psimd_f32 psimd_load4_stride_f32(const void* address, size_t stride) {
		return psimd_load_stride_f32(address, stride);
	}

	/* Store vector */
	PSIMD_INTRINSIC void psimd_store_s8(void* address, psimd_s8 value) {
		*((psimd_s8*) address) = value;
	}

	PSIMD_INTRINSIC void psimd_store_u8(void* address, psimd_u8 value) {
		*((psimd_u8*) address) = value;
	}

	PSIMD_INTRINSIC void psimd_store_s16(void* address, psimd_s16 value) {
		*((psimd_s16*) address) = value;
	}

	PSIMD_INTRINSIC void psimd_store_u16(void* address, psimd_u16 value) {
		*((psimd_u16*) address) = value;
	}

	PSIMD_INTRINSIC void psimd_store_s32(void* address, psimd_s32 value) {
		*((psimd_s32*) address) = value;
	}

	PSIMD_INTRINSIC void psimd_store_u32(void* address, psimd_u32 value) {
		*((psimd_u32*) address) = value;
	}

	PSIMD_INTRINSIC void psimd_store_f32(void* address, psimd_f32 value) {
		*((psimd_f32*) address) = value;
	}

	PSIMD_INTRINSIC void psimd_store1_f32(void* address, psimd_f32 value) {
		*((float*) address) = value[0];
	}

	PSIMD_INTRINSIC void psimd_store2_f32(void* address, psimd_f32 value) {
		float* address_f32 = (float*) address;
		address_f32[0] = value[0];
		address_f32[1] = value[1];
	}

	PSIMD_INTRINSIC void psimd_store3_f32(void* address, psimd_f32 value) {
		float* address_f32 = (float*) address;
		address_f32[0] = value[0];
		address_f32[1] = value[1];
		address_f32[2] = value[2];
	}

	PSIMD_INTRINSIC void psimd_store4_f32(void* address, psimd_f32 value) {
		psimd_store_f32(address, value);
	}

	PSIMD_INTRINSIC void psimd_store_stride_f32(void* address, size_t stride, psimd_f32 value) {
		float* address0_f32 = (float*) address;
		float* address1_f32 = address0_f32 + stride;
		float* address2_f32 = address1_f32 + stride;
		float* address3_f32 = address2_f32 + stride;
		*address0_f32 = value[0];
		*address1_f32 = value[1];
		*address2_f32 = value[2];
		*address3_f32 = value[3];
	}

	PSIMD_INTRINSIC void psimd_store1_stride_f32(void* address, size_t stride, psimd_f32 value) {
		psimd_store1_f32(address, value);
	}

	PSIMD_INTRINSIC void psimd_store2_stride_f32(void* address, size_t stride, psimd_f32 value) {
		float* address_f32 = (float*) address;
		address_f32[0]      = value[0];
		address_f32[stride] = value[1];
	}

	PSIMD_INTRINSIC void psimd_store3_stride_f32(void* address, size_t stride, psimd_f32 value) {
		float* address0_f32 = (float*) address;
		float* address1_f32 = address0_f32 + stride;
		float* address2_f32 = address1_f32 + stride;
		*address0_f32 = value[0];
		*address1_f32 = value[1];
		*address2_f32 = value[2];
	}

	/* Vector addition */
	PSIMD_INTRINSIC psimd_s8 psimd_add_s8(psimd_s8 a, psimd_s8 b) {
		return a + b;
	}

	PSIMD_INTRINSIC psimd_u8 psimd_add_u8(psimd_u8 a, psimd_u8 b) {
		return a + b;
	}

	PSIMD_INTRINSIC psimd_s16 psimd_add_s16(psimd_s16 a, psimd_s16 b) {
		return a + b;
	}

	PSIMD_INTRINSIC psimd_u16 psimd_add_u16(psimd_u16 a, psimd_u16 b) {
		return a + b;
	}

	PSIMD_INTRINSIC psimd_s32 psimd_add_s32(psimd_s32 a, psimd_s32 b) {
		return a + b;
	}

	PSIMD_INTRINSIC psimd_u32 psimd_add_u32(psimd_u32 a, psimd_u32 b) {
		return a + b;
	}

	PSIMD_INTRINSIC psimd_f32 psimd_add_f32(psimd_f32 a, psimd_f32 b) {
		#if defined(__ARM_ARCH_7A__) && defined(__ARM_NEON__) && !defined(__FAST_MATH__)
			return (psimd_f32) vaddq_f32((float32x4_t) a, (float32x4_t) b);
		#else
			return a + b;
		#endif
	}

	/* Vector subtraction */
	PSIMD_INTRINSIC psimd_s8 psimd_sub_s8(psimd_s8 a, psimd_s8 b) {
		return a - b;
	}

	PSIMD_INTRINSIC psimd_u8 psimd_sub_u8(psimd_u8 a, psimd_u8 b) {
		return a - b;
	}

	PSIMD_INTRINSIC psimd_s16 psimd_sub_s16(psimd_s16 a, psimd_s16 b) {
		return a - b;
	}

	PSIMD_INTRINSIC psimd_u16 psimd_sub_u16(psimd_u16 a, psimd_u16 b) {
		return a - b;
	}

	PSIMD_INTRINSIC psimd_s32 psimd_sub_s32(psimd_s32 a, psimd_s32 b) {
		return a - b;
	}

	PSIMD_INTRINSIC psimd_u32 psimd_sub_u32(psimd_u32 a, psimd_u32 b) {
		return a - b;
	}

	PSIMD_INTRINSIC psimd_f32 psimd_sub_f32(psimd_f32 a, psimd_f32 b) {
		#if defined(__ARM_ARCH_7A__) && defined(__ARM_NEON__) && !defined(__FAST_MATH__)
			return (psimd_f32) vsubq_f32((float32x4_t) a, (float32x4_t) b);
		#else
			return a - b;
		#endif
	}

	/* Vector multiplication */
	PSIMD_INTRINSIC psimd_s8 psimd_mul_s8(psimd_s8 a, psimd_s8 b) {
		return a * b;
	}

	PSIMD_INTRINSIC psimd_u8 psimd_mul_u8(psimd_u8 a, psimd_u8 b) {
		return a * b;
	}

	PSIMD_INTRINSIC psimd_s16 psimd_mul_s16(psimd_s16 a, psimd_s16 b) {
		return a * b;
	}

	PSIMD_INTRINSIC psimd_u16 psimd_mul_u16(psimd_u16 a, psimd_u16 b) {
		return a * b;
	}

	PSIMD_INTRINSIC psimd_s32 psimd_mul_s32(psimd_s32 a, psimd_s32 b) {
		return a * b;
	}

	PSIMD_INTRINSIC psimd_u32 psimd_mul_u32(psimd_u32 a, psimd_u32 b) {
		return a * b;
	}

	PSIMD_INTRINSIC psimd_f32 psimd_mul_f32(psimd_f32 a, psimd_f32 b) {
		#if defined(__ARM_ARCH_7A__) && defined(__ARM_NEON__) && !defined(__FAST_MATH__)
			return (psimd_f32) vmulq_f32((float32x4_t) a, (float32x4_t) b);
		#else
			return a * b;
		#endif
	}

	/* Vector and */
	PSIMD_INTRINSIC psimd_f32 psimd_andmask_f32(psimd_s32 mask, psimd_f32 v) {
		return to_float(mask & round_to_int(v));
	}

	/* Vector blend */
	PSIMD_INTRINSIC psimd_s8 psimd_blend_s8(psimd_s8 mask, psimd_s8 a, psimd_s8 b) {
		#if defined(__ARM_NEON__)
			return (psimd_s8) vbslq_s8((uint8x16_t) mask, (int8x16_t) a, (int8x16_t) b);
		#else
			return (mask & a) | (~mask & b);
		#endif
	}

	PSIMD_INTRINSIC psimd_u8 psimd_blend_u8(psimd_u8 mask, psimd_u8 a, psimd_u8 b) {
		#if defined(__ARM_NEON__)
			return (psimd_u8) vbslq_u8((uint8x16_t) mask, (uint8x16_t) a, (uint8x16_t) b);
		#else
			return (mask & a) | (~mask & b);
		#endif
	}
	
	PSIMD_INTRINSIC psimd_s16 psimd_blend_s16(psimd_s16 mask, psimd_s16 a, psimd_s16 b) {
		#if defined(__ARM_NEON__)
			return (psimd_s16) vbslq_s16((uint16x8_t) mask, (int16x8_t) a, (int16x8_t) b);
		#else
			return (mask & a) | (~mask & b);
		#endif
	}

	PSIMD_INTRINSIC psimd_u16 psimd_blend_u16(psimd_u16 mask, psimd_u16 a, psimd_u16 b) {
		#if defined(__ARM_NEON__)
			return (psimd_u16) vbslq_u16((uint16x8_t) mask, (uint16x8_t) a, (uint16x8_t) b);
		#else
			return (mask & a) | (~mask & b);
		#endif
	}
	
	PSIMD_INTRINSIC psimd_s32 psimd_blend_s32(psimd_s32 mask, psimd_s32 a, psimd_s32 b) {
		#if defined(__ARM_NEON__)
			return (psimd_s32) vbslq_s32((uint32x4_t) mask, (int32x4_t) a, (int32x4_t) b);
		#else
			return (mask & a) | (~mask & b);
		#endif
	}

	PSIMD_INTRINSIC psimd_u32 psimd_blend_u32(psimd_u32 mask, psimd_u32 a, psimd_u32 b) {
		#if defined(__ARM_NEON__)
			return (psimd_u32) vbslq_u32((uint32x4_t) mask, (uint32x4_t) a, (uint32x4_t) b);
		#else
			return (mask & a) | (~mask & b);
		#endif
	}
	
	PSIMD_INTRINSIC psimd_f32 psimd_blend_f32(psimd_s32 mask, psimd_f32 a, psimd_f32 b) {
		#if defined(__ARM_NEON__)
			return (psimd_f32) vbslq_f32((uint32x4_t) mask, (float32x4_t) a, (float32x4_t) b);
		#else
			return to_float(psimd_blend_s32(mask, truncate_to_int(a), truncate_to_int(b)));
		#endif
	}
	
	/* Vector blend on sign */
	PSIMD_INTRINSIC psimd_s8 psimd_signblend_s8(psimd_s8 x, psimd_s8 a, psimd_s8 b) {
		return psimd_blend_s8(x >> 7, a, b);
	}

	PSIMD_INTRINSIC psimd_u8 psimd_signblend_u8(psimd_s8 x, psimd_u8 a, psimd_u8 b) {
		return psimd_blend_u8(psimd_u8(x >> 7), a, b);
	}

	PSIMD_INTRINSIC psimd_s16 psimd_signblend_s16(psimd_s16 x, psimd_s16 a, psimd_s16 b) {
		return psimd_blend_s16(x >> 15, a, b);
	}

	PSIMD_INTRINSIC psimd_u16 psimd_signblend_u16(psimd_s16 x, psimd_u16 a, psimd_u16 b) {
		return psimd_blend_u16(psimd_u16(x >> 15), a, b);
	}

	PSIMD_INTRINSIC psimd_s32 psimd_signblend_s32(psimd_s32 x, psimd_s32 a, psimd_s32 b) {
		return psimd_blend_s32(x >> 31, a, b);
	}

	PSIMD_INTRINSIC psimd_u32 psimd_signblend_u32(psimd_s32 x, psimd_u32 a, psimd_u32 b) {
		return psimd_blend_u32(psimd_u32(x >> 31), a, b);
	}

	PSIMD_INTRINSIC psimd_f32 psimd_signblend_f32(psimd_f32 x, psimd_f32 a, psimd_f32 b) {
		const psimd_s32 mask = psimd_s32(truncate_to_int(x) >> 31);
		return psimd_blend_f32(mask, a, b);
	}
	

	/* Vector absolute value */
	PSIMD_INTRINSIC psimd_f32 psimd_abs_f32(psimd_f32 v) {
		return abs(v);
	}

	/* Vector negation */
	PSIMD_INTRINSIC psimd_f32 psimd_neg_f32(psimd_f32 v) {
		const psimd_s32 mask = truncate_to_int(psimd_splat_f32(-0.0f));
		return to_float(truncate_to_int(v) ^ mask);
	}

	/* Vector maximum */
	PSIMD_INTRINSIC psimd_s8 psimd_max_s8(psimd_s8 a, psimd_s8 b) {
		#if defined(__ARM_NEON__)
			return (psimd_s8) vmaxq_s8((int8x16_t) a, (int8x16_t) b);
		#else
			return max(a, b);
		#endif
	}

	PSIMD_INTRINSIC psimd_u8 psimd_max_u8(psimd_u8 a, psimd_u8 b) {
		#if defined(__ARM_NEON__)
			return (psimd_u8) vmaxq_u8((uint8x16_t) a, (uint8x16_t) b);
		#else
			return max(a, b);
		#endif
	}

	PSIMD_INTRINSIC psimd_s16 psimd_max_s16(psimd_s16 a, psimd_s16 b) {
		#if defined(__ARM_NEON__)
			return (psimd_s16) vmaxq_s16((int16x8_t) a, (int16x8_t) b);
		#else
			return max(a, b);
		#endif
	}

	PSIMD_INTRINSIC psimd_u16 psimd_max_u16(psimd_u16 a, psimd_u16 b) {
		#if defined(__ARM_NEON__)
			return (psimd_u16) vmaxq_u16((uint16x8_t) a, (uint16x8_t) b);
		#else
			return max(a, b);
		#endif
	}

	PSIMD_INTRINSIC psimd_s32 psimd_max_s32(psimd_s32 a, psimd_s32 b) {
		#if defined(__ARM_NEON__)
			return (psimd_s32) vmaxq_s32((int32x4_t) a, (int32x4_t) b);
		#else
			return max(a, b);
		#endif
	}

	PSIMD_INTRINSIC psimd_u32 psimd_max_u32(psimd_u32 a, psimd_u32 b) {
		#if defined(__ARM_NEON__)
			return (psimd_u32) vmaxq_u32((uint32x4_t) a, (uint32x4_t) b);
		#else
			return max(a, b);
		#endif
	}

	PSIMD_INTRINSIC psimd_f32 psimd_max_f32(psimd_f32 a, psimd_f32 b) {
		#if defined(__ARM_NEON__)
			return (psimd_f32) vmaxq_f32((float32x4_t) a, (float32x4_t) b);
		#else
			return max(a, b);
		#endif
	}

	/* Vector minimum */
	PSIMD_INTRINSIC psimd_s8 psimd_min_s8(psimd_s8 a, psimd_s8 b) {
		#if defined(__ARM_NEON__)
			return (psimd_s8) vminq_s8((int8x16_t) a, (int8x16_t) b);
		#else
			return min(a, b);
		#endif
	}

	PSIMD_INTRINSIC psimd_u8 psimd_min_u8(psimd_u8 a, psimd_u8 b) {
		#if defined(__ARM_NEON__)
			return (psimd_u8) vminq_u8((uint8x16_t) a, (uint8x16_t) b);
		#else
		return min(a, b);
		#endif
	}

	PSIMD_INTRINSIC psimd_s16 psimd_min_s16(psimd_s16 a, psimd_s16 b) {
		#if defined(__ARM_NEON__)
			return (psimd_s16) vminq_s16((int16x8_t) a, (int16x8_t) b);
		#else
		return min(a, b);
		#endif
	}

	PSIMD_INTRINSIC psimd_u16 psimd_min_u16(psimd_u16 a, psimd_u16 b) {
		#if defined(__ARM_NEON__)
			return (psimd_u16) vminq_u16((uint16x8_t) a, (uint16x8_t) b);
		#else
			return min(a, b);
		#endif
	}

	PSIMD_INTRINSIC psimd_s32 psimd_min_s32(psimd_s32 a, psimd_s32 b) {
		#if defined(__ARM_NEON__)
			return (psimd_s32) vminq_s32((int32x4_t) a, (int32x4_t) b);
		#else
			return min(a, b);
		#endif
	}

	PSIMD_INTRINSIC psimd_u32 psimd_min_u32(psimd_u32 a, psimd_u32 b) {
		#if defined(__ARM_NEON__)
			return (psimd_u32) vminq_u32((uint32x4_t) a, (uint32x4_t) b);
		#else
			return min(a, b);
		#endif
	}

	PSIMD_INTRINSIC psimd_f32 psimd_min_f32(psimd_f32 a, psimd_f32 b) {
		#if defined(__ARM_NEON__)
			return (psimd_f32) vminq_f32((float32x4_t) a, (float32x4_t) b);
		#else
			return min(a, b);
		#endif
	}

	/* Broadcast vector element */
	#if defined(__clang__)
		PSIMD_INTRINSIC psimd_f32 psimd_splat0_f32(psimd_f32 v) {
			return __builtin_shufflevector(v, v, 0, 0, 0, 0);
		}

		PSIMD_INTRINSIC psimd_f32 psimd_splat1_f32(psimd_f32 v) {
			return __builtin_shufflevector(v, v, 1, 1, 1, 1);
		}

		PSIMD_INTRINSIC psimd_f32 psimd_splat2_f32(psimd_f32 v) {
			return __builtin_shufflevector(v, v, 2, 2, 2, 2);
		}

		PSIMD_INTRINSIC psimd_f32 psimd_splat3_f32(psimd_f32 v) {
			return __builtin_shufflevector(v, v, 3, 3, 3, 3);
		}
	#else
		PSIMD_INTRINSIC psimd_f32 psimd_splat0_f32(psimd_f32 v) {
			return _mm_shuffle_ps(v, v, _MM_SHUFFLE(0, 0, 0, 0)); 
		}

		PSIMD_INTRINSIC psimd_f32 psimd_splat1_f32(psimd_f32 v) {
			return _mm_shuffle_ps(v, v, _MM_SHUFFLE(1, 1, 1, 1));
		}

		PSIMD_INTRINSIC psimd_f32 psimd_splat2_f32(psimd_f32 v) {
			return _mm_shuffle_ps(v, v, _MM_SHUFFLE(2, 2, 2, 2));
		}

		PSIMD_INTRINSIC psimd_f32 psimd_splat3_f32(psimd_f32 v) {
			return _mm_shuffle_ps(v, v, _MM_SHUFFLE(3, 3, 3, 3));
		}
	#endif

	/* Reversal of vector elements */
	#if defined(__clang__)
		PSIMD_INTRINSIC psimd_s8 psimd_reverse_s8(psimd_s8 v) {
			return __builtin_shufflevector(v, v, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);
		}

		PSIMD_INTRINSIC psimd_u8 psimd_reverse_u8(psimd_u8 v) {
			return __builtin_shufflevector(v, v, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);
		}

		PSIMD_INTRINSIC psimd_s16 psimd_reverse_s16(psimd_s16 v) {
			return __builtin_shufflevector(v, v, 7, 6, 5, 4, 3, 2, 1, 0);
		}

		PSIMD_INTRINSIC psimd_u16 psimd_reverse_u16(psimd_u16 v) {
			return __builtin_shufflevector(v, v, 7, 6, 5, 4, 3, 2, 1, 0);
		}

		PSIMD_INTRINSIC psimd_s32 psimd_reverse_s32(psimd_s32 v) {
			return __builtin_shufflevector(v, v, 3, 2, 1, 0);
		}

		PSIMD_INTRINSIC psimd_u32 psimd_reverse_u32(psimd_u32 v) {
			return __builtin_shufflevector(v, v, 3, 2, 1, 0);
		}

		PSIMD_INTRINSIC psimd_f32 psimd_reverse_f32(psimd_f32 v) {
			return __builtin_shufflevector(v, v, 3, 2, 1, 0);
		}
	#else
		static const __m128i vmA = _mm_setr_epi8(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);
		static const __m128i vmB = _mm_setr_epi8(14, 15, 12, 13, 10, 11, 8, 9, 6, 7, 4, 5, 2, 3, 0, 1);

		PSIMD_INTRINSIC psimd_s8 psimd_reverse_s8(psimd_s8 v) {
			return _mm_shuffle_epi8(v, vmA);
		}

		PSIMD_INTRINSIC psimd_u8 psimd_reverse_u8(psimd_u8 v) {
			return _mm_shuffle_epi8(v, vmA);
		}

		PSIMD_INTRINSIC psimd_s16 psimd_reverse_s16(psimd_s16 v) {
			return  _mm_shuffle_epi8(v, vmB);
		}

		PSIMD_INTRINSIC psimd_u16 psimd_reverse_u16(psimd_u16 v) {
			return  _mm_shuffle_epi8(v, vmB);
		}

		PSIMD_INTRINSIC psimd_s32 psimd_reverse_s32(psimd_s32 v) {
			return _mm_shuffle_epi32(v, _MM_SHUFFLE(3, 2, 1, 0));
		}

		PSIMD_INTRINSIC psimd_u32 psimd_reverse_u32(psimd_u32 v) {
			return _mm_shuffle_epi32(v, _MM_SHUFFLE(3, 2, 1, 0));
		}

		PSIMD_INTRINSIC psimd_f32 psimd_reverse_f32(psimd_f32 v) {
			return _mm_shuffle_ps(v, v, _MM_SHUFFLE(3, 2, 1, 0));
		}
	#endif

	/* Interleaving of vector elements */
	#if defined(__clang__)
		PSIMD_INTRINSIC psimd_s16 psimd_interleave_lo_s16(psimd_s16 a, psimd_s16 b) {
			return __builtin_shufflevector(a, b, 0, 8+0, 1, 8+1, 2, 8+2, 3, 8+3);
		}

		PSIMD_INTRINSIC psimd_s16 psimd_interleave_hi_s16(psimd_s16 a, psimd_s16 b) {
			return __builtin_shufflevector(a, b, 4, 8+4, 5, 8+5, 6, 8+6, 7, 8+7);
		}

		PSIMD_INTRINSIC psimd_u16 psimd_interleave_lo_u16(psimd_u16 a, psimd_u16 b) {
			return __builtin_shufflevector(a, b, 0, 8+0, 1, 8+1, 2, 8+2, 3, 8+3);
		}

		PSIMD_INTRINSIC psimd_u16 psimd_interleave_hi_u16(psimd_u16 a, psimd_u16 b) {
			return __builtin_shufflevector(a, b, 4, 8+4, 5, 8+5, 6, 8+6, 7, 8+7);
		}

		PSIMD_INTRINSIC psimd_s32 psimd_interleave_lo_s32(psimd_s32 a, psimd_s32 b) {
			return __builtin_shufflevector(a, b, 0, 4+0, 1, 4+1);
		}

		PSIMD_INTRINSIC psimd_s32 psimd_interleave_hi_s32(psimd_s32 a, psimd_s32 b) {
			return __builtin_shufflevector(a, b, 2, 4+2, 3, 4+3);
		}

		PSIMD_INTRINSIC psimd_u32 psimd_interleave_lo_u32(psimd_u32 a, psimd_u32 b) {
			return __builtin_shufflevector(a, b, 0, 4+0, 1, 4+1);
		}

		PSIMD_INTRINSIC psimd_u32 psimd_interleave_hi_u32(psimd_u32 a, psimd_u32 b) {
			return __builtin_shufflevector(a, b, 2, 4+2, 3, 4+3);
		}

		PSIMD_INTRINSIC psimd_f32 psimd_interleave_lo_f32(psimd_f32 a, psimd_f32 b) {
			return __builtin_shufflevector(a, b, 0, 4+0, 1, 4+1);
		}

		PSIMD_INTRINSIC psimd_f32 psimd_interleave_hi_f32(psimd_f32 a, psimd_f32 b) {
			return __builtin_shufflevector(a, b, 2, 4+2, 3, 4+3);
		}
	#else
		PSIMD_INTRINSIC psimd_s16 psimd_interleave_lo_s16(psimd_s16 a, psimd_s16 b) {
			return __builtin_shuffle(a, b, (psimd_s16) { 0, 8+0, 1, 8+1, 2, 8+2, 3, 8+3 });
		}

		PSIMD_INTRINSIC psimd_s16 psimd_interleave_hi_s16(psimd_s16 a, psimd_s16 b) {
			return __builtin_shuffle(a, b, (psimd_s16) { 4, 8+4, 5, 8+5, 6, 8+6, 7, 8+7 });
		}

		PSIMD_INTRINSIC psimd_u16 psimd_interleave_lo_u16(psimd_u16 a, psimd_u16 b) {
			return __builtin_shuffle(a, b, (psimd_s16) { 0, 8+0, 1, 8+1, 2, 8+2, 3, 8+3 });
		}

		PSIMD_INTRINSIC psimd_u16 psimd_interleave_hi_u16(psimd_u16 a, psimd_u16 b) {
			return __builtin_shuffle(a, b, (psimd_s16) { 4, 8+4, 5, 8+5, 6, 8+6, 7, 8+7 });
		}

		PSIMD_INTRINSIC psimd_s32 psimd_interleave_lo_s32(psimd_s32 a, psimd_s32 b) {
			return __builtin_shuffle(a, b, (psimd_s32) { 0, 4+0, 1, 4+1 });
		}

		PSIMD_INTRINSIC psimd_s32 psimd_interleave_hi_s32(psimd_s32 a, psimd_s32 b) {
			return __builtin_shuffle(a, b, (psimd_s32) { 2, 4+2, 3, 4+3 });
		}

		PSIMD_INTRINSIC psimd_u32 psimd_interleave_lo_u32(psimd_u32 a, psimd_u32 b) {
			return __builtin_shuffle(a, b, (psimd_s32) { 0, 4+0, 1, 4+1 });
		}

		PSIMD_INTRINSIC psimd_u32 psimd_interleave_hi_u32(psimd_u32 a, psimd_u32 b) {
			return __builtin_shuffle(a, b, (psimd_s32) { 2, 4+2, 3, 4+3 });
		}

		PSIMD_INTRINSIC psimd_f32 psimd_interleave_lo_f32(psimd_f32 a, psimd_f32 b) {
			return _mm_shuffle_ps(a, b, _MM_SHUFFLE(0, 4+0, 1, 4+1));
		}

		PSIMD_INTRINSIC psimd_f32 psimd_interleave_hi_f32(psimd_f32 a, psimd_f32 b) {
			return _mm_shuffle_ps(a, b, _MM_SHUFFLE(2, 4+2, 3, 4+3));
		}
	#endif

	/* Concatenation of low/high vector elements */
	#if defined(__clang__)
		PSIMD_INTRINSIC psimd_s16 psimd_concat_lo_s16(psimd_s16 a, psimd_s16 b) {
			return __builtin_shufflevector(a, b, 0, 1, 2, 3, 8+0, 8+1, 8+2, 8+3);
		}

		PSIMD_INTRINSIC psimd_s16 psimd_concat_hi_s16(psimd_s16 a, psimd_s16 b) {
			return __builtin_shufflevector(a, b, 4, 5, 6, 7, 8+4, 8+5, 8+6, 8+7);
		}

		PSIMD_INTRINSIC psimd_u16 psimd_concat_lo_u16(psimd_u16 a, psimd_u16 b) {
			return __builtin_shufflevector(a, b, 0, 1, 2, 3, 8+0, 8+1, 8+2, 8+3);
		}

		PSIMD_INTRINSIC psimd_u16 psimd_concat_hi_u16(psimd_u16 a, psimd_u16 b) {
			return __builtin_shufflevector(a, b, 4, 5, 6, 7, 8+4, 8+5, 8+6, 8+7);
		}

		PSIMD_INTRINSIC psimd_s32 psimd_concat_lo_s32(psimd_s32 a, psimd_s32 b) {
			return __builtin_shufflevector(a, b, 0, 1, 4+0, 4+1);
		}

		PSIMD_INTRINSIC psimd_s32 psimd_concat_hi_s32(psimd_s32 a, psimd_s32 b) {
			return __builtin_shufflevector(a, b, 2, 3, 4+2, 4+3);
		}

		PSIMD_INTRINSIC psimd_u32 psimd_concat_lo_u32(psimd_u32 a, psimd_u32 b) {
			return __builtin_shufflevector(a, b, 0, 1, 4+0, 4+1);
		}

		PSIMD_INTRINSIC psimd_u32 psimd_concat_hi_u32(psimd_u32 a, psimd_u32 b) {
			return __builtin_shufflevector(a, b, 2, 3, 4+2, 4+3);
		}

		PSIMD_INTRINSIC psimd_f32 psimd_concat_lo_f32(psimd_f32 a, psimd_f32 b) {
			return __builtin_shufflevector(a, b, 0, 1, 4+0, 4+1);
		}

		PSIMD_INTRINSIC psimd_f32 psimd_concat_hi_f32(psimd_f32 a, psimd_f32 b) {
			return __builtin_shufflevector(a, b, 2, 3, 4+2, 4+3);
		}
	#else
		PSIMD_INTRINSIC psimd_s16 psimd_concat_lo_s16(psimd_s16 a, psimd_s16 b) {
			return __builtin_shuffle(a, b, (psimd_s16) { 0, 1, 2, 3, 8+0, 8+1, 8+2, 8+3 });
		}

		PSIMD_INTRINSIC psimd_s16 psimd_concat_hi_s16(psimd_s16 a, psimd_s16 b) {
			return __builtin_shuffle(a, b, (psimd_s16) { 4, 5, 6, 7, 8+4, 8+5, 8+6, 8+7 });
		}

		PSIMD_INTRINSIC psimd_u16 psimd_concat_lo_u16(psimd_u16 a, psimd_u16 b) {
			return __builtin_shuffle(a, b, (psimd_s16) { 0, 1, 2, 3, 8+0, 8+1, 8+2, 8+3 });
		}

		PSIMD_INTRINSIC psimd_u16 psimd_concat_hi_u16(psimd_u16 a, psimd_u16 b) {
			return __builtin_shuffle(a, b, (psimd_s16) { 4, 5, 6, 7, 8+4, 8+5, 8+6, 8+7 });
		}

		PSIMD_INTRINSIC psimd_s32 psimd_concat_lo_s32(psimd_s32 a, psimd_s32 b) {
			return __builtin_shuffle(a, b, (psimd_s32) { 0, 1, 4+0, 4+1 });
		}

		PSIMD_INTRINSIC psimd_s32 psimd_concat_hi_s32(psimd_s32 a, psimd_s32 b) {
			return __builtin_shuffle(a, b, (psimd_s32) { 2, 3, 4+2, 4+3 });
		}

		PSIMD_INTRINSIC psimd_u32 psimd_concat_lo_u32(psimd_u32 a, psimd_u32 b) {
			return __builtin_shuffle(a, b, (psimd_s32) { 0, 1, 4+0, 4+1 });
		}

		PSIMD_INTRINSIC psimd_u32 psimd_concat_hi_u32(psimd_u32 a, psimd_u32 b) {
			return __builtin_shuffle(a, b, (psimd_s32) { 2, 3, 4+2, 4+3 });
		}

		PSIMD_INTRINSIC psimd_f32 psimd_concat_lo_f32(psimd_f32 a, psimd_f32 b) {
			return _mm_shuffle_ps(a, b, _MM_SHUFFLE(0, 1, 4+0, 4+1));
		}

		PSIMD_INTRINSIC psimd_f32 psimd_concat_hi_f32(psimd_f32 a, psimd_f32 b) {
			return _mm_shuffle_ps(a, b, _MM_SHUFFLE(2, 3, 4+2, 4+3));
		}
	#endif

	/* Concatenation of even/odd vector elements */
	#if defined(__clang__)
		PSIMD_INTRINSIC psimd_s16 psimd_concat_even_s16(psimd_s16 a, psimd_s16 b) {
			return __builtin_shufflevector(a, b, 0, 2, 4, 6, 8+0, 8+2, 8+4, 8+6);
		}

		PSIMD_INTRINSIC psimd_s16 psimd_concat_odd_s16(psimd_s16 a, psimd_s16 b) {
			return __builtin_shufflevector(a, b, 1, 3, 5, 7, 8+1, 8+3, 8+5, 8+7);
		}

		PSIMD_INTRINSIC psimd_u16 psimd_concat_even_u16(psimd_u16 a, psimd_u16 b) {
			return __builtin_shufflevector(a, b, 0, 2, 4, 6, 8+0, 8+2, 8+4, 8+6);
		}

		PSIMD_INTRINSIC psimd_u16 psimd_concat_odd_u16(psimd_u16 a, psimd_u16 b) {
			return __builtin_shufflevector(a, b, 1, 3, 5, 7, 8+1, 8+3, 8+5, 8+7);
		}

		PSIMD_INTRINSIC psimd_s32 psimd_concat_even_s32(psimd_s32 a, psimd_s32 b) {
			return __builtin_shufflevector(a, b, 0, 2, 4+0, 4+2);
		}

		PSIMD_INTRINSIC psimd_s32 psimd_concat_odd_s32(psimd_s32 a, psimd_s32 b) {
			return __builtin_shufflevector(a, b, 1, 3, 4+1, 4+3);
		}

		PSIMD_INTRINSIC psimd_u32 psimd_concat_even_u32(psimd_u32 a, psimd_u32 b) {
			return __builtin_shufflevector(a, b, 0, 2, 4+0, 4+2);
		}

		PSIMD_INTRINSIC psimd_u32 psimd_concat_odd_u32(psimd_u32 a, psimd_u32 b) {
			return __builtin_shufflevector(a, b, 1, 3, 4+1, 4+3);
		}

		PSIMD_INTRINSIC psimd_f32 psimd_concat_even_f32(psimd_f32 a, psimd_f32 b) {
			return __builtin_shufflevector(a, b, 0, 2, 4+0, 4+2);
		}

		PSIMD_INTRINSIC psimd_f32 psimd_concat_odd_f32(psimd_f32 a, psimd_f32 b) {
			return __builtin_shufflevector(a, b, 1, 3, 4+1, 4+3);
		}
	#else
		PSIMD_INTRINSIC psimd_s16 psimd_concat_even_s16(psimd_s16 a, psimd_s16 b) {
			return __builtin_shuffle(a, b, (psimd_s16) { 0, 2, 4, 6, 8+0, 8+2, 8+4, 8+6 });
		}

		PSIMD_INTRINSIC psimd_s16 psimd_concat_odd_s16(psimd_s16 a, psimd_s16 b) {
			return __builtin_shuffle(a, b, (psimd_s16) { 1, 3, 5, 7, 8+1, 8+3, 8+5, 8+7 });
		}

		PSIMD_INTRINSIC psimd_u16 psimd_concat_even_u16(psimd_u16 a, psimd_u16 b) {
			return __builtin_shuffle(a, b, (psimd_s16) { 0, 2, 4, 6, 8+0, 8+2, 8+4, 8+6 });
		}

		PSIMD_INTRINSIC psimd_u16 psimd_concat_odd_u16(psimd_u16 a, psimd_u16 b) {
			return __builtin_shuffle(a, b, (psimd_s16) { 1, 3, 5, 7, 8+1, 8+3, 8+5, 8+7 });
		}

		PSIMD_INTRINSIC psimd_s32 psimd_concat_even_s32(psimd_s32 a, psimd_s32 b) {
			return __builtin_shuffle(a, b, _MM_SHUFFLE(0, 2, 4 + 0, 4 + 2));
		}

		PSIMD_INTRINSIC psimd_s32 psimd_concat_odd_s32(psimd_s32 a, psimd_s32 b) {
			return __builtin_shuffle(a, b, _MM_SHUFFLE(1, 3, 4 + 1, 4 + 3));
		}

		PSIMD_INTRINSIC psimd_u32 psimd_concat_even_u32(psimd_u32 a, psimd_u32 b) {
			return __builtin_shuffle(a, b, _MM_SHUFFLE(0, 2, 4 + 0, 4 + 2));
		}

		PSIMD_INTRINSIC psimd_u32 psimd_concat_odd_u32(psimd_u32 a, psimd_u32 b) {
			
			return __builtin_shuffle(a, b, _MM_SHUFFLE(1, 3, 4 + 1, 4 + 3));
		}

		PSIMD_INTRINSIC psimd_f32 psimd_concat_even_f32(psimd_f32 a, psimd_f32 b) {
			return _mm_shuffle_ps(a, b, _MM_SHUFFLE(0, 2, 4+0, 4+2));
		}

		PSIMD_INTRINSIC psimd_f32 psimd_concat_odd_f32(psimd_f32 a, psimd_f32 b) {
			return _mm_shuffle_ps(a, b, _MM_SHUFFLE(1, 3, 4+1, 4+3));
		}
	#endif

	/* Vector reduce */
	#if defined(__clang__)
		PSIMD_INTRINSIC psimd_f32 psimd_allreduce_sum_f32(psimd_f32 v) {
			const psimd_f32 temp = v + __builtin_shufflevector(v, v, 2, 3, 0, 1);
			return temp + __builtin_shufflevector(temp, temp, 1, 0, 3, 2);
		}

		PSIMD_INTRINSIC psimd_f32 psimd_allreduce_max_f32(psimd_f32 v) {
			const psimd_f32 temp = psimd_max_f32(v, __builtin_shufflevector(v, v, 2, 3, 0, 1));
			return psimd_max_f32(temp, __builtin_shufflevector(temp, temp, 1, 0, 3, 2));
		}

		PSIMD_INTRINSIC psimd_f32 psimd_allreduce_min_f32(psimd_f32 v) {
			const psimd_f32 temp = psimd_min_f32(v, __builtin_shufflevector(v, v, 2, 3, 0, 1));
			return psimd_min_f32(temp, __builtin_shufflevector(temp, temp, 1, 0, 3, 2));
		}

		PSIMD_INTRINSIC float psimd_reduce_sum_f32(psimd_f32 v) {
			const psimd_f32 temp = v + __builtin_shufflevector(v, v, 2, 3, -1, -1);
			const psimd_f32 result = temp + __builtin_shufflevector(temp, temp, 1, -1, -1, -1);
			return result[0];
		}

		PSIMD_INTRINSIC float psimd_reduce_max_f32(psimd_f32 v) {
			const psimd_f32 temp = psimd_max_f32(v, __builtin_shufflevector(v, v, 2, 3, -1, -1));
			const psimd_f32 result = psimd_max_f32(temp, __builtin_shufflevector(temp, temp, 1, -1, -1, -1));
			return result[0];
		}

		PSIMD_INTRINSIC float psimd_reduce_min_f32(psimd_f32 v) {
			const psimd_f32 temp = psimd_min_f32(v, __builtin_shufflevector(v, v, 2, 3, -1, -1));
			const psimd_f32 result = psimd_min_f32(temp, __builtin_shufflevector(temp, temp, 1, -1, -1, -1));
			return result[0];
		}
	#else
		PSIMD_INTRINSIC psimd_f32 psimd_allreduce_sum_f32(psimd_f32 v) {
			const psimd_f32 temp = v + _mm_shuffle_ps(v, v, _MM_SHUFFLE(2, 3, 0, 1));
			return temp + _mm_shuffle_ps(temp, temp, _MM_SHUFFLE(1, 0, 3, 2 ));
		}

		PSIMD_INTRINSIC psimd_f32 psimd_allreduce_max_f32(psimd_f32 v) {
			const psimd_f32 temp = psimd_max_f32(v, _mm_shuffle_ps(v, v, _MM_SHUFFLE(2, 3, 0, 1)));
			return psimd_max_f32(temp, _mm_shuffle_ps(temp, temp, _MM_SHUFFLE(1, 0, 3, 2)));
		}

		PSIMD_INTRINSIC psimd_f32 psimd_allreduce_min_f32(psimd_f32 v) {
			const psimd_f32 temp = psimd_min_f32(v, _mm_shuffle_ps(v, v, _MM_SHUFFLE(2, 3, 0, 1)));
			return psimd_min_f32(temp, _mm_shuffle_ps(temp, temp, _MM_SHUFFLE(1, 0, 3, 2)));
		}

		PSIMD_INTRINSIC float psimd_reduce_sum_f32(psimd_f32 v) {
			const psimd_f32 result = psimd_allreduce_sum_f32(v);
			return result[0];
		}

		PSIMD_INTRINSIC float psimd_reduce_max_f32(psimd_f32 v) {
			const psimd_f32 result = psimd_allreduce_max_f32(v);
			return result[0];
		}

		PSIMD_INTRINSIC float psimd_reduce_min_f32(psimd_f32 v) {
			const psimd_f32 result = psimd_allreduce_min_f32(v);
			return result[0];
		}
	#endif
#endif

#endif /* PSIMD_H */
