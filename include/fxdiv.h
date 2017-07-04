#pragma once
#ifndef FXDIV_H
#define FXDIV_H

#ifdef __cplusplus
#include <cstddef>
#include <cstdint>
#include <climits>
#else
#include <stddef.h>
#include <stdint.h>
#include <limits.h>
#endif

#include <intrin.h>

static inline uint64_t fxdiv_mulext_uint32_t(uint32_t a, uint32_t b) 
{
	return (uint64_t) a * (uint64_t) b;
}

static inline uint32_t fxdiv_mulhi_uint32_t(uint32_t a, uint32_t b) 
{
	return (uint32_t) (((uint64_t) a * (uint64_t) b) >> 32);
}

static inline uint64_t fxdiv_mulhi_uint64_t(uint64_t a, uint64_t b) 
{
	return (uint64_t) __umulh((unsigned __int64) a, (unsigned __int64) b);
}

static inline size_t fxdiv_mulhi_size_t(size_t a, size_t b) 
{
	return (size_t) fxdiv_mulhi_uint64_t((uint64_t) a, (uint64_t) b);
}

struct fxdiv_divisor_uint32_t 
{
	uint32_t value;
	uint32_t m;
	uint8_t s1;
	uint8_t s2;
};

struct fxdiv_result_uint32_t 
{
	uint32_t quotient;
	uint32_t remainder;
};

struct fxdiv_divisor_uint64_t 
{
	uint64_t value;
	uint64_t m;
	uint8_t s1;
	uint8_t s2;
};

struct fxdiv_result_uint64_t 
{
	uint64_t quotient;
	uint64_t remainder;
};

struct fxdiv_divisor_size_t 
{
	size_t value;
	size_t m;
	uint8_t s1;
	uint8_t s2;
};

struct fxdiv_result_size_t 
{
	size_t quotient;
	size_t remainder;
};

static inline struct fxdiv_divisor_uint32_t fxdiv_init_uint32_t(uint32_t d) 
{
	struct fxdiv_divisor_uint32_t result = { d };

	if (d == 1) 
	{
		result.m = UINT32_C(1);
		result.s1 = 0;
		result.s2 = 0;
	} 
	else 
	{
		unsigned long l_minus_1;
		_BitScanReverse(&l_minus_1, (unsigned long) (d - 1));
	
		const uint32_t u_hi = (UINT32_C(2) << (uint32_t) l_minus_1) - d;

		/* Division of 64-bit number u_hi:UINT32_C(0) by 32-bit number d, 32-bit quotient output q */
		const uint32_t q = ((uint64_t) u_hi << 32) / d;
	
		result.m = q + UINT32_C(1);
		result.s1 = 1;
		result.s2 = (uint8_t) l_minus_1;
	}

	return result;
}

static inline struct fxdiv_divisor_uint64_t fxdiv_init_uint64_t(uint64_t d) 
{
	struct fxdiv_divisor_uint64_t result = { d };
	
	if (d == 1) 
	{
		result.m = UINT64_C(1);
		result.s1 = 0;
		result.s2 = 0;
	} 
	else 
	{
		unsigned long l_minus_1;
		_BitScanReverse64(&l_minus_1, (unsigned __int64) (d - 1));
		unsigned long bsr_d;
		_BitScanReverse64(&bsr_d, (unsigned __int64) d);
		const uint32_t nlz_d = bsr_d ^ 0x3F;
	
		uint64_t u_hi = (UINT64_C(2) << (uint32_t) l_minus_1) - d;

		/* Division of 128-bit number u_hi:UINT64_C(0) by 64-bit number d, 64-bit quotient output q */
		/* Implementation based on code from Hacker's delight */

		/* Normalize divisor and shift divident left */
		d <<= nlz_d;
		u_hi <<= nlz_d;
		/* Break divisor up into two 32-bit digits */
		const uint64_t d_hi = (uint32_t) (d >> 32);
		const uint32_t d_lo = (uint32_t) d;

		/* Compute the first quotient digit, q1 */
		uint64_t q1 = u_hi / d_hi;
		uint64_t r1 = u_hi - q1 * d_hi;

		while ((q1 >> 32) != 0 || fxdiv_mulext_uint32_t((uint32_t) q1, d_lo) > (r1 << 32)) 
		{
			q1 -= 1;
			r1 += d_hi;
			if ((r1 >> 32) != 0)
				break;
		}

		/* Multiply and subtract. */
		u_hi = (u_hi << 32) - q1 * d;

		/* Compute the second quotient digit, q0 */
		uint64_t q0 = u_hi / d_hi;
		uint64_t r0 = u_hi - q0 * d_hi;

		while ((q0 >> 32) != 0 || fxdiv_mulext_uint32_t((uint32_t) q0, d_lo) > (r0 << 32)) 
		{
			q0 -= 1;
			r0 += d_hi;
			if ((r0 >> 32) != 0)
				break;
		}
		
		const uint64_t q = (q1 << 32) | (uint32_t) q0;
	
		result.m = q + UINT64_C(1);
		result.s1 = 1;
		result.s2 = (uint8_t) l_minus_1;
	}

	return result;
}

static inline struct fxdiv_divisor_size_t fxdiv_init_size_t(size_t d) 
{
#if SIZE_MAX == UINT32_MAX
	const fxdiv_divisor_uint32_t uint_result = fxdiv_init_uint32_t((uint32_t) d);
#elif SIZE_MAX == UINT64_MAX
	const struct fxdiv_divisor_uint64_t uint_result = fxdiv_init_uint64_t((uint64_t) d);
#else
	#error Unsupported platform
#endif
	struct fxdiv_divisor_size_t size_result = 
	{
		(size_t) uint_result.value,
		(size_t) uint_result.m,
		uint_result.s1,
		uint_result.s2
	};
	return size_result;
}

static inline uint32_t fxdiv_quotient_uint32_t(uint32_t n, const struct fxdiv_divisor_uint32_t divisor) 
{
	const uint32_t t = fxdiv_mulhi_uint32_t(n, divisor.m);
	return (t + ((n - t) >> divisor.s1)) >> divisor.s2;
}

static inline uint64_t fxdiv_quotient_uint64_t(uint64_t n, const struct fxdiv_divisor_uint64_t divisor) 
{
	const uint64_t t = fxdiv_mulhi_uint64_t(n, divisor.m);
	return (t + ((n - t) >> divisor.s1)) >> divisor.s2;
}

static inline size_t fxdiv_quotient_size_t(size_t n, const struct fxdiv_divisor_size_t divisor) 
{
#if SIZE_MAX == UINT32_MAX
	const fxdiv_divisor_uint32_t uint32_divisor = 
	{
		(uint32_t) divisor.value,
		(uint32_t) divisor.m,
		divisor.s1,
		divisor.s2
	};
	return fxdiv_quotient_uint32_t((uint32_t) n, uint32_divisor);
#elif SIZE_MAX == UINT64_MAX
	const struct fxdiv_divisor_uint64_t uint64_divisor = 
	{
		(uint64_t) divisor.value,
		(uint64_t) divisor.m,
		divisor.s1,
		divisor.s2
	};
	return fxdiv_quotient_uint64_t((uint64_t) n, uint64_divisor);
#else
	#error Unsupported platform
#endif
}

static inline uint32_t fxdiv_remainder_uint32_t(uint32_t n, const struct fxdiv_divisor_uint32_t divisor) 
{
	const uint32_t quotient = fxdiv_quotient_uint32_t(n, divisor);
	return n - quotient * divisor.value;
}

static inline uint64_t fxdiv_remainder_uint64_t(uint64_t n, const struct fxdiv_divisor_uint64_t divisor) 
{
	const uint64_t quotient = fxdiv_quotient_uint64_t(n, divisor);
	return n - quotient * divisor.value;
}

static inline size_t fxdiv_remainder_size_t(size_t n, const struct fxdiv_divisor_size_t divisor) 
{
	const size_t quotient = fxdiv_quotient_size_t(n, divisor);
	return n - quotient * divisor.value;
}

static inline uint32_t fxdiv_round_down_uint32_t(uint32_t n, const struct fxdiv_divisor_uint32_t granularity) 
{
	const uint32_t quotient = fxdiv_quotient_uint32_t(n, granularity);
	return quotient * granularity.value;
}

static inline uint64_t fxdiv_round_down_uint64_t(uint64_t n, const struct fxdiv_divisor_uint64_t granularity) 
{
	const uint64_t quotient = fxdiv_quotient_uint64_t(n, granularity);
	return quotient * granularity.value;
}

static inline size_t fxdiv_round_down_size_t(size_t n, const struct fxdiv_divisor_size_t granularity) 
{
	const size_t quotient = fxdiv_quotient_size_t(n, granularity);
	return quotient * granularity.value;
}

static inline struct fxdiv_result_uint32_t fxdiv_divide_uint32_t(uint32_t n, const struct fxdiv_divisor_uint32_t divisor) 
{
	const uint32_t quotient = fxdiv_quotient_uint32_t(n, divisor);
	const uint32_t remainder = n - quotient * divisor.value;
	struct fxdiv_result_uint32_t result = { quotient, remainder };
	return result;
}

static inline struct fxdiv_result_uint64_t fxdiv_divide_uint64_t(uint64_t n, const struct fxdiv_divisor_uint64_t divisor) 
{
	const uint64_t quotient = fxdiv_quotient_uint64_t(n, divisor);
	const uint64_t remainder = n - quotient * divisor.value;
	struct fxdiv_result_uint64_t result = { quotient, remainder };
	return result;
}

static inline struct fxdiv_result_size_t fxdiv_divide_size_t(size_t n, const struct fxdiv_divisor_size_t divisor) 
{
	const size_t quotient = fxdiv_quotient_size_t(n, divisor);
	const size_t remainder = n - quotient * divisor.value;
	struct fxdiv_result_size_t result = { quotient, remainder };
	return result;
}

#endif /* FXDIV_H */
