#ifdef _MSC_VER
#include <intrin.h>
#else
#include <pthread.h>

static pthread_once_t hwinfo_init_control = PTHREAD_ONCE_INIT;
#endif

#if defined(__i386__) || defined(__x86_64__)

#include <cpuid.h>
#ifndef bit_AVX2
#define bit_AVX2 0x00000020
#endif

#if __native_client__
#define NNP_NACL_CODE_BUNDLE_SIZE 32
#include <irt.h>
#endif
#endif

#if defined(__ANDROID__) && defined(__arm__)
#include <cpu-features.h>
#endif

#include <nnpack.h>
#include <hwinfo.h>
#include <blas.h>
#include <transform.h>
#include <relu.h>
#include <softmax.h>

#if defined(NNP_BACKEND_SCALAR)
#include <../src/scalar/fft/aos.h>
#include <../src/scalar/fft/soa.h>
#include <../src/scalar/fft/real.h>
#include <../src/scalar/fft/dualreal.h>
#include <../src/scalar/winograd/f6x6k3x3.h>


#endif

hardware_info nnp_hwinfo = {  };

struct cpu_info
{
	int eax;
	int ebx;
	int ecx;
	int edx;
};

#ifndef _MSC_VER
static pthread_once_t hwinfo_init_control = PTHREAD_ONCE_INIT;
#ifndef __native_client__
/*
* This instruction may be not supported by Native Client validator, make sure it doesn't appear in the binary
*/
static inline uint64_t xgetbv(uint32_t ext_ctrl_reg) {
	uint32_t lo, hi;
	asm(".byte 0x0F, 0x01, 0xD0" : "=a" (lo), "=d" (hi) : "c" (ext_ctrl_reg));
	return (((uint64_t)hi) << 32) | (uint64_t)lo;
}
#endif
#else
static inline uint64_t xgetbv(uint32_t ext_ctrl_reg)
{
	return _xgetbv(ext_ctrl_reg);
}

static inline uint32_t __get_cpuid_max(unsigned int __level, unsigned int *__sig)
{
	cpu_info basic_info;
	__cpuid(&basic_info.eax, (int)__level);

	if (__sig)
		*__sig = (unsigned int)basic_info.ebx;

	return (uint32_t)basic_info.eax;
}
#endif


static void init_x86_hwinfo() 
{
	const uint32_t max_base_info = __get_cpuid_max(0, NULL);
	const uint32_t max_extended_info = __get_cpuid_max(0x80000000, NULL);
#ifdef __native_client__
	/*
	* Under Native Client sandbox we can't just ask the CPU:
	* - First, some instructions (XGETBV) necessary to query AVX support are not white-listed in the validator.
	* - Secondly, even if CPU supports some instruction, but validator doesn't know about it (e.g. due a bug in the
	*   ISA detection in the validator), all instructions from the "unsupported" ISA extensions will be replaced by
	*   HLTs when the module is loaded.
	* Thus, instead of quering the CPU about supported ISA extensions, we query the validator: we pass bundles with
	* instructions from ISA extensions to dynamic code generation APIs, and test if they are accepted.
	*/

	static const uint8_t avx_bundle[NNP_NACL_CODE_BUNDLE_SIZE] = {
		/* VPERMILPS ymm0, ymm1, 0xAA */
		0xC4, 0xE3, 0x7D, 0x04, 0xC1, 0xAA,
		/* Fill remainder with HLTs */
		0xF4, 0xF4, 0xF4, 0xF4, 0xF4, 0xF4, 0xF4, 0xF4, 0xF4, 0xF4,
		0xF4, 0xF4, 0xF4, 0xF4, 0xF4, 0xF4, 0xF4, 0xF4, 0xF4, 0xF4, 0xF4, 0xF4, 0xF4, 0xF4, 0xF4, 0xF4,
	};
	static const uint8_t fma3_bundle[NNP_NACL_CODE_BUNDLE_SIZE] = {
		/* VFMADDSUB213PS ymm0, ymm1, ymm2 */
		0xC4, 0xE2, 0x75, 0xA6, 0xC2,
		/* Fill remainder with HLTs */
		0xF4, 0xF4, 0xF4, 0xF4, 0xF4, 0xF4, 0xF4, 0xF4, 0xF4, 0xF4, 0xF4,
		0xF4, 0xF4, 0xF4, 0xF4, 0xF4, 0xF4, 0xF4, 0xF4, 0xF4, 0xF4, 0xF4, 0xF4, 0xF4, 0xF4, 0xF4, 0xF4,
	};
	static const uint8_t avx2_bundle[NNP_NACL_CODE_BUNDLE_SIZE] = {
		/* VPERMPS ymm0, ymm1, ymm2 */
		0xC4, 0xE2, 0x75, 0x16, 0xC2,
		/* Fill remainder with HLTs */
		0xF4, 0xF4, 0xF4, 0xF4, 0xF4, 0xF4, 0xF4, 0xF4, 0xF4, 0xF4, 0xF4,
		0xF4, 0xF4, 0xF4, 0xF4, 0xF4, 0xF4, 0xF4, 0xF4, 0xF4, 0xF4, 0xF4, 0xF4, 0xF4, 0xF4, 0xF4, 0xF4,
	};

	struct nacl_irt_code_data_alloc nacl_irt_code_data_alloc = { 0 };
	if (nacl_interface_query(NACL_IRT_CODE_DATA_ALLOC_v0_1, &nacl_irt_code_data_alloc,
		sizeof(nacl_irt_code_data_alloc)) == sizeof(nacl_irt_code_data_alloc))
	{
		struct nacl_irt_dyncode nacl_irt_dyncode = { 0 };
		if (nacl_interface_query(NACL_IRT_DYNCODE_v0_1, &nacl_irt_dyncode,
			sizeof(nacl_irt_dyncode)) == sizeof(nacl_irt_dyncode))
		{
			const size_t allocation_size = 65536;
			uintptr_t code_segment = 0;
			if (nacl_irt_code_data_alloc.allocate_code_data(0, allocation_size, 0, 0, &code_segment) == 0) {
				nnp_hwinfo.isa.has_avx =
					!nacl_irt_dyncode.dyncode_create((void*)code_segment, avx_bundle, NNP_NACL_CODE_BUNDLE_SIZE);
				code_segment += NNP_NACL_CODE_BUNDLE_SIZE;

				nnp_hwinfo.isa.has_fma3 =
					!nacl_irt_dyncode.dyncode_create((void*)code_segment, fma3_bundle, NNP_NACL_CODE_BUNDLE_SIZE);
				code_segment += NNP_NACL_CODE_BUNDLE_SIZE;

				nnp_hwinfo.isa.has_avx2 =
					!nacl_irt_dyncode.dyncode_create((void*)code_segment, avx2_bundle, NNP_NACL_CODE_BUNDLE_SIZE);
			}
		}
	}
#else
	// Under normal environments, just ask the CPU about supported ISA extensions.
	if (max_base_info >= 1)
	{
		cpu_info basic_info;

#ifdef _MSC_VER
		__cpuid(&basic_info.eax, 1);
#else
		__cpuid(1, basic_info.eax, basic_info.ebx, basic_info.ecx, basic_info.edx);
#endif
		
		// OSXSAVE: ecx[bit 27] in basic info
		const bool osxsave = !!(basic_info.ecx & bit_OSXSAVE);
		// Check that AVX[bit 2] and SSE[bit 1] registers are preserved by OS
		const bool ymm_regs = (osxsave ? ((xgetbv(0) & 0b110ul) == 0b110ul) : false);

		cpu_info structured_info = { 0 };
		if (max_base_info >= 7)
#ifdef _MSC_VER
			__cpuidex(&structured_info.eax, 7, 0);
#else
			__cpuid_count(7, 0, structured_info.eax, structured_info.ebx, structured_info.ecx, structured_info.edx);
#endif
		
		if (ymm_regs) 
		{
			// AVX: ecx[bit 28] in basic info
			nnp_hwinfo.isa.has_avx = !!(basic_info.ecx & bit_AVX);
			// FMA3: ecx[bit 12] in basic info
			nnp_hwinfo.isa.has_fma3 = !!(basic_info.ecx & bit_FMA);
			// AVX2: ebx[bit 5] in structured feature info
			nnp_hwinfo.isa.has_avx2 = !!(structured_info.ebx & bit_AVX2);
		}
	}
	
	// Detect CPU vendor
	cpu_info vendor_info;
#ifdef _MSC_VER
	__cpuid(&vendor_info.eax, 0);
#else
	__cpuid(0, vendor_info.eax, vendor_info.ebx, vendor_info.ecx, vendor_info.edx);
#endif
	const uint32_t Auth = UINT32_C(0x68747541), enti = UINT32_C(0x69746E65), cAMD = UINT32_C(0x444D4163);
	const uint32_t Genu = UINT32_C(0x756E6547), ineI = UINT32_C(0x49656E69), ntel = UINT32_C(0x6C65746E);
	const uint32_t Cent = UINT32_C(0x746E6543), aurH = UINT32_C(0x48727561), auls = UINT32_C(0x736C7561);
	const bool is_intel = !((vendor_info.ebx ^ Genu) | (vendor_info.edx ^ ineI) | (vendor_info.ecx ^ ntel));
	const bool is_amd = !((vendor_info.ebx ^ Auth) | (vendor_info.edx ^ enti) | (vendor_info.ecx ^ cAMD));
	const bool is_via = !((vendor_info.ebx ^ Cent) | (vendor_info.edx ^ aurH) | (vendor_info.ecx ^ auls));

	// Detect cache
	if (max_base_info >= 4)
	{
		for (uint32_t cache_id = 0; ; cache_id++) 
		{
			cpu_info cpuInfo;
#ifdef _MSC_VER
			__cpuidex(&cpuInfo.eax, 4, cache_id);
#else
			__cpuid_count(4, cache_id, cache_info.eax, cache_info.ebx, cache_info.ecx, cache_info.edx);
#endif
			// eax[bits 0-4]: cache type (0 - no more caches, 1 - data, 2 - instruction, 3 - unified)
			const uint32_t type = cpuInfo.eax & 0x1F;
			if (type == 0) 
				break;
			else 
			{
				if ((type == 1) || (type == 3))
				{
					// eax[bits 5-7]: cache level (starts at 1)
					const uint32_t level = (cpuInfo.eax >> 5) & 0x7;
					// eax[bits 14-25]: number of IDs for logical processors sharing the cache - 1
					const uint32_t threads = ((cpuInfo.eax >> 14) & 0xFFF) + 1;
					// eax[bits 26-31]: number of IDs for processor cores in the physical package - 1
					const uint32_t cores = (cpuInfo.eax >> 26) + 1;

					// ebx[bits 0-11]: line size - 1
					const uint32_t line_size = (cpuInfo.ebx & 0xFFF) + 1;
					// ebx[bits 12-21]: line_partitions - 1
					const uint32_t line_partitions = ((cpuInfo.ebx >> 12) & 0x3FF) + 1;
					// ebx[bits 22-31]: associativity - 1
					const uint32_t associativity = (cpuInfo.ebx >> 22) + 1;
					// ecx: number of sets - 1
					const uint32_t sets = cpuInfo.ecx + 1;
					// edx[bit 1]: cache inclusiveness
					const bool inclusive = !!(cpuInfo.edx & 0x2);
					
					const cache_info cacheInfo =
					{
						sets * associativity * line_partitions * line_size,
						associativity,
						threads,
						inclusive
					};

					switch (level)
					{
					case 1:
						nnp_hwinfo.cache.l1 = cacheInfo;
						break;
					case 2:
						nnp_hwinfo.cache.l2 = cacheInfo;
						break;
					case 3:
						nnp_hwinfo.cache.l3 = cacheInfo;
						break;
					case 4:
						nnp_hwinfo.cache.l4 = cacheInfo;
						break;
					}
				}
			}
		}
	}
#endif
}

#if !(defined(__x86_64__) || defined(__i386__) || defined(_MSC_VER)) || defined(__ANDROID__)
static void init_static_hwinfo(void) 
{
	nnp_hwinfo.cache.l1 = cache_info
	{
		16 * 1024,
		4,
		1,
		true
	};
	nnp_hwinfo.cache.l2 = cache_info
	{
		128 * 1024,
		4,
		1,
		true
	};
	nnp_hwinfo.cache.l3 = cache_info
	{
		2 * 1024 * 1024,
		8,
		1,
		true
	};
}
#endif

#if !defined(__i386__) && !defined(__x86_64__) && !defined(_MSC_VER) && defined(__APPLE__) 
static void init_static_ios_hwinfo(void) 
{
	nnp_hwinfo.cache.l1 = cache_info 
	{
		1 * 32 * 1024,
		1,
		1,
		false
	};
	nnp_hwinfo.cache.l2 = cache_info
	{
		1024 * 1024,
		1,
		1,
		false
	};
	nnp_hwinfo.cache.l3 = cache_info
	{
		2 * 1024 * 1024,
		8,
		1,
		false
	};
}
#endif


#if NNP_BACKEND_X86_64 || NNP_BACKEND_WIN64
static const nnp_shdotxf_function shdotxf_function[8] = {
	nnp_shdotxf1__avx2,
	nnp_shdotxf2__avx2,
	nnp_shdotxf3__avx2,
	nnp_shdotxf4__avx2,
	nnp_shdotxf5__avx2,
	nnp_shdotxf6__avx2,
	nnp_shdotxf7__avx2,
	nnp_shdotxf8__avx2
};
static const nnp_sdotxf_function sdotxf_function[8] = {
	nnp_sdotxf1__avx2,
	nnp_sdotxf2__avx2,
	nnp_sdotxf3__avx2,
	nnp_sdotxf4__avx2,
	nnp_sdotxf5__avx2,
	nnp_sdotxf6__avx2,
	nnp_sdotxf7__avx2,
	nnp_sdotxf8__avx2
};
#elif NNP_BACKEND_ARM
static const nnp_sdotxf_function sdotxf_function[8] = {
	nnp_sdotxf1__neon,
	nnp_sdotxf2__neon,
	nnp_sdotxf3__neon,
	nnp_sdotxf4__neon,
	nnp_sdotxf5__neon,
	nnp_sdotxf6__neon,
	nnp_sdotxf7__neon,
	nnp_sdotxf8__neon,
};

static const nnp_shdotxf_function shdotxf_function[8] = {
	nnp_shdotxf1__psimd,
	nnp_shdotxf2__psimd,
	nnp_shdotxf3__psimd,
	nnp_shdotxf4__psimd,
	nnp_shdotxf5__psimd,
	nnp_shdotxf6__psimd,
	nnp_shdotxf7__psimd,
	nnp_shdotxf8__psimd,
};
#elif NNP_BACKEND_PSIMD
static const nnp_shdotxf_function shdotxf_function[8] = {
	nnp_shdotxf1__psimd,
	nnp_shdotxf2__psimd,
	nnp_shdotxf3__psimd,
	nnp_shdotxf4__psimd,
	nnp_shdotxf5__psimd,
	nnp_shdotxf6__psimd,
	nnp_shdotxf7__psimd,
	nnp_shdotxf8__psimd,
};
static const nnp_sdotxf_function sdotxf_function[8] = {
	nnp_sdotxf1__psimd,
	nnp_sdotxf2__psimd,
	nnp_sdotxf3__psimd,
	nnp_sdotxf4__psimd,
	nnp_sdotxf5__psimd,
	nnp_sdotxf6__psimd,
	nnp_sdotxf7__psimd,
	nnp_sdotxf8__psimd,
};
#elif NNP_BACKEND_SCALAR
static const nnp_shdotxf_function shdotxf_function[8] = {
	nnp_shdotxf1__scalar,
	nnp_shdotxf2__scalar,
	nnp_shdotxf3__scalar,
	nnp_shdotxf4__scalar,
	nnp_shdotxf5__scalar,
	nnp_shdotxf6__scalar,
	nnp_shdotxf7__scalar,
	nnp_shdotxf8__scalar,
};
static const nnp_sdotxf_function sdotxf_function[8] =  {
	nnp_sdotxf1__scalar,
	nnp_sdotxf2__scalar,
	nnp_sdotxf3__scalar,
	nnp_sdotxf4__scalar,
	nnp_sdotxf5__scalar,
	nnp_sdotxf6__scalar,
	nnp_sdotxf7__scalar,
	nnp_sdotxf8__scalar,
};
#endif


static void init_hwinfo() 
{
	init_x86_hwinfo();
	
	// Compute high-level cache blocking parameters
	nnp_hwinfo.blocking.l1 = nnp_hwinfo.cache.l1.size;

	if (nnp_hwinfo.cache.l1.threads > 1u) 
		nnp_hwinfo.blocking.l1 /= nnp_hwinfo.cache.l1.threads;
	
	if (nnp_hwinfo.cache.l2.size != 0u) 
	{
		nnp_hwinfo.blocking.l2 = nnp_hwinfo.cache.l2.size;
		if (nnp_hwinfo.cache.l2.inclusive)
			nnp_hwinfo.blocking.l2 -= nnp_hwinfo.cache.l1.size;
		
		if (nnp_hwinfo.cache.l2.threads > 1u)
			nnp_hwinfo.blocking.l2 /= nnp_hwinfo.cache.l2.threads;
	}

	if (nnp_hwinfo.cache.l3.size != 0u) 
	{
		nnp_hwinfo.blocking.l3 = nnp_hwinfo.cache.l3.size;
		if (nnp_hwinfo.cache.l3.inclusive)
			nnp_hwinfo.blocking.l3 -= nnp_hwinfo.cache.l2.size;
	}

	nnp_hwinfo.blocking.l4 = nnp_hwinfo.cache.l4.size;

	if (nnp_hwinfo.cache.l1.size && nnp_hwinfo.cache.l2.size && nnp_hwinfo.cache.l3.size)
	{
#if NNP_BACKEND_X86_64 || NNP_BACKEND_WIN64
		if (nnp_hwinfo.isa.has_avx2 && nnp_hwinfo.isa.has_fma3)
		{
			nnp_hwinfo.simd_width = 8u;
			nnp_hwinfo.transforms.fft8x8_with_offset_and_store = (nnp_transform_2d_with_offset)nnp_fft8x8_with_offset_and_store__avx2;
			nnp_hwinfo.transforms.fft8x8_with_offset_and_stream = (nnp_transform_2d_with_offset)nnp_fft8x8_with_offset_and_stream__avx2;
			nnp_hwinfo.transforms.ifft8x8_with_offset = (nnp_transform_2d_with_offset)nnp_ifft8x8_with_offset__avx2;
			nnp_hwinfo.transforms.ifft8x8_with_bias = (nnp_transform_2d_with_bias)nnp_ifft8x8_with_bias__avx2;
			nnp_hwinfo.transforms.ifft8x8_with_bias_with_relu = (nnp_transform_2d_with_bias)nnp_ifft8x8_with_bias_with_relu__avx2;
			nnp_hwinfo.transforms.fft16x16_with_offset_and_store = (nnp_transform_2d_with_offset)nnp_fft16x16_with_offset_and_store__avx2;
			nnp_hwinfo.transforms.fft16x16_with_offset_and_stream = (nnp_transform_2d_with_offset)nnp_fft16x16_with_offset_and_stream__avx2;
			nnp_hwinfo.transforms.ifft16x16_with_offset = (nnp_transform_2d_with_offset)nnp_ifft16x16_with_offset__avx2;
			nnp_hwinfo.transforms.ifft16x16_with_bias = (nnp_transform_2d_with_bias)nnp_ifft16x16_with_bias__avx2;
			nnp_hwinfo.transforms.ifft16x16_with_bias_with_relu = (nnp_transform_2d_with_bias)nnp_ifft16x16_with_bias_with_relu__avx2;
			nnp_hwinfo.transforms.iwt_f6x6_3x3_with_offset_and_store = (nnp_transform_2d_with_offset)nnp_iwt8x8_3x3_with_offset_and_store__avx2;
			nnp_hwinfo.transforms.iwt_f6x6_3x3_with_offset_and_stream = (nnp_transform_2d_with_offset)nnp_iwt8x8_3x3_with_offset_and_stream__avx2;
			nnp_hwinfo.transforms.kwt_f6x6_3x3 = (nnp_transform_2d_with_offset)nnp_kwt8x8_3x3_and_stream__avx2;
			nnp_hwinfo.transforms.kwt_f6x6_3Rx3R = (nnp_transform_2d_with_offset)nnp_kwt8x8_3Rx3R_and_stream__avx2;
			nnp_hwinfo.transforms.owt_f6x6_3x3 = (nnp_transform_2d_with_offset)nnp_owt8x8_3x3__avx2;
			nnp_hwinfo.transforms.owt_f6x6_3x3_with_bias = (nnp_transform_2d_with_bias)nnp_owt8x8_3x3_with_bias__avx2;
			nnp_hwinfo.transforms.owt_f6x6_3x3_with_bias_with_relu = (nnp_transform_2d_with_bias)nnp_owt8x8_3x3_with_bias_with_relu__avx2;

			nnp_hwinfo.activations.relu = nnp_relu__avx2;
			nnp_hwinfo.activations.inplace_relu = nnp_inplace_relu__avx2;
			nnp_hwinfo.activations.grad_relu = nnp_grad_relu__avx2;
			nnp_hwinfo.activations.softmax = nnp_softmax__avx2;
			nnp_hwinfo.activations.inplace_softmax = nnp_inplace_softmax__avx2;

			nnp_hwinfo.sdotxf = sdotxf
			{
				sdotxf_function,
				NNP_COUNT_OF(sdotxf_function)
			};;

			nnp_hwinfo.shdotxf = shdotxf
			{
				shdotxf_function,
				NNP_COUNT_OF(shdotxf_function)
			};

			nnp_hwinfo.conv1x1 = convolution
			{
				nnp_conv1x1_only_2x4__fma3,
				nnp_conv1x1_upto_2x4__fma3,
				2u,
				4u
			};

			nnp_hwinfo.sgemm = sgemm
			{
				nnp_sgemm_only_4x24__fma3,
				nnp_sgemm_upto_4x24__fma3,
				4u,
				24u
			};

			nnp_hwinfo.sxgemm = sxgemm
			{
				(nnp_fast_tuple_gemm_function)nnp_s8gemm_only_3x4__fma3,
				(nnp_full_tuple_gemm_function)nnp_s8gemm_upto_3x4__fma3,
				3u,
				4u
			};

			nnp_hwinfo.cxgemm = cxgemm
			{
				(nnp_fast_tuple_gemm_function)nnp_s4c6gemm_only_2x2__fma3,
				(nnp_full_tuple_gemm_function)nnp_s4c6gemm_upto_2x2__fma3,
				(nnp_fast_tuple_gemm_function)nnp_c8gemm_only_2x2__fma3,
				(nnp_full_tuple_gemm_function)nnp_c8gemm_upto_2x2__fma3,
				(nnp_fast_tuple_gemm_function)nnp_s4c6gemm_conjb_only_2x2__fma3,
				(nnp_full_tuple_gemm_function)nnp_s4c6gemm_conjb_upto_2x2__fma3,
				(nnp_fast_tuple_gemm_function)nnp_c8gemm_conjb_only_2x2__fma3,
				(nnp_full_tuple_gemm_function)nnp_c8gemm_conjb_upto_2x2__fma3,
				(nnp_fast_tuple_gemm_function)nnp_s4c6gemm_conjb_transc_only_2x2__fma3,
				(nnp_full_tuple_gemm_function)nnp_s4c6gemm_conjb_transc_upto_2x2__fma3,
				(nnp_fast_tuple_gemm_function)nnp_c8gemm_conjb_transc_only_2x2__fma3,
				(nnp_full_tuple_gemm_function)nnp_c8gemm_conjb_transc_upto_2x2__fma3,
				2u,
				2u
			};

			nnp_hwinfo.supported = true;
		}
#elif NNP_BACKEND_PSIMD
		nnp_hwinfo.simd_width = 4;
		nnp_hwinfo.transforms.fft8x8_with_offset_and_store = (nnp_transform_2d_with_offset)nnp_fft8x8_with_offset__psimd;
		nnp_hwinfo.transforms.fft8x8_with_offset_and_stream = (nnp_transform_2d_with_offset)nnp_fft8x8_with_offset__psimd;
		nnp_hwinfo.transforms.ifft8x8_with_offset = (nnp_transform_2d_with_offset)nnp_ifft8x8_with_offset__psimd;
		nnp_hwinfo.transforms.ifft8x8_with_bias = (nnp_transform_2d_with_bias)nnp_ifft8x8_with_bias__psimd;
		nnp_hwinfo.transforms.ifft8x8_with_bias_with_relu = (nnp_transform_2d_with_bias)nnp_ifft8x8_with_bias_with_relu__psimd;
		nnp_hwinfo.transforms.fft16x16_with_offset_and_store = (nnp_transform_2d_with_offset)nnp_fft16x16_with_offset__psimd;
		nnp_hwinfo.transforms.fft16x16_with_offset_and_stream = (nnp_transform_2d_with_offset)nnp_fft16x16_with_offset__psimd;
		nnp_hwinfo.transforms.ifft16x16_with_offset = (nnp_transform_2d_with_offset)nnp_ifft16x16_with_offset__psimd;
		nnp_hwinfo.transforms.ifft16x16_with_bias = (nnp_transform_2d_with_bias)nnp_ifft16x16_with_bias__psimd;
		nnp_hwinfo.transforms.ifft16x16_with_bias_with_relu = (nnp_transform_2d_with_bias)nnp_ifft16x16_with_bias_with_relu__psimd;
		nnp_hwinfo.transforms.iwt_f6x6_3x3_with_offset_and_store = (nnp_transform_2d_with_offset)nnp_iwt8x8_3x3_with_offset__psimd;
		nnp_hwinfo.transforms.iwt_f6x6_3x3_with_offset_and_stream = (nnp_transform_2d_with_offset)nnp_iwt8x8_3x3_with_offset__psimd;
		nnp_hwinfo.transforms.kwt_f6x6_3x3 = (nnp_transform_2d_with_offset)nnp_kwt8x8_3x3__psimd;
		nnp_hwinfo.transforms.kwt_f6x6_3Rx3R = (nnp_transform_2d_with_offset)nnp_kwt8x8_3Rx3R__psimd;
		nnp_hwinfo.transforms.owt_f6x6_3x3 = (nnp_transform_2d_with_offset)nnp_owt8x8_3x3__psimd;
		nnp_hwinfo.transforms.owt_f6x6_3x3_with_bias = (nnp_transform_2d_with_bias)nnp_owt8x8_3x3_with_bias__psimd;
		nnp_hwinfo.transforms.owt_f6x6_3x3_with_bias_with_relu = (nnp_transform_2d_with_bias)nnp_owt8x8_3x3_with_bias_with_relu__psimd;
		nnp_hwinfo.activations.relu = nnp_relu__psimd;
		nnp_hwinfo.activations.inplace_relu = nnp_inplace_relu__psimd;
		nnp_hwinfo.activations.grad_relu = nnp_grad_relu__psimd;
		nnp_hwinfo.activations.softmax = nnp_softmax__psimd;
		nnp_hwinfo.activations.inplace_softmax = nnp_inplace_softmax__psimd;
		nnp_hwinfo.sdotxf = sdotxf
		{
			sdotxf,
			NNP_COUNT_OF(sdotxf)
		};
		nnp_hwinfo.shdotxf = shdotxf
		{
			shdotxf,
			NNP_COUNT_OF(shdotxf)
		};
		nnp_hwinfo.conv1x1 = convolution
		{
			nnp_conv1x1_only_2x4__psimd,
			nnp_conv1x1_upto_2x4__psimd,
			2,
			4
		};
		nnp_hwinfo.sgemm = sgemm
		{
			nnp_sgemm_only_4x8__psimd,
			nnp_sgemm_upto_4x8__psimd,
			4,
			8
		};
		nnp_hwinfo.sxgemm = sxgemm
		{
			(nnp_fast_tuple_gemm_function)nnp_s4gemm_only_3x4__psimd,
			(nnp_full_tuple_gemm_function)nnp_s4gemm_upto_3x4__psimd,
			3,
			4
		};
		nnp_hwinfo.cxgemm = cxgemm
		{
			(nnp_fast_tuple_gemm_function)nnp_s4c2gemm_only_2x2__psimd,
			(nnp_full_tuple_gemm_function)nnp_s4c2gemm_upto_2x2__psimd,
			(nnp_fast_tuple_gemm_function)nnp_c4gemm_only_2x2__psimd,
			(nnp_full_tuple_gemm_function)nnp_c4gemm_upto_2x2__psimd,
			(nnp_fast_tuple_gemm_function)nnp_s4c2gemm_conjb_only_2x2__psimd,
			(nnp_full_tuple_gemm_function)nnp_s4c2gemm_conjb_upto_2x2__psimd,
			(nnp_fast_tuple_gemm_function)nnp_c4gemm_conjb_only_2x2__psimd,
			(nnp_full_tuple_gemm_function)nnp_c4gemm_conjb_upto_2x2__psimd,
			(nnp_fast_tuple_gemm_function)nnp_s4c2gemm_conjb_transc_only_2x2__psimd,
			(nnp_full_tuple_gemm_function)nnp_s4c2gemm_conjb_transc_upto_2x2__psimd,
			(nnp_fast_tuple_gemm_function)nnp_c4gemm_conjb_transc_only_2x2__psimd,
			(nnp_full_tuple_gemm_function)nnp_c4gemm_conjb_transc_upto_2x2__psimd,
			2,
			2
		};
		nnp_hwinfo.supported = true;
#elif NNP_BACKEND_ARM
#if defined(__ANDROID__) && defined(__arm__) && !defined(__aarch64__)
		const bool has_fp16 = (android_getCpuFeatures() & ANDROID_CPU_ARM_FEATURE_VFP_FP16) != 0;
#else
		const bool has_fp16 = true;
#endif

		nnp_hwinfo.simd_width = 4;
		nnp_hwinfo.transforms.fft8x8_with_offset_and_store = (nnp_transform_2d_with_offset)nnp_fft8x8_with_offset__psimd;
		nnp_hwinfo.transforms.fft8x8_with_offset_and_stream = (nnp_transform_2d_with_offset)nnp_fft8x8_with_offset__psimd;
		nnp_hwinfo.transforms.ifft8x8_with_offset = (nnp_transform_2d_with_offset)nnp_ifft8x8_with_offset__psimd;
		nnp_hwinfo.transforms.ifft8x8_with_bias = (nnp_transform_2d_with_bias)nnp_ifft8x8_with_bias__psimd;
		nnp_hwinfo.transforms.ifft8x8_with_bias_with_relu = (nnp_transform_2d_with_bias)nnp_ifft8x8_with_bias_with_relu__psimd;
		nnp_hwinfo.transforms.fft16x16_with_offset_and_store = (nnp_transform_2d_with_offset)nnp_fft16x16_with_offset__psimd;
		nnp_hwinfo.transforms.fft16x16_with_offset_and_stream = (nnp_transform_2d_with_offset)nnp_fft16x16_with_offset__psimd;
		nnp_hwinfo.transforms.ifft16x16_with_offset = (nnp_transform_2d_with_offset)nnp_ifft16x16_with_offset__psimd;
		nnp_hwinfo.transforms.ifft16x16_with_bias = (nnp_transform_2d_with_bias)nnp_ifft16x16_with_bias__psimd;
		nnp_hwinfo.transforms.ifft16x16_with_bias_with_relu = (nnp_transform_2d_with_bias)nnp_ifft16x16_with_bias_with_relu__psimd;
		nnp_hwinfo.transforms.iwt_f6x6_3x3_with_offset_and_store = (nnp_transform_2d_with_offset)nnp_iwt8x8_3x3_with_offset__neon;
		nnp_hwinfo.transforms.iwt_f6x6_3x3_with_offset_and_stream = (nnp_transform_2d_with_offset)nnp_iwt8x8_3x3_with_offset__neon;
		nnp_hwinfo.transforms.kwt_f6x6_3x3 = (nnp_transform_2d_with_offset)nnp_kwt8x8_3x3__neon;
		nnp_hwinfo.transforms.kwt_f6x6_3Rx3R = (nnp_transform_2d_with_offset)nnp_kwt8x8_3Rx3R__neon;
		nnp_hwinfo.transforms.owt_f6x6_3x3 = (nnp_transform_2d_with_offset)nnp_owt8x8_3x3__neon;
		nnp_hwinfo.transforms.owt_f6x6_3x3_with_bias = (nnp_transform_2d_with_bias)nnp_owt8x8_3x3_with_bias__neon;
		nnp_hwinfo.transforms.owt_f6x6_3x3_with_bias_with_relu = (nnp_transform_2d_with_bias)nnp_owt8x8_3x3_with_bias_with_relu__neon;
		if (has_fp16)
		{
			nnp_hwinfo.transforms.iwt_f6x6_3x3_fp16_with_offset = (nnp_transform_2d_with_offset)nnp_iwt8x8_3x3_fp16_with_offset__neonhp;
			nnp_hwinfo.transforms.kwt_f6x6_3x3_fp16 = (nnp_transform_2d_with_offset)nnp_kwt8x8_3x3_fp16__neonhp;
			nnp_hwinfo.transforms.owt_f6x6_3x3_fp16_with_bias = (nnp_transform_2d_with_bias)nnp_owt8x8_3x3_fp16_with_bias__neonhp;
			nnp_hwinfo.transforms.owt_f6x6_3x3_fp16_with_bias_with_relu = (nnp_transform_2d_with_bias)nnp_owt8x8_3x3_fp16_with_bias_with_relu__neonhp;
		}
		nnp_hwinfo.activations.relu = nnp_relu__neon;
		nnp_hwinfo.activations.inplace_relu = nnp_inplace_relu__neon;
		nnp_hwinfo.activations.grad_relu = nnp_grad_relu__neon;
		nnp_hwinfo.activations.softmax = nnp_softmax__psimd;
		nnp_hwinfo.activations.inplace_softmax = nnp_inplace_softmax__psimd;
		nnp_hwinfo.sdotxf = sdotxf
		{
			sdotxf,
			NNP_COUNT_OF(sdotxf)
		};
		nnp_hwinfo.shdotxf = shdotxf
		{
			shdotxf,
			NNP_COUNT_OF(shdotxf)
		};
		nnp_hwinfo.conv1x1 = convolution
		{
			nnp_conv1x1_only_2x4__neon,
			nnp_conv1x1_upto_2x4__neon,
			2,
			4
		};
		nnp_hwinfo.sgemm = sgemm)
		{
		nnp_sgemm_only_4x12__neon,
			nnp_sgemm_upto_4x12__neon,
			4,
			12
		};
		nnp_hwinfo.sxgemm = sxgemm
		{
			(nnp_fast_tuple_gemm_function)nnp_s4gemm_only_3x4__neon,
			(nnp_full_tuple_gemm_function)nnp_s4gemm_upto_3x4__neon,
			3,
			4
		};
		if (has_fp16)
		{
			nnp_hwinfo.hxgemm = hxgemm
			{
				(nnp_fast_tuple_gemm_function)nnp_h4gemm_only_3x4__neonhp,
				(nnp_full_tuple_gemm_function)nnp_h4gemm_upto_3x4__neonhp,
				3,
				4
			};
		}
		nnp_hwinfo.cxgemm = cxgemm
		{
			(nnp_fast_tuple_gemm_function)nnp_s4c2gemm_only_2x2__neon,
			(nnp_full_tuple_gemm_function)nnp_s4c2gemm_upto_2x2__neon,
			(nnp_fast_tuple_gemm_function)nnp_c4gemm_only_2x2__neon,
			(nnp_full_tuple_gemm_function)nnp_c4gemm_upto_2x2__neon,
			(nnp_fast_tuple_gemm_function)nnp_s4c2gemm_conjb_only_2x2__neon,
			(nnp_full_tuple_gemm_function)nnp_s4c2gemm_conjb_upto_2x2__neon,
			(nnp_fast_tuple_gemm_function)nnp_c4gemm_conjb_only_2x2__neon,
			(nnp_full_tuple_gemm_function)nnp_c4gemm_conjb_upto_2x2__neon,
			(nnp_fast_tuple_gemm_function)nnp_s4c2gemm_conjb_transc_only_2x2__neon,
			(nnp_full_tuple_gemm_function)nnp_s4c2gemm_conjb_transc_upto_2x2__neon,
			(nnp_fast_tuple_gemm_function)nnp_c4gemm_conjb_transc_only_2x2__neon,
			(nnp_full_tuple_gemm_function)nnp_c4gemm_conjb_transc_upto_2x2__neon,
			2,
			2
		};
#if defined(__ANDROID__) && defined(__arm__) && !defined(__aarch64__)
		nnp_hwinfo.supported = (android_getCpuFeatures() & ANDROID_CPU_ARM_FEATURE_NEON) != 0;
#else
		nnp_hwinfo.supported = true;
#endif
#elif NNP_BACKEND_SCALAR
		nnp_hwinfo.simd_width = 1;
		nnp_hwinfo.transforms.fft8x8_with_offset_and_store = (nnp_transform_2d_with_offset)nnp_fft8x8_with_offset__scalar;
		nnp_hwinfo.transforms.fft8x8_with_offset_and_stream = (nnp_transform_2d_with_offset)nnp_fft8x8_with_offset__scalar;
		nnp_hwinfo.transforms.ifft8x8_with_offset = (nnp_transform_2d_with_offset)nnp_ifft8x8_with_offset__scalar;
		nnp_hwinfo.transforms.ifft8x8_with_bias = (nnp_transform_2d_with_bias)nnp_ifft8x8_with_bias__scalar;
		nnp_hwinfo.transforms.ifft8x8_with_bias_with_relu = (nnp_transform_2d_with_bias)nnp_ifft8x8_with_bias_with_relu__scalar;
		nnp_hwinfo.transforms.fft16x16_with_offset_and_store = (nnp_transform_2d_with_offset)nnp_fft16x16_with_offset__scalar;
		nnp_hwinfo.transforms.fft16x16_with_offset_and_stream = (nnp_transform_2d_with_offset)nnp_fft16x16_with_offset__scalar;
		nnp_hwinfo.transforms.ifft16x16_with_offset = (nnp_transform_2d_with_offset)nnp_ifft16x16_with_offset__scalar;
		nnp_hwinfo.transforms.ifft16x16_with_bias = (nnp_transform_2d_with_bias)nnp_ifft16x16_with_bias__scalar;
		nnp_hwinfo.transforms.ifft16x16_with_bias_with_relu = (nnp_transform_2d_with_bias)nnp_ifft16x16_with_bias_with_relu__scalar;
		nnp_hwinfo.transforms.iwt_f6x6_3x3_with_offset_and_store = (nnp_transform_2d_with_offset)nnp_iwt8x8_3x3_with_offset__scalar;
		nnp_hwinfo.transforms.iwt_f6x6_3x3_with_offset_and_stream = (nnp_transform_2d_with_offset)nnp_iwt8x8_3x3_with_offset__scalar;
		nnp_hwinfo.transforms.kwt_f6x6_3x3 = (nnp_transform_2d_with_offset)nnp_kwt8x8_3x3__scalar;
		nnp_hwinfo.transforms.kwt_f6x6_3Rx3R = (nnp_transform_2d_with_offset)nnp_kwt8x8_3Rx3R__scalar;
		nnp_hwinfo.transforms.owt_f6x6_3x3 = (nnp_transform_2d_with_offset)nnp_owt8x8_3x3__scalar;
		nnp_hwinfo.transforms.owt_f6x6_3x3_with_bias = (nnp_transform_2d_with_bias)nnp_owt8x8_3x3_with_bias__scalar;
		nnp_hwinfo.transforms.owt_f6x6_3x3_with_bias_with_relu = (nnp_transform_2d_with_bias)nnp_owt8x8_3x3_with_bias_with_relu__scalar;
		nnp_hwinfo.activations.relu = nnp_relu__scalar;
		nnp_hwinfo.activations.inplace_relu = nnp_inplace_relu__scalar;
		nnp_hwinfo.activations.grad_relu = nnp_grad_relu__scalar;
		nnp_hwinfo.activations.softmax = nnp_softmax__scalar;
		nnp_hwinfo.activations.inplace_softmax = nnp_inplace_softmax__scalar;

		nnp_hwinfo.sdotxf = sdotxf
		{
			sdotxf_function,
			NNP_COUNT_OF(sdotxf_function),
		};
		nnp_hwinfo.shdotxf = shdotxf
		{
			shdotxf_function,
			NNP_COUNT_OF(shdotxf_function),
		};
		nnp_hwinfo.conv1x1 = convolution
		{
			nnp_conv1x1_only_2x4__scalar,
			nnp_conv1x1_upto_2x4__scalar,
			2,
			4
		};
		nnp_hwinfo.sgemm = sgemm
		{
			nnp_sgemm_only_4x3__scalar,
			nnp_sgemm_upto_4x3__scalar,
			4,
			3
		};
		nnp_hwinfo.sxgemm = sxgemm
		{
			(nnp_fast_tuple_gemm_function)nnp_sgemm_only_4x3__scalar,
			(nnp_full_tuple_gemm_function)nnp_sgemm_upto_4x3__scalar,
			4,
			3
		};
		nnp_hwinfo.cxgemm = cxgemm
		{
			(nnp_fast_tuple_gemm_function)nnp_s2gemm_only_2x2__scalar,
			(nnp_full_tuple_gemm_function)nnp_s2gemm_upto_2x2__scalar,
			(nnp_fast_tuple_gemm_function)nnp_cgemm_only_2x2__scalar,
			(nnp_full_tuple_gemm_function)nnp_cgemm_upto_2x2__scalar,
			(nnp_fast_tuple_gemm_function)nnp_s2gemm_only_2x2__scalar,
			(nnp_full_tuple_gemm_function)nnp_s2gemm_upto_2x2__scalar,
			(nnp_fast_tuple_gemm_function)nnp_cgemm_conjb_only_2x2__scalar,
			(nnp_full_tuple_gemm_function)nnp_cgemm_conjb_upto_2x2__scalar,
			(nnp_fast_tuple_gemm_function)nnp_s2gemm_transc_only_2x2__scalar,
			(nnp_full_tuple_gemm_function)nnp_s2gemm_transc_upto_2x2__scalar,
			(nnp_fast_tuple_gemm_function)nnp_cgemm_conjb_transc_only_2x2__scalar,
			(nnp_full_tuple_gemm_function)nnp_cgemm_conjb_transc_upto_2x2__scalar,
			2,
			2
		};
		nnp_hwinfo.supported = true;
#else
#error Unsupported backend
#endif
	}
}

nnp_status nnp_initialize()
{
#ifdef _MSC_VER
	init_hwinfo();
#else
	pthread_once(&hwinfo_init_control, &init_hwinfo);
#endif

	if (nnp_hwinfo.supported)
		return nnp_status_success;
	else
		return nnp_status_unsupported_hardware;
}

nnp_status nnp_deinitialize() 
{
	return nnp_status_success;
}
