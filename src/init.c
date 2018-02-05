#if defined(_MSC_VER)
	#include <intrin.h>

	#define bit_OSXSAVE     (1 << 27)
	#define bit_AVX         (1 << 28)
	#define bit_FMA         (1 << 12)
	#ifndef bit_AVX2
		#define bit_AVX2    0x00000020
	#endif
#else
	#include <pthread.h>
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
#include <nnpack/hwinfo.h>
#include <nnpack/blas.h>
#include <nnpack/transform.h>
#include <nnpack/relu.h>
#include <nnpack/softmax.h>

struct hardware_info nnp_hwinfo = { false };

struct cpu_info {
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
    static inline uint64_t xgetbv(uint32_t ext_ctrl_reg) {
		return _xgetbv(ext_ctrl_reg);
	}

	static inline uint32_t __get_cpuid_max(unsigned int __level, unsigned int *__sig) {
		struct cpu_info basic_info;
		__cpuid(&basic_info.eax, (int)__level);

		if (__sig)
			*__sig = (unsigned int)basic_info.ebx;

		return (uint32_t)basic_info.eax;
	}
#endif


static void init_x86_hwinfo() {
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
	if (max_base_info >= 1)	{
		struct cpu_info basic_info;

#if defined(_MSC_VER)
		__cpuid(&basic_info.eax, 1);
#else
		__cpuid(1, basic_info.eax, basic_info.ebx, basic_info.ecx, basic_info.edx);
#endif
		
		// OSXSAVE: ecx[bit 27] in basic info
		const bool osxsave = !!(basic_info.ecx & bit_OSXSAVE);
		// Check that AVX[bit 2] and SSE[bit 1] registers are preserved by OS
		const bool ymm_regs = (osxsave ? ((xgetbv(0) & 0b110ul) == 0b110ul) : false);

		struct cpu_info structured_info = { 0 };
		if (max_base_info >= 7)
#if defined(_MSC_VER)
			__cpuidex(&structured_info.eax, 7, 0);
#else
			__cpuid_count(7, 0, structured_info.eax, structured_info.ebx, structured_info.ecx, structured_info.edx);
#endif
		
		if (ymm_regs) {
			// AVX: ecx[bit 28] in basic info
			nnp_hwinfo.isa.has_avx = !!(basic_info.ecx & bit_AVX);
			// FMA3: ecx[bit 12] in basic info
			nnp_hwinfo.isa.has_fma3 = !!(basic_info.ecx & bit_FMA);
			// AVX2: ebx[bit 5] in structured feature info
			nnp_hwinfo.isa.has_avx2 = !!(structured_info.ebx & bit_AVX2);
		}
	}
	
	// Detect CPU vendor
	struct cpu_info vendor_info;
#if defined(_MSC_VER)
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
	if (max_base_info >= 4)	{
		for (uint32_t cache_id = 0; ; cache_id++) {
			struct cpu_info cache_info;
#if defined(_MSC_VER)
			__cpuidex(&cache_info.eax, 4, cache_id);
#else
			__cpuid_count(4, cache_id, cache_info.eax, cache_info.ebx, cache_info.ecx, cache_info.edx);
#endif
			/* eax[bits 0-4]: cache type (0 - no more caches, 1 - data, 2 - instruction, 3 - unified) */
			const uint32_t type = cache_info.eax & 0x1F;
			if (type == 0)
			{
				break;
			}
			else if ((type == 1) || (type == 3))
			{
				/* eax[bits 5-7]: cache level (starts at 1) */
				const uint32_t level = (cache_info.eax >> 5) & 0x7;
				/* eax[bits 14-25]: number of IDs for logical processors sharing the cache - 1 */
				const uint32_t threads = ((cache_info.eax >> 14) & 0xFFF) + 1;
				/* eax[bits 26-31]: number of IDs for processor cores in the physical package - 1 */
				const uint32_t cores = (cache_info.eax >> 26) + 1;

				/* ebx[bits 0-11]: line size - 1 */
				const uint32_t line_size = (cache_info.ebx & 0xFFF) + 1;
				/* ebx[bits 12-21]: line_partitions - 1 */
				const uint32_t line_partitions = ((cache_info.ebx >> 12) & 0x3FF) + 1;
				/* ebx[bits 22-31]: associativity - 1 */
				const uint32_t associativity = (cache_info.ebx >> 22) + 1;
				/* ecx: number of sets - 1 */
				const uint32_t sets = cache_info.ecx + 1;
				/* edx[bit 1]: cache inclusiveness */
				const bool inclusive = !!(cache_info.edx & 0x2);

				const struct cache_info cache_info = {
					.size = sets * associativity * line_partitions * line_size,
					.associativity = associativity,
					.threads = threads,
					.inclusive = inclusive
				};
				
				//struct cpu_info cpuInfo;

				switch (level) {
				case 1:
					nnp_hwinfo.cache.l1 = cache_info;
					break;
				case 2:
					nnp_hwinfo.cache.l2 = cache_info;
					break;
				case 3:
					nnp_hwinfo.cache.l3 = cache_info;
					break;
				case 4:
					nnp_hwinfo.cache.l4 = cache_info;
					break;
				}
			}
		}
	}
#endif
}

#if !(defined(__x86_64__) || defined(__i386__) || defined(_MSC_VER)) || defined(__ANDROID__)
static void init_static_hwinfo(void) {
	nnp_hwinfo.cache.l1 = cache_info {
		.size = 16 * 1024,
		.associativity = 4,
		.threads = 1,
		true 
	};
	nnp_hwinfo.cache.l2 = cache_info {
		.size = 128 * 1024,
		.associativity = 4,
		.threads = 1,
		.inclusive = true
	};
	nnp_hwinfo.cache.l3 = cache_info {
		.size = 2 * 1024 * 1024,
		.associativity = 8,
		.threads = 1,
		.inclusive = true
	};
}
#endif

#if !defined(__i386__) && !defined(__x86_64__) && !defined(_MSC_VER) && defined(__APPLE__) 
static void init_static_ios_hwinfo(void) {
	nnp_hwinfo.cache.l1 = cache_info {
		.size = 1 * 32 * 1024,
		.associativity = 1,
		.threads = 1,
		.inclusive = false
	};
	nnp_hwinfo.cache.l2 = cache_info {
		.size = 1024 * 1024,
		.associativity = 1,
		.threads = 1,
		.inclusive = false
	};
	nnp_hwinfo.cache.l3 = cache_info {
		.size = 2 * 1024 * 1024,
		.associativity = 8,
		.threads = 1,
		.inclusive = false
	};
}
#endif

#if !NNP_CONVOLUTION_ONLY
	#if NNP_BACKEND_X86_64
	static const nnp_sdotxf_function sdotxf_function[8] = {
		[0] = nnp_sdotxf1__avx2,
		[1] = nnp_sdotxf2__avx2,
		[2] = nnp_sdotxf3__avx2,
		[3] = nnp_sdotxf4__avx2,
		[4] = nnp_sdotxf5__avx2,
		[5] = nnp_sdotxf6__avx2,
		[6] = nnp_sdotxf7__avx2,
		[7] = nnp_sdotxf8__avx2
	};
	static const nnp_shdotxf_function shdotxf_function[8] = {
		[0] = nnp_shdotxf1__avx2,
		[1] = nnp_shdotxf2__avx2,
		[2] = nnp_shdotxf3__avx2,
		[3] = nnp_shdotxf4__avx2,
		[4] = nnp_shdotxf5__avx2,
		[5] = nnp_shdotxf6__avx2,
		[6] = nnp_shdotxf7__avx2,
		[7] = nnp_shdotxf8__avx2
	};
	#elif NNP_BACKEND_ARM
	static const nnp_sdotxf_function sdotxf_function[8] = {
		[0] = nnp_sdotxf1__neon,
		[1] = nnp_sdotxf2__neon,
		[2] = nnp_sdotxf3__neon,
		[3] = nnp_sdotxf4__neon,
		[4] = nnp_sdotxf5__neon,
		[5] = nnp_sdotxf6__neon,
		[6] = nnp_sdotxf7__neon,
		[7] = nnp_sdotxf8__neon,
	};
	static const nnp_shdotxf_function shdotxf_function[8] = {
		[0] = nnp_shdotxf1__psimd,
		[1] = nnp_shdotxf2__psimd,
		[2] = nnp_shdotxf3__psimd,
		[3] = nnp_shdotxf4__psimd,
		[4] = nnp_shdotxf5__psimd,
		[5] = nnp_shdotxf6__psimd,
		[6] = nnp_shdotxf7__psimd,
		[7] = nnp_shdotxf8__psimd,
	};
	#elif NNP_BACKEND_PSIMD
	static const nnp_sdotxf_function sdotxf_function[8] = {
		[0] = nnp_sdotxf1__psimd,
		[1] = nnp_sdotxf2__psimd,
		[2] = nnp_sdotxf3__psimd,
		[3] = nnp_sdotxf4__psimd,
		[4] = nnp_sdotxf5__psimd,
		[5] = nnp_sdotxf6__psimd,
		[6] = nnp_sdotxf7__psimd,
		[7] = nnp_sdotxf8__psimd,
	};
	static const nnp_shdotxf_function shdotxf_function[8] = {
		[0] = nnp_shdotxf1__psimd,
		[1] = nnp_shdotxf2__psimd,
		[2] = nnp_shdotxf3__psimd,
		[3] = nnp_shdotxf4__psimd,
		[4] = nnp_shdotxf5__psimd,
		[5] = nnp_shdotxf6__psimd,
		[6] = nnp_shdotxf7__psimd,
		[7] = nnp_shdotxf8__psimd,
	};
	#elif NNP_BACKEND_SCALAR
	static const nnp_sdotxf_function sdotxf_function[8] = {
		[0] = nnp_sdotxf1__scalar,
		[1] = nnp_sdotxf2__scalar,
		[2] = nnp_sdotxf3__scalar,
		[3] = nnp_sdotxf4__scalar,
		[4] = nnp_sdotxf5__scalar,
		[5] = nnp_sdotxf6__scalar,
		[6] = nnp_sdotxf7__scalar,
		[7] = nnp_sdotxf8__scalar,
	};
	static const nnp_shdotxf_function shdotxf_function[8] = {
		[0] = nnp_shdotxf1__scalar,
		[1] = nnp_shdotxf2__scalar,
		[2] = nnp_shdotxf3__scalar,
		[3] = nnp_shdotxf4__scalar,
		[4] = nnp_shdotxf5__scalar,
		[5] = nnp_shdotxf6__scalar,
		[6] = nnp_shdotxf7__scalar,
		[7] = nnp_shdotxf8__scalar,
	};
	#endif
#endif /* !NNP_INFERENCE_ONLY */

static void init_hwinfo(void) {
	init_x86_hwinfo();
	
	// Compute high-level cache blocking parameters
	nnp_hwinfo.blocking.l1 = nnp_hwinfo.cache.l1.size;

	if (nnp_hwinfo.cache.l1.threads > 1) 
		nnp_hwinfo.blocking.l1 /= nnp_hwinfo.cache.l1.threads;
	
	if (nnp_hwinfo.cache.l2.size != 0) {
		nnp_hwinfo.blocking.l2 = nnp_hwinfo.cache.l2.size;
		if (nnp_hwinfo.cache.l2.inclusive)
			nnp_hwinfo.blocking.l2 -= nnp_hwinfo.cache.l1.size;
		
		if (nnp_hwinfo.cache.l2.threads > 1)
			nnp_hwinfo.blocking.l2 /= nnp_hwinfo.cache.l2.threads;
	}

	if (nnp_hwinfo.cache.l3.size != 0) {
		nnp_hwinfo.blocking.l3 = nnp_hwinfo.cache.l3.size;
		if (nnp_hwinfo.cache.l3.inclusive)
			nnp_hwinfo.blocking.l3 -= nnp_hwinfo.cache.l2.size;
	}

	nnp_hwinfo.blocking.l4 = nnp_hwinfo.cache.l4.size;

	if (nnp_hwinfo.cache.l1.size && nnp_hwinfo.cache.l2.size && nnp_hwinfo.cache.l3.size) {
#if NNP_BACKEND_X86_64
		if (nnp_hwinfo.isa.has_avx2 && nnp_hwinfo.isa.has_fma3)	{
			nnp_hwinfo.simd_width = 8;
			nnp_hwinfo.transforms.fft8x8_with_offset_and_store = (nnp_transform_2d_with_offset)nnp_fft8x8_with_offset_and_store__avx2;
			nnp_hwinfo.transforms.fft8x8_with_offset_and_stream = (nnp_transform_2d_with_offset)nnp_fft8x8_with_offset_and_stream__avx2;
#if !NNP_INFERENCE_ONLY
			nnp_hwinfo.transforms.ifft8x8_with_offset = (nnp_transform_2d_with_offset)nnp_ifft8x8_with_offset__avx2;
#endif /* !NNP_INFERENCE_ONLY */
			nnp_hwinfo.transforms.ifft8x8_with_bias = (nnp_transform_2d_with_bias)nnp_ifft8x8_with_bias__avx2;
			nnp_hwinfo.transforms.ifft8x8_with_bias_with_relu = (nnp_transform_2d_with_bias)nnp_ifft8x8_with_bias_with_relu__avx2;
			nnp_hwinfo.transforms.fft16x16_with_offset_and_store = (nnp_transform_2d_with_offset)nnp_fft16x16_with_offset_and_store__avx2;
			nnp_hwinfo.transforms.fft16x16_with_offset_and_stream = (nnp_transform_2d_with_offset)nnp_fft16x16_with_offset_and_stream__avx2;
#if !NNP_INFERENCE_ONLY
			nnp_hwinfo.transforms.ifft16x16_with_offset = (nnp_transform_2d_with_offset)nnp_ifft16x16_with_offset__avx2;
#endif /* !NNP_INFERENCE_ONLY */
			nnp_hwinfo.transforms.ifft16x16_with_bias = (nnp_transform_2d_with_bias)nnp_ifft16x16_with_bias__avx2;
			nnp_hwinfo.transforms.ifft16x16_with_bias_with_relu = (nnp_transform_2d_with_bias)nnp_ifft16x16_with_bias_with_relu__avx2;
			nnp_hwinfo.transforms.iwt_f6x6_3x3_with_offset_and_store = (nnp_transform_2d_with_offset)nnp_iwt8x8_3x3_with_offset_and_store__avx2;
			nnp_hwinfo.transforms.iwt_f6x6_3x3_with_offset_and_stream = (nnp_transform_2d_with_offset)nnp_iwt8x8_3x3_with_offset_and_stream__avx2;
			nnp_hwinfo.transforms.kwt_f6x6_3x3 = (nnp_transform_2d_with_offset)nnp_kwt8x8_3x3_and_stream__avx2;
#if !NNP_INFERENCE_ONLY
			nnp_hwinfo.transforms.kwt_f6x6_3Rx3R = (nnp_transform_2d_with_offset)nnp_kwt8x8_3Rx3R_and_stream__avx2;
			nnp_hwinfo.transforms.owt_f6x6_3x3 = (nnp_transform_2d_with_offset)nnp_owt8x8_3x3__avx2;
#endif /* !NNP_INFERENCE_ONLY */
			nnp_hwinfo.transforms.owt_f6x6_3x3_with_bias = (nnp_transform_2d_with_bias)nnp_owt8x8_3x3_with_bias__avx2;
			nnp_hwinfo.transforms.owt_f6x6_3x3_with_bias_with_relu = (nnp_transform_2d_with_bias)nnp_owt8x8_3x3_with_bias_with_relu__avx2;
#if !NNP_CONVOLUTION_ONLY
			nnp_hwinfo.activations.relu = nnp_relu__avx2;
			nnp_hwinfo.activations.inplace_relu = nnp_inplace_relu__avx2;
			nnp_hwinfo.activations.grad_relu = nnp_grad_relu__avx2;
			nnp_hwinfo.activations.softmax = nnp_softmax__avx2;
			nnp_hwinfo.activations.inplace_softmax = nnp_inplace_softmax__avx2;

			nnp_hwinfo.sdotxf = (struct sdotxf) {
				.functions = sdotxf_function,
				.fusion = NNP_COUNT_OF(sdotxf_function)
			};;

			nnp_hwinfo.shdotxf = (struct shdotxf) {
				.functions = shdotxf_function,
				.fusion = NNP_COUNT_OF(shdotxf_function)
			};
#endif /* !NNP_INFERENCE_ONLY */
			nnp_hwinfo.conv1x1 = (struct convolution) {
				.only_mr_x_nr = nnp_conv1x1_only_2x4__fma3,
				.upto_mr_x_nr = nnp_conv1x1_upto_2x4__fma3,
				.mr = 2,
				.nr = 4
			};
			
			nnp_hwinfo.sgemm = (struct sgemm) {
				.only_mr_x_nr = nnp_sgemm_only_4x24__fma3,
				.upto_mr_x_nr = nnp_sgemm_upto_4x24__fma3,
				.mr = 4,
				.nr = 24
			};

			nnp_hwinfo.sxgemm = (struct sxgemm) {
				.only_mr_x_nr = (nnp_fast_tuple_gemm_function)nnp_s8gemm_only_3x4__fma3,
				.upto_mr_x_nr = (nnp_full_tuple_gemm_function)nnp_s8gemm_upto_3x4__fma3,
				.mr = 3,
				.nr = 4
			};

			nnp_hwinfo.cxgemm = (struct cxgemm) {
#if !NNP_INFERENCE_ONLY
				.s4cX_only_mr_x_nr = (nnp_fast_tuple_gemm_function)nnp_s4c6gemm_only_2x2__fma3,
				.s4cX_upto_mr_x_nr = (nnp_full_tuple_gemm_function)nnp_s4c6gemm_upto_2x2__fma3,
				.cX_only_mr_x_nr = (nnp_fast_tuple_gemm_function)nnp_c8gemm_only_2x2__fma3,
				.cX_upto_mr_x_nr = (nnp_full_tuple_gemm_function)nnp_c8gemm_upto_2x2__fma3,
#endif /* !NNP_INFERENCE_ONLY */
				.s4cX_conjb_only_mr_x_nr = (nnp_fast_tuple_gemm_function)nnp_s4c6gemm_conjb_only_2x2__fma3,
				.s4cX_conjb_upto_mr_x_nr = (nnp_full_tuple_gemm_function)nnp_s4c6gemm_conjb_upto_2x2__fma3,
				.cX_conjb_only_mr_x_nr = (nnp_fast_tuple_gemm_function)nnp_c8gemm_conjb_only_2x2__fma3,
				.cX_conjb_upto_mr_x_nr = (nnp_full_tuple_gemm_function)nnp_c8gemm_conjb_upto_2x2__fma3,
#if !NNP_INFERENCE_ONLY
				.s4cX_conjb_transc_only_mr_x_nr = (nnp_fast_tuple_gemm_function)nnp_s4c6gemm_conjb_transc_only_2x2__fma3,
				.s4cX_conjb_transc_upto_mr_x_nr = (nnp_full_tuple_gemm_function)nnp_s4c6gemm_conjb_transc_upto_2x2__fma3,
				.cX_conjb_transc_only_mr_x_nr = (nnp_fast_tuple_gemm_function)nnp_c8gemm_conjb_transc_only_2x2__fma3,
				.cX_conjb_transc_upto_mr_x_nr = (nnp_full_tuple_gemm_function)nnp_c8gemm_conjb_transc_upto_2x2__fma3,
#endif /* !NNP_INFERENCE_ONLY */
				.mr = 2,
				.nr = 2
			};
		
			nnp_hwinfo.supported = true;
		}
#elif NNP_BACKEND_PSIMD
		nnp_hwinfo.simd_width = 4;
		nnp_hwinfo.transforms.fft8x8_with_offset_and_store = (nnp_transform_2d_with_offset)nnp_fft8x8_with_offset__psimd;
		nnp_hwinfo.transforms.fft8x8_with_offset_and_stream = (nnp_transform_2d_with_offset)nnp_fft8x8_with_offset__psimd;
#if !NNP_INFERENCE_ONLY
		nnp_hwinfo.transforms.ifft8x8_with_offset = (nnp_transform_2d_with_offset)nnp_ifft8x8_with_offset__psimd;
#endif /* !NNP_INFERENCE_ONLY */
		nnp_hwinfo.transforms.ifft8x8_with_bias = (nnp_transform_2d_with_bias)nnp_ifft8x8_with_bias__psimd;
		nnp_hwinfo.transforms.ifft8x8_with_bias_with_relu = (nnp_transform_2d_with_bias)nnp_ifft8x8_with_bias_with_relu__psimd;
		nnp_hwinfo.transforms.fft16x16_with_offset_and_store = (nnp_transform_2d_with_offset)nnp_fft16x16_with_offset__psimd;
		nnp_hwinfo.transforms.fft16x16_with_offset_and_stream = (nnp_transform_2d_with_offset)nnp_fft16x16_with_offset__psimd;
#if !NNP_INFERENCE_ONLY
		nnp_hwinfo.transforms.ifft16x16_with_offset = (nnp_transform_2d_with_offset)nnp_ifft16x16_with_offset__psimd;
#endif /* !NNP_INFERENCE_ONLY */
		nnp_hwinfo.transforms.ifft16x16_with_bias = (nnp_transform_2d_with_bias)nnp_ifft16x16_with_bias__psimd;
		nnp_hwinfo.transforms.ifft16x16_with_bias_with_relu = (nnp_transform_2d_with_bias)nnp_ifft16x16_with_bias_with_relu__psimd;
		nnp_hwinfo.transforms.iwt_f6x6_3x3_with_offset_and_store = (nnp_transform_2d_with_offset)nnp_iwt8x8_3x3_with_offset__psimd;
		nnp_hwinfo.transforms.iwt_f6x6_3x3_with_offset_and_stream = (nnp_transform_2d_with_offset)nnp_iwt8x8_3x3_with_offset__psimd;
		nnp_hwinfo.transforms.kwt_f6x6_3x3 = (nnp_transform_2d_with_offset)nnp_kwt8x8_3x3__psimd;
#if !NNP_INFERENCE_ONLY
		nnp_hwinfo.transforms.kwt_f6x6_3Rx3R = (nnp_transform_2d_with_offset)nnp_kwt8x8_3Rx3R__psimd;
		nnp_hwinfo.transforms.owt_f6x6_3x3 = (nnp_transform_2d_with_offset)nnp_owt8x8_3x3__psimd;
#endif /* !NNP_INFERENCE_ONLY */
		nnp_hwinfo.transforms.owt_f6x6_3x3_with_bias = (nnp_transform_2d_with_bias)nnp_owt8x8_3x3_with_bias__psimd;
		nnp_hwinfo.transforms.owt_f6x6_3x3_with_bias_with_relu = (nnp_transform_2d_with_bias)nnp_owt8x8_3x3_with_bias_with_relu__psimd;
#if !NNP_CONVOLUTION_ONLY
		nnp_hwinfo.activations.relu = nnp_relu__psimd;
		nnp_hwinfo.activations.inplace_relu = nnp_inplace_relu__psimd;
		nnp_hwinfo.activations.grad_relu = nnp_grad_relu__psimd;
		nnp_hwinfo.activations.softmax = nnp_softmax__psimd;
		nnp_hwinfo.activations.inplace_softmax = nnp_inplace_softmax__psimd;
		nnp_hwinfo.sdotxf = (struct sdotxf) {
			.functions = sdotxf_function,
			.fusion = NNP_COUNT_OF(sdotxf_function)
		};
		nnp_hwinfo.shdotxf = (struct shdotxf) {
			.functions = shdotxf_function,
			.fusion = NNP_COUNT_OF(shdotxf_function)
		};
#endif /* !NNP_INFERENCE_ONLY */
		nnp_hwinfo.conv1x1 = (struct convolution) {
			.only_mr_x_nr = nnp_conv1x1_only_2x4__psimd,
			.upto_mr_x_nr = nnp_conv1x1_upto_2x4__psimd,
			.mr = 2,
			.nr = 4
		};
		nnp_hwinfo.sgemm = (struct sgemm) {
			.only_mr_x_nr = nnp_sgemm_only_4x8__psimd,
			.upto_mr_x_nr = nnp_sgemm_upto_4x8__psimd,
			.mr = 4,
			.nr = 8
		};
		nnp_hwinfo.sxgemm = (struct sxgemm) {
			.only_mr_x_nr = (nnp_fast_tuple_gemm_function)nnp_s4gemm_only_3x4__psimd,
			.upto_mr_x_nr = (nnp_full_tuple_gemm_function)nnp_s4gemm_upto_3x4__psimd,
			.mr = 3,
			.nr = 4
		};
		nnp_hwinfo.cxgemm = (struct cxgemm) {
#if !NNP_INFERENCE_ONLY
			.s4cX_only_mr_x_nr = (nnp_fast_tuple_gemm_function)nnp_s4c2gemm_only_2x2__psimd,
			.s4cX_upto_mr_x_nr = (nnp_full_tuple_gemm_function)nnp_s4c2gemm_upto_2x2__psimd,
			.cX_only_mr_x_nr = (nnp_fast_tuple_gemm_function)nnp_c4gemm_only_2x2__psimd,
			.cX_upto_mr_x_nr = (nnp_full_tuple_gemm_function)nnp_c4gemm_upto_2x2__psimd,
#endif /* !NNP_INFERENCE_ONLY */
			.s4cX_conjb_only_mr_x_nr = (nnp_fast_tuple_gemm_function)nnp_s4c2gemm_conjb_only_2x2__psimd,
			.s4cX_conjb_upto_mr_x_nr = (nnp_full_tuple_gemm_function)nnp_s4c2gemm_conjb_upto_2x2__psimd,
			.cX_conjb_only_mr_x_nr = (nnp_fast_tuple_gemm_function)nnp_c4gemm_conjb_only_2x2__psimd,
			.cX_conjb_upto_mr_x_nr = (nnp_full_tuple_gemm_function)nnp_c4gemm_conjb_upto_2x2__psimd,
#if !NNP_INFERENCE_ONLY
			.s4cX_conjb_transc_only_mr_x_nr = (nnp_fast_tuple_gemm_function)nnp_s4c2gemm_conjb_transc_only_2x2__psimd,
			.s4cX_conjb_transc_upto_mr_x_nr = (nnp_full_tuple_gemm_function)nnp_s4c2gemm_conjb_transc_upto_2x2__psimd,
			.cX_conjb_transc_only_mr_x_nr = (nnp_fast_tuple_gemm_function)nnp_c4gemm_conjb_transc_only_2x2__psimd,
			.cX_conjb_transc_upto_mr_x_nr = (nnp_full_tuple_gemm_function)nnp_c4gemm_conjb_transc_upto_2x2__psimd,
#endif /* !NNP_INFERENCE_ONLY */
			.mr = 2,
			.nr = 2
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
#if !NNP_INFERENCE_ONLY
		nnp_hwinfo.transforms.ifft8x8_with_offset = (nnp_transform_2d_with_offset)nnp_ifft8x8_with_offset__psimd;
#endif /* !NNP_INFERENCE_ONLY */
		nnp_hwinfo.transforms.ifft8x8_with_bias = (nnp_transform_2d_with_bias)nnp_ifft8x8_with_bias__psimd;
		nnp_hwinfo.transforms.ifft8x8_with_bias_with_relu = (nnp_transform_2d_with_bias)nnp_ifft8x8_with_bias_with_relu__psimd;
		nnp_hwinfo.transforms.fft16x16_with_offset_and_store = (nnp_transform_2d_with_offset)nnp_fft16x16_with_offset__psimd;
		nnp_hwinfo.transforms.fft16x16_with_offset_and_stream = (nnp_transform_2d_with_offset)nnp_fft16x16_with_offset__psimd;
#if !NNP_INFERENCE_ONLY
		nnp_hwinfo.transforms.ifft16x16_with_offset = (nnp_transform_2d_with_offset)nnp_ifft16x16_with_offset__psimd;
#endif /* !NNP_INFERENCE_ONLY */
		nnp_hwinfo.transforms.ifft16x16_with_bias = (nnp_transform_2d_with_bias)nnp_ifft16x16_with_bias__psimd;
		nnp_hwinfo.transforms.ifft16x16_with_bias_with_relu = (nnp_transform_2d_with_bias)nnp_ifft16x16_with_bias_with_relu__psimd;
		nnp_hwinfo.transforms.iwt_f6x6_3x3_with_offset_and_store = (nnp_transform_2d_with_offset)nnp_iwt8x8_3x3_with_offset__neon;
		nnp_hwinfo.transforms.iwt_f6x6_3x3_with_offset_and_stream = (nnp_transform_2d_with_offset)nnp_iwt8x8_3x3_with_offset__neon;
		nnp_hwinfo.transforms.kwt_f6x6_3x3 = (nnp_transform_2d_with_offset)nnp_kwt8x8_3x3__neon;
#if !NNP_INFERENCE_ONLY
		nnp_hwinfo.transforms.kwt_f6x6_3Rx3R = (nnp_transform_2d_with_offset)nnp_kwt8x8_3Rx3R__neon;
		nnp_hwinfo.transforms.owt_f6x6_3x3 = (nnp_transform_2d_with_offset)nnp_owt8x8_3x3__neon;
#endif /* !NNP_INFERENCE_ONLY */
		nnp_hwinfo.transforms.owt_f6x6_3x3_with_bias = (nnp_transform_2d_with_bias)nnp_owt8x8_3x3_with_bias__neon;
		nnp_hwinfo.transforms.owt_f6x6_3x3_with_bias_with_relu = (nnp_transform_2d_with_bias)nnp_owt8x8_3x3_with_bias_with_relu__neon;
		if (has_fp16) {
			nnp_hwinfo.transforms.iwt_f6x6_3x3_fp16_with_offset = (nnp_transform_2d_with_offset)nnp_iwt8x8_3x3_fp16_with_offset__neonhp;
			nnp_hwinfo.transforms.kwt_f6x6_3x3_fp16 = (nnp_transform_2d_with_offset)nnp_kwt8x8_3x3_fp16__neonhp;
			nnp_hwinfo.transforms.owt_f6x6_3x3_fp16_with_bias = (nnp_transform_2d_with_bias)nnp_owt8x8_3x3_fp16_with_bias__neonhp;
			nnp_hwinfo.transforms.owt_f6x6_3x3_fp16_with_bias_with_relu = (nnp_transform_2d_with_bias)nnp_owt8x8_3x3_fp16_with_bias_with_relu__neonhp;
		}
#if !!NNP_CONVOLUTION_ONLY
		nnp_hwinfo.activations.relu = nnp_relu__neon;
		nnp_hwinfo.activations.inplace_relu = nnp_inplace_relu__neon;
		nnp_hwinfo.activations.grad_relu = nnp_grad_relu__neon;
		nnp_hwinfo.activations.softmax = nnp_softmax__psimd;
		nnp_hwinfo.activations.inplace_softmax = nnp_inplace_softmax__psimd;
		nnp_hwinfo.sdotxf = (struct sdotxf) {
			.functions = sdotxf_function,
			.fusion = NNP_COUNT_OF(sdotxf_function)
		};
		nnp_hwinfo.shdotxf = (struct shdotxf) {
			.functions = shdotxf_function,
			.fusion = NNP_COUNT_OF(shdotxf_function)
		};
#endif /* !NNP_CONVOLUTION_ONLY */
		nnp_hwinfo.conv1x1 = (struct convolution) {
			.only_mr_x_nr nnp_conv1x1_only_2x4__neon,
			.upto_mr_x_nr nnp_conv1x1_upto_2x4__neon,
			.mr = 2,
			.nr = 4
		};
		nnp_hwinfo.sgemm = (struct sgemm) {
			.mr = 6,
			.nr = 8,
#if CPUINFO_ARCH_ARM
			.only_mr_x_nr = nnp_sgemm_only_6x8__aarch32_neon,
#else
			.only_mr_x_nr = nnp_sgemm_only_6x8__neon,
#endif
			.upto_mr_x_nr = nnp_sgemm_upto_6x8__neon,
		};
		nnp_hwinfo.sxgemm = (struct sxgemm) {
			.only_mr_x_nr = (nnp_fast_tuple_gemm_function)nnp_s4gemm_only_3x4__neon,
			.upto_mr_x_nr = (nnp_full_tuple_gemm_function)nnp_s4gemm_upto_3x4__neon,
			.mr = 3,
			.nr = 4
		};
		if (has_fp16) {
			nnp_hwinfo.hxgemm = (struct hxgemm) {
				.only_mr_x_nr = (nnp_fast_tuple_gemm_function)nnp_h4gemm_only_3x4__neonhp,
				.upto_mr_x_nr = (nnp_full_tuple_gemm_function)nnp_h4gemm_upto_3x4__neonhp,
				.mr = 3,
				.br = 4
			};
		}
		nnp_hwinfo.cxgemm = (struct cxgemm) {
#if !NNP_INFERENCE_ONLY
			.s4cX_only_mr_x_nr = (nnp_fast_tuple_gemm_function)nnp_s4c2gemm_only_2x2__neon,
			.s4cX_upto_mr_x_nr = (nnp_full_tuple_gemm_function)nnp_s4c2gemm_upto_2x2__neon,
			.cX_only_mr_x_nr = (nnp_fast_tuple_gemm_function)nnp_c4gemm_only_2x2__neon,
			.cX_upto_mr_x_nr = (nnp_full_tuple_gemm_function)nnp_c4gemm_upto_2x2__neon,
#endif /* !NNP_INFERENCE_ONLY */
			.s4cX_conjb_only_mr_x_nr = (nnp_fast_tuple_gemm_function)nnp_s4c2gemm_conjb_only_2x2__neon,
			.s4cX_conjb_upto_mr_x_nr = (nnp_full_tuple_gemm_function)nnp_s4c2gemm_conjb_upto_2x2__neon,
			.cX_conjb_only_mr_x_nr = (nnp_fast_tuple_gemm_function)nnp_c4gemm_conjb_only_2x2__neon,
			.cX_conjb_upto_mr_x_nr = (nnp_full_tuple_gemm_function)nnp_c4gemm_conjb_upto_2x2__neon,
#if !NNP_INFERENCE_ONLY
			.s4cX_conjb_transc_only_mr_x_nr = (nnp_fast_tuple_gemm_function)nnp_s4c2gemm_conjb_transc_only_2x2__neon,
			.s4cX_conjb_transc_upto_mr_x_nr = (nnp_full_tuple_gemm_function)nnp_s4c2gemm_conjb_transc_upto_2x2__neon,
			.cX_conjb_transc_only_mr_x_nr = (nnp_fast_tuple_gemm_function)nnp_c4gemm_conjb_transc_only_2x2__neon,
			.cX_conjb_transc_upto_mr_x_nr = (nnp_full_tuple_gemm_function)nnp_c4gemm_conjb_transc_upto_2x2__neon,
#endif /* !NNP_INFERENCE_ONLY */
			.mr = 2,
			.nr = 2
		};

#if defined(__ANDROID__) && defined(__arm__) && !defined(__aarch64__)
		nnp_hwinfo.supported = (android_getCpuFeatures() & ANDROID_CPU_ARM_FEATURE_NEON) != 0;
#else
		nnp_hwinfo.supported = true;
#endif
#elif NNP_BACKEND_SCALAR
		nnp_hwinfo.simd_width = 1u;
		nnp_hwinfo.transforms.fft8x8_with_offset_and_store = (nnp_transform_2d_with_offset)nnp_fft8x8_with_offset__scalar;
		nnp_hwinfo.transforms.fft8x8_with_offset_and_stream = (nnp_transform_2d_with_offset)nnp_fft8x8_with_offset__scalar;
#if !NNP_CONVOLUTION_ONLY
		nnp_hwinfo.transforms.ifft8x8_with_offset = (nnp_transform_2d_with_offset)nnp_ifft8x8_with_offset__scalar;
#endif /* !NNP_INFERENCE_ONLY */
		nnp_hwinfo.transforms.ifft8x8_with_bias = (nnp_transform_2d_with_bias)nnp_ifft8x8_with_bias__scalar;
		nnp_hwinfo.transforms.ifft8x8_with_bias_with_relu = (nnp_transform_2d_with_bias)nnp_ifft8x8_with_bias_with_relu__scalar;
		nnp_hwinfo.transforms.fft16x16_with_offset_and_store = (nnp_transform_2d_with_offset)nnp_fft16x16_with_offset__scalar;
		nnp_hwinfo.transforms.fft16x16_with_offset_and_stream = (nnp_transform_2d_with_offset)nnp_fft16x16_with_offset__scalar;
#if !NNP_CONVOLUTION_ONLY
		nnp_hwinfo.transforms.ifft16x16_with_offset = (nnp_transform_2d_with_offset)nnp_ifft16x16_with_offset__scalar;
#endif /* !NNP_INFERENCE_ONLY */
		nnp_hwinfo.transforms.ifft16x16_with_bias = (nnp_transform_2d_with_bias)nnp_ifft16x16_with_bias__scalar;
		nnp_hwinfo.transforms.ifft16x16_with_bias_with_relu = (nnp_transform_2d_with_bias)nnp_ifft16x16_with_bias_with_relu__scalar;
		nnp_hwinfo.transforms.iwt_f6x6_3x3_with_offset_and_store = (nnp_transform_2d_with_offset)nnp_iwt8x8_3x3_with_offset__scalar;
		nnp_hwinfo.transforms.iwt_f6x6_3x3_with_offset_and_stream = (nnp_transform_2d_with_offset)nnp_iwt8x8_3x3_with_offset__scalar;
		nnp_hwinfo.transforms.kwt_f6x6_3x3 = (nnp_transform_2d_with_offset)nnp_kwt8x8_3x3__scalar;
#if !NNP_CONVOLUTION_ONLY
		nnp_hwinfo.transforms.kwt_f6x6_3Rx3R = (nnp_transform_2d_with_offset)nnp_kwt8x8_3Rx3R__scalar;
		nnp_hwinfo.transforms.owt_f6x6_3x3 = (nnp_transform_2d_with_offset)nnp_owt8x8_3x3__scalar;
#endif /* !NNP_INFERENCE_ONLY */
		nnp_hwinfo.transforms.owt_f6x6_3x3_with_bias = (nnp_transform_2d_with_bias)nnp_owt8x8_3x3_with_bias__scalar;
		nnp_hwinfo.transforms.owt_f6x6_3x3_with_bias_with_relu = (nnp_transform_2d_with_bias)nnp_owt8x8_3x3_with_bias_with_relu__scalar;
#if !NNP_CONVOLUTION_ONLY
		nnp_hwinfo.activations.relu = nnp_relu__scalar;
		nnp_hwinfo.activations.inplace_relu = nnp_inplace_relu__scalar;
		nnp_hwinfo.activations.grad_relu = nnp_grad_relu__scalar;
		nnp_hwinfo.activations.softmax = nnp_softmax__scalar;
		nnp_hwinfo.activations.inplace_softmax = nnp_inplace_softmax__scalar;

		nnp_hwinfo.sdotxf = (struct sdotxf) {
			.functions = sdotxf_function,
			.fusion = NNP_COUNT_OF(sdotxf_function),
		};
		nnp_hwinfo.shdotxf = (struct shdotxf) {
			.functions = shdotxf_function,
			.fusion = NNP_COUNT_OF(shdotxf_function),
		};
#endif /* !NNP_INFERENCE_ONLY */
		nnp_hwinfo.conv1x1 = (struct convolution) {
			.only_mr_x_nr nnp_conv1x1_only_2x4__scalar,
			.upto_mr_x_nr nnp_conv1x1_upto_2x4__scalar,
			.mr = 2,
			.nr = 4
		};
		nnp_hwinfo.sgemm = (struct sgemm) {
			.only_mr_x_nr = nnp_sgemm_only_4x3__scalar,
			.upto_mr_x_nr = nnp_sgemm_upto_4x3__scalar,
			.mr = 4,
			.nr = 3
		};
		nnp_hwinfo.sxgemm = (struct sxgemm) {
			.only_mr_x_nr = (nnp_fast_tuple_gemm_function)nnp_sgemm_only_4x3__scalar,
			.upto_mr_x_nr = (nnp_full_tuple_gemm_function)nnp_sgemm_upto_4x3__scalar,
			.mr = 4,
			.nr = 3
		};
		nnp_hwinfo.cxgemm = (struct cxgemm) {
#if !NNP_CONVOLUTION_ONLY
			.s4cX_only_mr_x_nr = (nnp_fast_tuple_gemm_function)nnp_s2gemm_only_2x2__scalar,
			.s4cX_upto_mr_x_nr = (nnp_full_tuple_gemm_function)nnp_s2gemm_upto_2x2__scalar,
			.cX_only_mr_x_nr = (nnp_fast_tuple_gemm_function)nnp_cgemm_only_2x2__scalar,
			.cX_upto_mr_x_nr = (nnp_full_tuple_gemm_function)nnp_cgemm_upto_2x2__scalar,
#endif /* !NNP_INFERENCE_ONLY */
			.s4cX_conjb_only_mr_x_nr = (nnp_fast_tuple_gemm_function)nnp_s2gemm_only_2x2__scalar,
			.s4cX_conjb_upto_mr_x_nr = (nnp_full_tuple_gemm_function)nnp_s2gemm_upto_2x2__scalar,
			.cX_conjb_only_mr_x_nr = (nnp_fast_tuple_gemm_function)nnp_cgemm_conjb_only_2x2__scalar,
			.cX_conjb_upto_mr_x_nr = (nnp_full_tuple_gemm_function)nnp_cgemm_conjb_upto_2x2__scalar,
#if !NNP_CONVOLUTION_ONLY
			.s4cX_conjb_transc_only_mr_x_nr = (nnp_fast_tuple_gemm_function)nnp_s2gemm_transc_only_2x2__scalar,
			.s4cX_conjb_transc_upto_mr_x_nr = (nnp_full_tuple_gemm_function)nnp_s2gemm_transc_upto_2x2__scalar,
			.cX_conjb_transc_only_mr_x_nr = (nnp_fast_tuple_gemm_function)nnp_cgemm_conjb_transc_only_2x2__scalar,
			.cX_conjb_transc_upto_mr_x_nr = (nnp_full_tuple_gemm_function)nnp_cgemm_conjb_transc_upto_2x2__scalar,
#endif /* !NNP_INFERENCE_ONLY */
			.mr = 2,
			.nr = 2
		};

		nnp_hwinfo.supported = true;
#else
#error Unsupported backend
#endif
	}

	nnp_hwinfo.initialized = true;
}

enum nnp_status nnp_initialize() {

#ifdef _MSC_VER
	// Flush denormals to zero (the FTZ flag).
	// _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
	// Interpret denormal inputs as zero (the DAZ flag).
	// _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);

	init_hwinfo();
#else
	pthread_once(&hwinfo_init_control, &init_hwinfo);
#endif

	if (nnp_hwinfo.supported)
		return nnp_status_success;
	else
		return nnp_status_unsupported_hardware;
}

enum nnp_status nnp_deinitialize() {
	return nnp_status_success;
}
