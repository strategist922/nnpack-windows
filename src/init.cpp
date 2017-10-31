#include <intrin.h>

#include <nnpack.h>
#include <hwinfo.h>
#include <blas.h>
#include <transform.h>
#include <relu.h>
#include <softmax.h>



hardware_info nnp_hwinfo = {  };

struct cpu_info
{
	int eax;
	int ebx;
	int ecx;
	int edx;
};

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
	

static void init_x86_hwinfo() 
{
	const uint32_t max_base_info = __get_cpuid_max(0, NULL);
	const uint32_t max_extended_info = __get_cpuid_max(0x80000000, NULL);

	// Under normal environments, just ask the CPU about supported ISA extensions.
	if (max_base_info >= 1)
	{
		cpu_info basic_info;
		__cpuid(&basic_info.eax, 1);
		
		// OSXSAVE: ecx[bit 27] in basic info
		const bool osxsave = !!(basic_info.ecx & bit_OSXSAVE);
		// Check that AVX[bit 2] and SSE[bit 1] registers are preserved by OS
		const bool ymm_regs = (osxsave ? ((xgetbv(0) & 0b110ul) == 0b110ul) : false);

		cpu_info structured_info = { 0 };
		if (max_base_info >= 7)
			__cpuidex(&structured_info.eax, 7, 0);
		
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
	__cpuid(&vendor_info.eax, 0);
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

			__cpuidex(&cpuInfo.eax, 4, cache_id);
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
}


static const nnp_sdotxf_function sdotxf_function[8] = 
{
	nnp_sdotxf1__avx2,
	nnp_sdotxf2__avx2,
	nnp_sdotxf3__avx2,
	nnp_sdotxf4__avx2,
	nnp_sdotxf5__avx2,
	nnp_sdotxf6__avx2,
	nnp_sdotxf7__avx2,
	nnp_sdotxf8__avx2
};

static const nnp_shdotxf_function shdotxf_function[8] =
{
	nnp_shdotxf1__avx2,
	nnp_shdotxf2__avx2,
	nnp_shdotxf3__avx2,
	nnp_shdotxf4__avx2,
	nnp_shdotxf5__avx2,
	nnp_shdotxf6__avx2,
	nnp_shdotxf7__avx2,
	nnp_shdotxf8__avx2
};


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
	}
	
	nnp_hwinfo.initialized = true;
}

nnp_status nnp_initialize()
{
	init_hwinfo();
	
	if (nnp_hwinfo.supported)
		return nnp_status_success;
	else
		return nnp_status_unsupported_hardware;
}

nnp_status nnp_deinitialize() 
{
	return nnp_status_success;
}
