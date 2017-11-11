#include <nnpack.h>
#include <utils.h>
#include <hwinfo.h>
#include <validation.h>
#include <system.h>
#include <macros.h>

struct NNP_CACHE_ALIGN input_packing_context 
{
	const float* matrix;
	float* packed_matrix;
	size_t input_channels;
	size_t outer_subblock_max;
};

static void pack_input_matrix(
	const struct input_packing_context* context,
	const size_t outer_block_start,
	const size_t input_channels_block_start,
	const size_t outer_block_size,
	const size_t input_channels_block_size)
{
	const float* matrix             = context->matrix;
	float* packed_matrix            = context->packed_matrix;
	const size_t input_channels     = context->input_channels;
	const size_t outer_subblock_max = context->outer_subblock_max;

	for (size_t outer_subblock_start = 0; outer_subblock_start < outer_block_size; outer_subblock_start += outer_subblock_max) 
	{
		const size_t outer_subblock_size = min(outer_block_size - outer_subblock_start, outer_subblock_max);
		for (size_t input_channels_block_offset = 0; input_channels_block_offset < input_channels_block_size; input_channels_block_offset++) 
		{
			const size_t input_channel = input_channels_block_start + input_channels_block_offset;
			for (size_t outer_subblock_offset = 0; outer_subblock_offset < outer_subblock_size; outer_subblock_offset++) 
			{
				const size_t index = (outer_block_start + outer_subblock_start + outer_subblock_offset) * input_channels + input_channel;
				const size_t packed_index = outer_block_start * input_channels + input_channels_block_start * outer_block_size + outer_subblock_start * input_channels_block_size + input_channels_block_offset * outer_subblock_size + outer_subblock_offset;
				packed_matrix[packed_index] = matrix[index];
			}
		}
	}
}

struct NNP_CACHE_ALIGN kernel_packing_context 
{
	const float* matrix;
	float* packed_matrix;

	size_t simd_width;
	size_t input_channels;
	size_t outer_subblock_max;
	size_t input_channels_block_start;
	size_t input_channels_block_size;
};

static void pack_kernel_matrix(
	const struct kernel_packing_context* context,
	const size_t outer_block_start,
	const size_t outer_block_size)
{
	const float* matrix                     = context->matrix;
	float* packed_matrix                    = context->packed_matrix;
	const size_t input_channels             = context->input_channels;
	const size_t outer_subblock_max         = context->outer_subblock_max;
	const size_t input_channels_block_start = context->input_channels_block_start;
	const size_t input_channels_block_size  = context->input_channels_block_size;
	const size_t simd_width                 = context->simd_width;

	for (size_t outer_subblock_start = 0; outer_subblock_start < outer_block_size; outer_subblock_start += outer_subblock_max) 
	{
		const size_t outer_subblock_size   = min(outer_block_size - outer_subblock_start, outer_subblock_max);
		const size_t outer_subblock_stride = round_up(outer_subblock_size, simd_width);
		for (size_t input_channels_block_offset = 0; input_channels_block_offset < input_channels_block_size; input_channels_block_offset++) 
		{
			const size_t input_channel = input_channels_block_start + input_channels_block_offset;
			for (size_t outer_subblock_offset = 0; outer_subblock_offset < outer_subblock_size; outer_subblock_offset++) 
			{
				const size_t index = (outer_block_start + outer_subblock_start + outer_subblock_offset) * input_channels + input_channel;
				const size_t packed_index = (outer_block_start + outer_subblock_start) * input_channels_block_size + input_channels_block_offset * outer_subblock_stride + outer_subblock_offset;
				packed_matrix[packed_index] = matrix[index];
			}
		}
	}
}

struct NNP_CACHE_ALIGN matrix_multiplication_context 
{
	const float* input;
	const float* kernel;
	float* output;
	const size_t input_channels;
	const size_t output_channels;
	size_t batch_block_start;
	size_t batch_block_size;
	size_t input_channels_block_start;
	size_t input_channels_block_size;
	const size_t output_channels_subblock_max;
	const size_t batch_subblock_max;
	const size_t simd_width;
	nnp_fast_sgemm_function fast_sgemm_function;
	nnp_full_sgemm_function full_sgemm_function;
};

static void compute_matrix_multiplication(
	const struct matrix_multiplication_context* context,
	const size_t output_channels_block_start,
	const size_t batch_subblock_start,
	const size_t output_channels_block_size,
	const size_t batch_subblock_size)
{
	const float* input                       = context->input;
	const float* kernel                      = context->kernel;
	float* output                            = context->output;
	const size_t input_channels              = context->input_channels;
	const size_t output_channels             = context->output_channels;
	const size_t input_channels_block_start   = context->input_channels_block_start;
	const size_t input_channels_block_size    = context->input_channels_block_size;
	const size_t batch_block_start           = context->batch_block_start;
	const size_t batch_block_size            = context->batch_block_size;
	const size_t output_channels_subblock_max = context->output_channels_subblock_max;
	const size_t batch_subblock_max          = context->batch_subblock_max;
	const size_t simd_width                  = context->simd_width;
	const nnp_fast_sgemm_function fast_sgemm = context->fast_sgemm_function;
	const nnp_full_sgemm_function full_sgemm = context->full_sgemm_function;

	for (size_t output_channels_subblock_start = 0; output_channels_subblock_start < output_channels_block_size; output_channels_subblock_start += output_channels_subblock_max) 
	{
		const size_t output_channels_subblock_size = min(output_channels_block_size - output_channels_subblock_start, output_channels_subblock_max);
		if ((batch_subblock_size == batch_subblock_max) && (output_channels_subblock_size == output_channels_subblock_max)) 
		{
			fast_sgemm(
				input_channels_block_size, input_channels_block_start,
				&input[batch_block_start * input_channels + input_channels_block_start * batch_block_size + batch_subblock_start * input_channels_block_size],
				&kernel[(output_channels_block_start + output_channels_subblock_start) * input_channels_block_size],
				&output[(batch_block_start + batch_subblock_start) * output_channels + (output_channels_block_start + output_channels_subblock_start)],
				output_channels);
		} 
		else 
		{
			full_sgemm(
				batch_subblock_size, output_channels_subblock_size,
				input_channels_block_size, input_channels_block_start,
				&input[batch_block_start * input_channels + input_channels_block_start * batch_block_size + batch_subblock_start * input_channels_block_size],
				&kernel[(output_channels_block_start + output_channels_subblock_start) * input_channels_block_size],
				&output[(batch_block_start + batch_subblock_start) * output_channels + (output_channels_block_start + output_channels_subblock_start)],
				output_channels);
		}
	}
}

static void compute_fully_connected_output(
	const size_t simd_width,
	const size_t batch_size,
	const size_t batch_block_max,
	const size_t batch_subblock_max,
	const size_t input_channels,
	const size_t input_channels_block_max,
	const size_t output_channels,
	const size_t output_channels_block_max,
	const size_t output_channels_subblock_max,
	const float* input,	
	const float* kernel, 
	float* output,
	float* packed_input, 
	float* packed_kernel,
	struct nnp_profile* profile)
{
	NNP_INPUT_TRANSFORM_START(profile)
	struct input_packing_context input_packing_context = 
	{
		input,
		packed_input,
		input_channels,
		batch_subblock_max
	};
	pthreadpool_compute_2d_tiled(
		(pthreadpool_function_2d_tiled_t)pack_input_matrix,
		&input_packing_context,
		batch_size, input_channels,
		batch_block_max, input_channels_block_max);
	NNP_INPUT_TRANSFORM_END(profile)

	struct matrix_multiplication_context matrix_multiplication_context = 
	{
		packed_input,
		packed_kernel,
		output,
		input_channels,
		output_channels,
		0,
		0,
		0,
		0,
		output_channels_subblock_max,
		batch_subblock_max,
		simd_width,
		nnp_hwinfo.sgemm.only_mr_x_nr,
		nnp_hwinfo.sgemm.upto_mr_x_nr
	};
	
	for (size_t input_channels_block_start = 0; input_channels_block_start < input_channels; input_channels_block_start += input_channels_block_max) 
	{
		const size_t input_channels_block_size = min(input_channels - input_channels_block_start, input_channels_block_max);

		NNP_KERNEL_TRANSFORM_START(profile)
		struct kernel_packing_context kernel_packing_context = 
		{
			kernel,
			packed_kernel,
			simd_width,
			input_channels,
			output_channels_subblock_max,
			input_channels_block_start,
			input_channels_block_size
		};
		pthreadpool_compute_1d_tiled(
			(pthreadpool_function_1d_tiled_t)pack_kernel_matrix,
			&kernel_packing_context,
			output_channels, output_channels_block_max);
		NNP_KERNEL_TRANSFORM_END(profile)

		NNP_BLOCK_MULTIPLICATION_START(profile)
		matrix_multiplication_context.input_channels_block_start = input_channels_block_start;
		matrix_multiplication_context.input_channels_block_size = input_channels_block_size;
		for (size_t batch_block_start = 0; batch_block_start < batch_size; batch_block_start += batch_block_max) 
		{
			const size_t batch_block_size = min(batch_size - batch_block_start, batch_block_max);

			matrix_multiplication_context.batch_block_start = batch_block_start;
			matrix_multiplication_context.batch_block_size = batch_block_size;
			pthreadpool_compute_2d_tiled(
				(pthreadpool_function_2d_tiled_t)compute_matrix_multiplication,
				&matrix_multiplication_context,
				output_channels,           batch_block_size,
				output_channels_block_max, batch_subblock_max);
		}
		NNP_BLOCK_MULTIPLICATION_END(profile)
	}
}

enum nnp_status nnp_fully_connected_output(
	const size_t batch_size,
	const size_t input_channels,
	const size_t output_channels,
	const float* input,
	const float* kernel,
	float* output,
	struct nnp_profile* profile)
{
	NNP_TOTAL_START(profile)

	/* Basic validation of parameters. This check detects invalid, but not unsupported parameters. */
	enum nnp_status status = validate_fully_connected_arguments(batch_size, input_channels, output_channels);
	if (status != nnp_status_success)
		return status;
	
	const size_t cache_elements_l1 = nnp_hwinfo.blocking.l1 / sizeof(float);
	const size_t cache_elements_l2 = nnp_hwinfo.blocking.l2 / sizeof(float);
	const size_t cache_elements_l3 = nnp_hwinfo.blocking.l3 / sizeof(float);

	const size_t simd_width = nnp_hwinfo.simd_width;
	const size_t batch_subblock_max = nnp_hwinfo.sgemm.mr;
	const size_t output_channels_subblock_max = nnp_hwinfo.sgemm.nr;

	const size_t input_channels_block_max = cache_elements_l1 / (batch_subblock_max + output_channels_subblock_max);
	const size_t batch_block_max = round_down(cache_elements_l3 / input_channels_block_max, batch_subblock_max);
	const size_t output_channels_block_max = round_down(cache_elements_l2 / input_channels_block_max, output_channels_subblock_max);

	/* Calculate memory footprint and allocate memory */
	const size_t packed_input_size = round_up(round_up(batch_size, batch_subblock_max) * input_channels * sizeof(float), 64);
	const size_t packed_kernel_size = round_up(round_up(output_channels, output_channels_subblock_max) * input_channels_block_max * sizeof(float), 64);
	
	void* memory_block_input = NULL;
	void* memory_block_kernel = NULL;

	memory_block_input = allocate_memory(packed_input_size);
	memory_block_kernel = allocate_memory(packed_kernel_size);

	if (memory_block_input == NULL || memory_block_kernel == NULL)
	{
		status = nnp_status_out_of_memory;
		goto cleanup;
	}

	float* packed_input = (float*)memory_block_input;
	float* packed_kernel = (float*)memory_block_kernel;

	/* Do the computation */
	compute_fully_connected_output(
		simd_width,
		batch_size, batch_block_max, batch_subblock_max,
		input_channels, input_channels_block_max,
		output_channels, output_channels_block_max, output_channels_subblock_max,
		input, kernel, output,
		packed_input, packed_kernel, profile);

cleanup:
	release_memory(memory_block_input, packed_input_size);
	release_memory(memory_block_kernel, packed_kernel_size);
	NNP_TOTAL_END(profile)
	return status;
}