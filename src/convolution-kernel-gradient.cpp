#ifdef __cplusplus
#include <cstddef>
#include <cstdint>
#include <cstdbool>
#else
#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>
#endif
#include <malloc.h>


#include <nnpack.h>
#include <utils.h>
#include <hwinfo.h>
#include <validation.h>


struct __declspec(align(64)) input_transform_context
{
	const size_t tuple_elements;
	const size_t input_elements;
	const size_t batch_block_size;
	const size_t input_channels;
	const size_t input_stride;
	const uint32_t row_offset;
	const uint32_t column_offset;
	const uint32_t row_count;
	const uint32_t column_count;
	const float* input;
	float* input_transform;
	const nnp_transform_2d_with_offset transform_function;
};

static void compute_input_transform(
	const struct input_transform_context* context,
	const size_t batch_block_offset,       const size_t input_channels_subblock_start,
	const size_t /* batch_block_offset_range */, const size_t input_channels_subblock_size)
{
	const size_t tuple_elements                  = context->tuple_elements;
	const size_t input_elements                  = context->input_elements;
	const size_t batch_block_size                = context->batch_block_size;
	const size_t input_channels                  = context->input_channels;
	const size_t input_stride                    = context->input_stride;
	const uint32_t row_count                     = context->row_count;
	const uint32_t column_count                  = context->column_count;
	const uint32_t row_offset                    = context->row_offset;
	const uint32_t column_offset                 = context->column_offset;
	const float* input                           = context->input;
	float* input_transform                       = context->input_transform;
	const nnp_transform_2d_with_offset transform = context->transform_function;

	for (size_t input_channels_subblock_offset = 0ull; input_channels_subblock_offset < input_channels_subblock_size; input_channels_subblock_offset++) 
	{
		const size_t input_channel = input_channels_subblock_start + input_channels_subblock_offset;
		transform(
			input +	(batch_block_offset * input_channels + input_channel) * input_elements,
			input_transform + (input_channels_subblock_start * batch_block_size + batch_block_offset * input_channels_subblock_size + input_channels_subblock_offset) * tuple_elements,
			input_stride, batch_block_size * input_channels * tuple_elements * sizeof(float),
			row_count, column_count, row_offset, column_offset);
	}
}

struct __declspec(align(64)) grad_output_transform_context
{
	const size_t tuple_elements;
	const size_t output_elements;
	const size_t batch_block_size;
	const size_t output_channels;
	const size_t grad_output_stride;
	const uint32_t row_count;
	const uint32_t column_count;
	const float* grad_output;
	float* grad_output_transform;
	const nnp_transform_2d_with_offset transform_function;
};

static void compute_grad_output_transform(
	const struct grad_output_transform_context* context,
	const size_t batch_block_offset,      const size_t output_channels_subblock_start,
	const size_t /* batch_block_offset_range */, const size_t output_channels_subblock_size)
{
	const size_t tuple_elements                  = context->tuple_elements;
	const size_t output_elements                 = context->output_elements;
	const size_t batch_block_size                = context->batch_block_size;
	const size_t output_channels                 = context->output_channels;
	const size_t grad_output_stride              = context->grad_output_stride;
	const uint32_t row_count                     = context->row_count;
	const uint32_t column_count                  = context->column_count;
	const float* grad_output                     = context->grad_output;
	float* grad_output_transform                 = context->grad_output_transform;
	const nnp_transform_2d_with_offset transform = context->transform_function;

	for (size_t output_channels_subblock_offset = 0ull; output_channels_subblock_offset < output_channels_subblock_size; output_channels_subblock_offset++) 
	{
		const size_t output_channel = output_channels_subblock_start + output_channels_subblock_offset;
		transform(
			grad_output + (batch_block_offset * output_channels + output_channel) * output_elements,
			grad_output_transform +	(output_channels_subblock_start * batch_block_size + batch_block_offset * output_channels_subblock_size + output_channels_subblock_offset) * tuple_elements,
			grad_output_stride,
			batch_block_size * output_channels * tuple_elements * sizeof(float),
			row_count, column_count, 0u, 0u);
	}
}

struct __declspec(align(64)) grad_kernel_transform_context
{
	const size_t tuple_elements;
	const size_t input_channels;
	const size_t output_channels;
	const size_t output_channels_block_max;
	const struct nnp_size kernel_size;
	const float* grad_kernel_transform;
	float* grad_kernel;
	const nnp_transform_2d_with_offset transform_function;
};

static void compute_grad_kernel_transform(
	const struct grad_kernel_transform_context* context,
	const size_t output_channel,       const size_t input_channels_subblock_start,
	const size_t /* output_channel_range */, const size_t input_channels_subblock_size)
{
	const size_t tuple_elements                  = context->tuple_elements;
	const size_t input_channels                  = context->input_channels;
	const size_t output_channels                 = context->output_channels;
	const size_t output_channels_block_max       = context->output_channels_block_max;
	const struct nnp_size kernel_size            = context->kernel_size;
	const float* grad_kernel_transform           = context->grad_kernel_transform;
	float* grad_kernel                           = context->grad_kernel;
	const nnp_transform_2d_with_offset transform = context->transform_function;

	const size_t output_channels_block_start  = round_down(output_channel, output_channels_block_max);
	const size_t output_channels_block_size   = min(output_channels - output_channels_block_start, output_channels_block_max);
	const size_t output_channels_block_offset = output_channel - output_channels_block_start;
	const size_t kernel_elements = kernel_size.height * kernel_size.width;

	for (size_t input_channels_subblock_offset = 0ull; input_channels_subblock_offset < input_channels_subblock_size; input_channels_subblock_offset++) 
	{
		const size_t input_channel = input_channels_subblock_start + input_channels_subblock_offset;
		transform(
			grad_kernel_transform +	(output_channels_block_start * input_channels + input_channels_subblock_start * output_channels_block_size + output_channels_block_offset * input_channels_subblock_size + input_channels_subblock_offset) * tuple_elements,
			grad_kernel + (output_channel * input_channels + input_channel) * kernel_elements,
			output_channels * input_channels * tuple_elements * sizeof(float),
			kernel_size.width,
			uint32_t(kernel_size.height), uint32_t(kernel_size.width), 0u, 0u);
	}
}

struct __declspec(align(64)) matrix_multiplication_context
{
	const size_t tuple_elements;
	const size_t batch_size;
	const size_t batch_block_size;
	const size_t batch_block_update;
	const size_t input_channels;
	const size_t input_channels_block_start;
	const size_t input_channels_block_size;
	const size_t input_channels_subblock_max;
	const size_t output_channels;
	const size_t output_channels_subblock_max;
	const float* grad_output_transform;
	const float* input_transform;
	float* grad_kernel_transform;

	nnp_fast_tuple_gemm_function fast_gemm;
	nnp_full_tuple_gemm_function full_gemm;
};

static void compute_matrix_multiplication(
	const struct matrix_multiplication_context* context,
	const size_t output_channels_block_start, const size_t input_channels_subblock_start,
	size_t output_channels_block_size,  const size_t input_channels_subblock_size)
{
	const size_t tuple_elements               = context->tuple_elements;
	const size_t batch_size                   = context->batch_size;
	const size_t batch_block_size             = context->batch_block_size;
	const size_t batch_block_update           = context->batch_block_update;
	const size_t input_channels               = context->input_channels;
	const size_t input_channels_block_start   = context->input_channels_block_start;
	const size_t input_channels_block_size    = context->input_channels_block_size;
	const size_t input_channels_subblock_max  = context->input_channels_subblock_max;
	const size_t output_channels              = context->output_channels;
	const size_t output_channels_subblock_max = context->output_channels_subblock_max;

	const float* grad_output_transform = context->grad_output_transform + output_channels_block_start * batch_block_size * tuple_elements;
	const float* input_transform       = context->input_transform + (input_channels_block_start + input_channels_subblock_start) * batch_block_size * tuple_elements;
	float* grad_kernel_transform       = context->grad_kernel_transform + (output_channels_block_start * input_channels + (input_channels_block_start + input_channels_subblock_start) * output_channels_block_size) * tuple_elements;

	if (input_channels_subblock_size == input_channels_subblock_max) 
	{
		const nnp_fast_tuple_gemm_function fast_gemm = context->fast_gemm;
		while (output_channels_block_size >= output_channels_subblock_max) 
		{
			output_channels_block_size -= output_channels_subblock_max;

			fast_gemm(
				batch_block_size, batch_block_update,
				input_transform,
				grad_output_transform,
				grad_kernel_transform,
				input_channels_subblock_size * tuple_elements);

			grad_output_transform += output_channels_subblock_max * batch_block_size * tuple_elements;
			grad_kernel_transform += output_channels_subblock_max * input_channels_subblock_size * tuple_elements;
		}
	}

	const nnp_full_tuple_gemm_function full_gemm = context->full_gemm;
	while (output_channels_block_size != 0ull) 
	{
		const size_t output_channels_subblock_size = min(output_channels_block_size, output_channels_subblock_max);
		output_channels_block_size -= output_channels_subblock_size;

		full_gemm(
			uint32_t(input_channels_subblock_size), uint32_t(output_channels_subblock_size),
			batch_block_size, batch_block_update,
			input_transform,
			grad_output_transform,
			grad_kernel_transform,
			input_channels_subblock_size * tuple_elements);

		grad_output_transform += output_channels_subblock_max * batch_block_size * tuple_elements;
		grad_kernel_transform += output_channels_subblock_max * input_channels_subblock_size * tuple_elements;
	}
}

static enum nnp_status compute_fast_convolution_kernel_gradient(
	const size_t batch_size,
	const size_t input_channels,
	const size_t output_channels,
	const struct nnp_size tile_size,
	const struct nnp_size input_size,
	const struct nnp_padding input_padding,
	const struct nnp_size kernel_size,
	const struct nnp_size output_size,
	const float* input,
	const float* grad_output,
	float* grad_kernel,
	struct nnp_workspace_pointers* workspace_buffer,
	const nnp_transform_2d_with_offset input_transform_function,
	const nnp_transform_2d_with_offset grad_output_transform_function,
	const nnp_transform_2d_with_offset grad_kernel_transform_function)
{
	const size_t simd_width = nnp_hwinfo.simd_width;
	const size_t tuple_elements = simd_width << 1ull;
	const size_t tile_elements = tile_size.height * tile_size.width;
	const size_t tuple_count = tile_elements / tuple_elements;

	const struct nnp_size output_tile = { tile_size.width - kernel_size.width + 1ull, tile_size.height - kernel_size.height + 1ull };

	/* Calculate cache blocking parameters */
	const size_t cache_elements_l1 = nnp_hwinfo.blocking.l1 / (tuple_elements * sizeof(float));
	const size_t cache_elements_l2 = nnp_hwinfo.blocking.l2 / (tuple_elements * sizeof(float));
	const size_t cache_elements_l3 = nnp_hwinfo.blocking.l3 / (tuple_elements * sizeof(float));

	const size_t input_channels_subblock_max = nnp_hwinfo.cxgemm.mr;
	const size_t output_channels_subblock_max = nnp_hwinfo.cxgemm.nr;

	const size_t batch_block_max = round_down(cache_elements_l1 / (input_channels_subblock_max + output_channels_subblock_max), 2ull);
	const size_t input_channels_block_max =	round_down(cache_elements_l3 / batch_block_max, input_channels_subblock_max);
	const size_t output_channels_block_max = round_down(cache_elements_l2 / batch_block_max, output_channels_subblock_max);

	/* Calculate memory footprint and allocate memory */
	const size_t input_transform_size = min(batch_size, batch_block_max) * input_channels * tile_elements * sizeof(float);
	const size_t grad_output_transform_size = min(batch_size, batch_block_max) * output_channels * tile_elements * sizeof(float);
	const size_t grad_kernel_transform_size = output_channels * input_channels * tile_elements * sizeof(float);
	
	void* memory_block_input = NULL;
	void* memory_block_grad_output = NULL;
	void* memory_block_grad_kernel = NULL;

	if (workspace_buffer == NULL)
	{
		memory_block_grad_kernel = _aligned_malloc(grad_kernel_transform_size, 64ull);
		memory_block_input = _aligned_malloc(input_transform_size, 64ull);
		memory_block_grad_output = _aligned_malloc(grad_output_transform_size, 64ull);

		if (memory_block_grad_kernel == NULL || memory_block_input == NULL || memory_block_grad_output == NULL)
			return nnp_status_out_of_memory;
	}
	else
	{
		if (workspace_buffer->kernel == NULL || workspace_buffer->input == NULL || workspace_buffer->output == NULL)
		{
			memory_block_grad_kernel = _aligned_malloc(grad_kernel_transform_size, 64ull);
			memory_block_input = _aligned_malloc(input_transform_size, 64ull);
			memory_block_grad_output = _aligned_malloc(grad_output_transform_size, 64ull);

			if (memory_block_grad_kernel == NULL || memory_block_input == NULL || memory_block_grad_output == NULL)
				return nnp_status_out_of_memory;

			*workspace_buffer = nnp_workspace_pointers{ memory_block_grad_kernel, memory_block_input, memory_block_grad_output };
		}
		else
		{
			memory_block_grad_kernel = workspace_buffer->kernel;
			memory_block_input = workspace_buffer->input;
			memory_block_grad_output = workspace_buffer->output;
		}
	}

	float* grad_kernel_transform = static_cast<float*>(memory_block_grad_kernel);
	float* input_transform = static_cast<float*>(memory_block_input);
	float* grad_output_transform = static_cast<float*>(memory_block_grad_output);
	
	for (size_t y = 0ull; y < output_size.height; y += output_tile.height) 
	{
		const size_t input_y = min(doz(y, input_padding.top), input_size.height);
		for (size_t x = 0ull; x < output_size.width; x += output_tile.width) 
		{
			const size_t input_x = min(doz(x, input_padding.left), input_size.width);

			for (size_t batch_block_start = 0ull; batch_block_start < batch_size; batch_block_start += batch_block_max) 
			{
				const size_t batch_block_size = min(batch_size - batch_block_start, batch_block_max);

				/* Input transform */
				struct input_transform_context input_transform_context = 
				{
					tuple_elements,
					input_size.height * input_size.width,
					batch_block_size,
					input_channels,
					input_size.width,
					uint32_t(doz(input_padding.top, y)),
					uint32_t(doz(input_padding.left, x)),
					uint32_t(min(input_size.height - input_y, tile_size.height - input_transform_context.row_offset)),
					uint32_t(min(input_size.width - input_x, tile_size.width - input_transform_context.column_offset)),
					input + (batch_block_start * input_channels * input_size.height + input_y) * input_size.width + input_x,
					input_transform,
					input_transform_function
				};

				pthreadpool_compute_2d_tiled(
					(pthreadpool_function_2d_tiled_t)compute_input_transform,
					&input_transform_context,
					batch_block_size, input_channels,
					1u, input_channels_subblock_max);
				
				struct grad_output_transform_context grad_output_transform_context = 
				{
					tuple_elements,
					output_size.height * output_size.width,
					batch_block_size,
					output_channels,
					output_size.width,
					uint32_t(min(output_tile.height, output_size.height - y)),
					uint32_t(min(output_tile.width, output_size.width - x)),
					grad_output + (batch_block_start * output_channels * output_size.height + y) * output_size.width + x,
					grad_output_transform,
					grad_output_transform_function
				};
				pthreadpool_compute_2d_tiled(
					(pthreadpool_function_2d_tiled_t) compute_grad_output_transform,
					&grad_output_transform_context,
					batch_block_size, output_channels,
					1u, output_channels_subblock_max);
				
				for (size_t tuple_index = 0ull; tuple_index < tuple_count; tuple_index++) 
				{
					for (size_t input_channels_block_start = 0ull; input_channels_block_start < input_channels; input_channels_block_start += input_channels_block_max) 
					{
						const size_t input_channels_block_size = min(input_channels - input_channels_block_start, input_channels_block_max);

						struct matrix_multiplication_context matrix_multiplication_context = 
						{
							tuple_elements,
							batch_size,
							batch_block_size,
							batch_block_start | x | y,
							input_channels,
							input_channels_block_start,
							input_channels_block_size,
							input_channels_subblock_max,
							output_channels,
							output_channels_subblock_max,
							grad_output_transform +	tuple_index * tuple_elements * batch_block_size * output_channels,
							input_transform + tuple_index * tuple_elements * batch_block_size * input_channels,
							grad_kernel_transform + tuple_index * tuple_elements * output_channels * input_channels,
							nnp_hwinfo.cxgemm.cX_conjb_transc_only_mr_x_nr,
							nnp_hwinfo.cxgemm.cX_conjb_transc_upto_mr_x_nr
						};

						if (tuple_index < NNP_COMPLEX_TUPLE_INDEX) 
						{
							matrix_multiplication_context.fast_gemm = nnp_hwinfo.cxgemm.s4cX_conjb_transc_only_mr_x_nr;
							matrix_multiplication_context.full_gemm = nnp_hwinfo.cxgemm.s4cX_conjb_transc_upto_mr_x_nr;
						} 
						
						pthreadpool_compute_2d_tiled(
							(pthreadpool_function_2d_tiled_t)compute_matrix_multiplication,
							&matrix_multiplication_context,
							output_channels,           input_channels_block_size,
							output_channels_block_max, input_channels_subblock_max);
					}
				}
			}
		}
	}
	/* Grad kernel transform */
	struct grad_kernel_transform_context grad_kernel_transform_context = 
	{
		tuple_elements,
		input_channels,
		output_channels,
		output_channels_block_max,
		kernel_size,
		grad_kernel_transform,
		grad_kernel,
		grad_kernel_transform_function
	};
	
	pthreadpool_compute_2d_tiled(
		(pthreadpool_function_2d_tiled_t) compute_grad_kernel_transform,
		&grad_kernel_transform_context,
		output_channels, input_channels,
		1ull, input_channels_subblock_max);
	

	if (workspace_buffer == NULL)
	{
		_aligned_free(memory_block_grad_kernel);
		_aligned_free(memory_block_input);
		_aligned_free(memory_block_grad_output);
	}
	else
	{
		if (memory_block_grad_kernel != workspace_buffer->kernel || memory_block_input != workspace_buffer->input || memory_block_grad_output != workspace_buffer->output)
		{
			_aligned_free(memory_block_grad_kernel);
			_aligned_free(memory_block_input);
			_aligned_free(memory_block_grad_output);
		}
	}

	return nnp_status_success;
}

enum nnp_status nnp_convolution_kernel_gradient(
	enum nnp_convolution_algorithm algorithm,
	const size_t batch_size,
	const size_t input_channels,
	const size_t output_channels,
	const struct nnp_size input_size,
	const struct nnp_padding input_padding,
	const struct nnp_size kernel_size,
	const float* input,
	const float* grad_output,
	float* grad_kernel,
	struct nnp_workspace_pointers* workspace_buffer)
{
	const struct nnp_size output_size = { input_padding.left + input_size.width + input_padding.right - kernel_size.width + 1ull, input_padding.top + input_size.height + input_padding.bottom - kernel_size.height + 1ull };

	/* If requested, choose optimal convolution algorithm */
	if (algorithm == nnp_convolution_algorithm_auto) 
	{
		if (max(kernel_size.width, kernel_size.height) > 8ull) 
			algorithm = nnp_convolution_algorithm_ft16x16;
		else 
		{
			const size_t tile_count_8x8 = divide_round_up(output_size.height, 8ull - kernel_size.height + 1ull) *	divide_round_up(output_size.width, 8ull - kernel_size.width + 1ull);
			const size_t tile_count_16x16 =	divide_round_up(output_size.height, 16ull - kernel_size.height + 1ull) * divide_round_up(output_size.width, 16ull - kernel_size.width + 1ull);
			if (tile_count_8x8 <= 4 * tile_count_16x16) 
				/* 8x8 tiles are more efficient */
				algorithm = nnp_convolution_algorithm_ft8x8;
			else 
				algorithm = nnp_convolution_algorithm_ft16x16;
		}
	}

	/* Choose tiling parameters and transform functions depending on convolution algorithm */
	struct nnp_size tile_size;
	nnp_transform_2d_with_offset input_transform_function;
	nnp_transform_2d_with_offset grad_output_transform_function;
	nnp_transform_2d_with_offset grad_kernel_transform_function;
	switch (algorithm) 
	{
		case nnp_convolution_algorithm_ft8x8:
			input_transform_function = nnp_hwinfo.transforms.fft8x8_with_offset_and_stream;
			grad_output_transform_function = nnp_hwinfo.transforms.fft8x8_with_offset_and_stream;
			grad_kernel_transform_function = nnp_hwinfo.transforms.ifft8x8_with_offset;
			tile_size = nnp_size { 8ull, 8ull };
			break;
		case nnp_convolution_algorithm_ft16x16:
			input_transform_function = nnp_hwinfo.transforms.fft16x16_with_offset_and_stream;
			grad_output_transform_function = nnp_hwinfo.transforms.fft16x16_with_offset_and_stream;
			grad_kernel_transform_function = nnp_hwinfo.transforms.ifft16x16_with_offset;
			tile_size = nnp_size { 16ull, 16ull };
			break;
		case nnp_convolution_algorithm_wt8x8:
			/*
			 * Winograd transform is not supported for this operation:
			 * it needs F(5x5, 4x4) transform and presently we implement only F(3x3, 6x6)
			 */
			return nnp_status_unsupported_algorithm;
			
		default:
			return nnp_status_invalid_algorithm;
	}

	return compute_fast_convolution_kernel_gradient(batch_size, input_channels, output_channels, tile_size, input_size, input_padding, kernel_size, output_size, input, grad_output, grad_kernel, workspace_buffer, input_transform_function, grad_output_transform_function, grad_kernel_transform_function);
}
