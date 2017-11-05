#include <nnpack.h>
#include <macros.h>
#include <utils.h>
#include <system.h>

#include <hwinfo.h>
#include <validation.h>

struct NNP_CACHE_ALIGN input_transform_context
{
	size_t tuple_elements;
	size_t input_elements;
	size_t batch_block_size;
	size_t input_channels;
	size_t input_stride;
	uint32_t row_offset;
	uint32_t column_offset;
	uint32_t row_count;
	uint32_t column_count;
	const float* input;
	float* input_transform;
	nnp_transform_2d_with_offset transform_function;
};

static void compute_input_transform(
	const input_transform_context* context,
	const size_t batch_block_offset,
	const size_t input_channels_subblock_start,
	const size_t batch_block_offset_range,
	const size_t input_channels_subblock_size)
{
	const size_t tuple_elements                  = context->tuple_elements;
	const size_t input_elements                  = context->input_elements;
	const size_t batch_block_size                = context->batch_block_size;
	const size_t input_channels                  = context->input_channels;
	const size_t input_stride                    = context->input_stride;
	const uint32_t row_offset                    = context->row_offset;
	const uint32_t column_offset                 = context->column_offset;
	const uint32_t row_count                     = context->row_count;
	const uint32_t column_count                  = context->column_count;
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
			row_count,
			column_count,
			row_offset,
			column_offset);
	}
}

struct NNP_CACHE_ALIGN grad_output_transform_context
{
	size_t tuple_elements;
	size_t output_elements;
	size_t batch_block_size;
	size_t output_channels;
	size_t grad_output_stride;
	uint32_t row_count;
	uint32_t column_count;
	const float* grad_output;
	float* grad_output_transform;
	nnp_transform_2d_with_offset transform_function;
};

static void compute_grad_output_transform(
	const grad_output_transform_context* context,
	const size_t batch_block_offset,
	const size_t output_channels_subblock_start,
	const size_t batch_block_offset_range,
	const size_t output_channels_subblock_size)
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
			row_count,
			column_count,
			0u,
			0u);
	}
}

struct NNP_CACHE_ALIGN grad_kernel_transform_context
{
	size_t tuple_elements;
	size_t input_channels;
	size_t output_channels;
	size_t output_channels_block_max;
	nnp_size kernel_size;
	const float* grad_kernel_transform;
	float* grad_kernel;
	nnp_transform_2d_with_offset transform_function;
};

static void compute_grad_kernel_transform(
	const grad_kernel_transform_context* context,
	const size_t output_channel,
	const size_t input_channels_subblock_start,
	const size_t output_channel_range,
	const size_t input_channels_subblock_size)
{
	const size_t tuple_elements                  = context->tuple_elements;
	const size_t input_channels                  = context->input_channels;
	const size_t output_channels                 = context->output_channels;
	const size_t output_channels_block_max       = context->output_channels_block_max;
	const nnp_size kernel_size                   = context->kernel_size;
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
			uint32_t(kernel_size.height),
			uint32_t(kernel_size.width),
			0u,
			0u);
	}
}

struct NNP_CACHE_ALIGN matrix_multiplication_context
{
	size_t tuple_elements;
	size_t batch_size;
	size_t batch_block_size;
	size_t batch_block_update;
	size_t input_channels;
	size_t input_channels_block_start;
	size_t input_channels_block_size;
	size_t input_channels_subblock_max;
	size_t output_channels;
	size_t output_channels_subblock_max;
	const float* grad_output_transform;
	const float* input_transform;
	float* grad_kernel_transform;
	nnp_fast_tuple_gemm_function fast_gemm;
	nnp_full_tuple_gemm_function full_gemm;
};

static void compute_matrix_multiplication(
	const struct matrix_multiplication_context* context,
	const size_t output_channels_block_start,
	const size_t input_channels_subblock_start,
	size_t output_channels_block_size,
	const size_t input_channels_subblock_size)
{
	const size_t tuple_elements                  = context->tuple_elements;
	const size_t batch_size                      = context->batch_size;
	const size_t batch_block_size                = context->batch_block_size;
	const size_t batch_block_update              = context->batch_block_update;
	const size_t input_channels                  = context->input_channels;
	const size_t input_channels_block_start      = context->input_channels_block_start;
	const size_t input_channels_block_size       = context->input_channels_block_size;
	const size_t input_channels_subblock_max     = context->input_channels_subblock_max;
	const size_t output_channels                 = context->output_channels;
	const size_t output_channels_subblock_max    = context->output_channels_subblock_max;
	const float* grad_output_transform           = context->grad_output_transform + output_channels_block_start * batch_block_size * tuple_elements;
	const float* input_transform                 = context->input_transform + (input_channels_block_start + input_channels_subblock_start) * batch_block_size * tuple_elements;
	float* grad_kernel_transform                 = context->grad_kernel_transform + (output_channels_block_start * input_channels + (input_channels_block_start + input_channels_subblock_start) * output_channels_block_size) * tuple_elements;
	const nnp_fast_tuple_gemm_function fast_gemm = context->fast_gemm;
	const nnp_full_tuple_gemm_function full_gemm = context->full_gemm;

	if (input_channels_subblock_size == input_channels_subblock_max) 
	{
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
	const nnp_size tile_size,
	const nnp_size input_size,
	const nnp_padding input_padding,
	const nnp_size kernel_size,
	const nnp_size output_size,
	const float* input,
	const float* grad_output,
	float* grad_kernel,
	nnp_workspace_pointers* workspace_buffer,
	const nnp_transform_2d_with_offset input_transform_function,
	const nnp_transform_2d_with_offset grad_output_transform_function,
	const nnp_transform_2d_with_offset grad_kernel_transform_function,
	nnp_profile* profile)
{
	const size_t simd_width = nnp_hwinfo.simd_width;
	const size_t tuple_elements = simd_width << 1ull;
	const size_t tile_elements = tile_size.height * tile_size.width;
	const size_t tuple_count = tile_elements / tuple_elements;

	const nnp_size output_tile = { tile_size.width - kernel_size.width + 1ull, tile_size.height - kernel_size.height + 1ull };

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
		memory_block_grad_kernel = allocate_memory(grad_kernel_transform_size);
		memory_block_input = allocate_memory(input_transform_size);
		memory_block_grad_output = allocate_memory(grad_output_transform_size);

		if (memory_block_grad_kernel == NULL || memory_block_input == NULL || memory_block_grad_output == NULL)
			return nnp_status_out_of_memory;
	}
	else
	{
		if (workspace_buffer->kernel == NULL || workspace_buffer->input == NULL || workspace_buffer->output == NULL)
		{
			memory_block_grad_kernel = allocate_memory(grad_kernel_transform_size);
			memory_block_input = allocate_memory(input_transform_size);
			memory_block_grad_output = allocate_memory(grad_output_transform_size);

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
		const uint32_t row_offset = uint32_t(doz(input_padding.top, y));

		for (size_t x = 0ull; x < output_size.width; x += output_tile.width) 
		{
			const size_t input_x = min(doz(x, input_padding.left), input_size.width);
			const uint32_t column_offset = uint32_t(doz(input_padding.left, x));

			for (size_t batch_block_start = 0ull; batch_block_start < batch_size; batch_block_start += batch_block_max) 
			{
				const size_t batch_block_size = min(batch_size - batch_block_start, batch_block_max);

				/* Input transform */
				NNP_INPUT_TRANSFORM_START(profile)
				input_transform_context input_transform_context = 
				{
					tuple_elements,
					input_size.height * input_size.width,
					batch_block_size,
					input_channels,
					input_size.width,
					row_offset,
					column_offset,
					uint32_t(min(input_size.height - input_y, tile_size.height - row_offset)),
					uint32_t(min(input_size.width - input_x, tile_size.width - column_offset)),
					input + (batch_block_start * input_channels * input_size.height + input_y) * input_size.width + input_x,
					input_transform,
					input_transform_function
				};
				pthreadpool_compute_2d_tiled(
					(pthreadpool_function_2d_tiled_t)compute_input_transform,
					&input_transform_context,
					batch_block_size,
					input_channels,
					1u,
					input_channels_subblock_max);
				NNP_INPUT_TRANSFORM_END(profile)

				/* Grad output transform */
				NNP_OUTPUT_TRANSFORM_START(profile)
				grad_output_transform_context grad_output_transform_context = 
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
					(pthreadpool_function_2d_tiled_t)compute_grad_output_transform,
					&grad_output_transform_context,
					batch_block_size,
					output_channels,
					1u,
					output_channels_subblock_max);
				NNP_OUTPUT_TRANSFORM_END(profile)

				NNP_BLOCK_MULTIPLICATION_START(profile)
				for (size_t tuple_index = 0ull; tuple_index < tuple_count; tuple_index++) 
				{
					for (size_t input_channels_block_start = 0ull; input_channels_block_start < input_channels; input_channels_block_start += input_channels_block_max) 
					{
						const size_t input_channels_block_size = min(input_channels - input_channels_block_start, input_channels_block_max);

						matrix_multiplication_context matrix_multiplication_context = 
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
							output_channels,
							input_channels_block_size,
							output_channels_block_max,
							input_channels_subblock_max);
					}
				}
				NNP_BLOCK_MULTIPLICATION_END(profile)
			}
		}
	}

	/* Grad kernel transform */
	NNP_KERNEL_TRANSFORM_START(profile)
	grad_kernel_transform_context grad_kernel_transform_context = 
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
		(pthreadpool_function_2d_tiled_t)compute_grad_kernel_transform,
		&grad_kernel_transform_context,
		output_channels,
		input_channels,
		1ull,
		input_channels_subblock_max);
	NNP_KERNEL_TRANSFORM_END(profile)

	if (workspace_buffer == NULL)
	{
		release_memory(memory_block_grad_kernel, grad_kernel_transform_size);
		release_memory(memory_block_input, input_transform_size);
		release_memory(memory_block_grad_output, grad_output_transform_size);
	}
	else
	{
		if (memory_block_grad_kernel != workspace_buffer->kernel || memory_block_input != workspace_buffer->input || memory_block_grad_output != workspace_buffer->output)
		{
			release_memory(memory_block_grad_kernel, grad_kernel_transform_size);
			release_memory(memory_block_input, input_transform_size);
			release_memory(memory_block_grad_output, grad_output_transform_size);
		}
	}

	return nnp_status_success;
}

nnp_status nnp_convolution_kernel_gradient(
	nnp_convolution_algorithm algorithm,
	const size_t batch_size,
	const size_t input_channels,
	const size_t output_channels,
	const nnp_size input_size,
	const nnp_padding input_padding,
	const nnp_size kernel_size,
	const float* input,
	const float* grad_output,
	float* grad_kernel,
	nnp_workspace_pointers* workspace_buffer,
	const nnp_activation activation,
	const void* activation_parameters,
	nnp_profile* profile)
{
	NNP_TOTAL_START(profile)

	const nnp_size output_size = 
	{
		input_padding.left + input_size.width + input_padding.right - kernel_size.width + 1ull,
		input_padding.top + input_size.height + input_padding.bottom - kernel_size.height + 1ull
	};

	/* Basic validation of parameters. This check detects invalid, but not unsupported parameters. */
	nnp_status status = validate_convolution_arguments(batch_size, input_channels, output_channels,	input_size, input_padding, kernel_size, nnp_size { 1ull, 1ull }, activation, activation_parameters);
	if (status != nnp_status_success)
		goto cleanup;
	
	if (activation != nnp_activation_identity) 
	{
		status = nnp_status_unsupported_activation;
		goto cleanup;
	}

	if (activation_parameters != NULL) 
	{
		status = nnp_status_unsupported_activation_parameters;
		goto cleanup;
	}

	/* If requested, choose optimal convolution algorithm */
	if (algorithm == nnp_convolution_algorithm_auto) 
	{
		if (max(kernel_size.width, kernel_size.height) > 8ull) 
			algorithm = nnp_convolution_algorithm_ft16x16;
		else 
		{
			const size_t tile_count_8x8 =	divide_round_up(output_size.height, 8ull - kernel_size.height + 1ull) *
											divide_round_up(output_size.width, 8ull - kernel_size.width + 1ull);
			const size_t tile_count_16x16 =	divide_round_up(output_size.height, 16ull - kernel_size.height + 1ull) *
											divide_round_up(output_size.width, 16ull - kernel_size.width + 1ull);

			if (tile_count_8x8 <= 4 * tile_count_16x16) 
				/* 8x8 tiles are more efficient */
				algorithm = nnp_convolution_algorithm_ft8x8;
			else 
				algorithm = nnp_convolution_algorithm_ft16x16;
		}
	}

	/* Choose tiling parameters and transform functions depending on convolution algorithm */
	switch (algorithm) 
	{
		case nnp_convolution_algorithm_ft8x8:
			status = compute_fast_convolution_kernel_gradient(batch_size, input_channels, output_channels, nnp_size{ 8ull, 8ull }, input_size, input_padding, kernel_size, output_size, input, grad_output, grad_kernel, workspace_buffer, nnp_hwinfo.transforms.fft8x8_with_offset_and_stream, nnp_hwinfo.transforms.fft8x8_with_offset_and_stream, nnp_hwinfo.transforms.ifft8x8_with_offset, profile);
			break;

		case nnp_convolution_algorithm_ft16x16:
			status = compute_fast_convolution_kernel_gradient(batch_size, input_channels, output_channels, nnp_size{ 16ull, 16ull }, input_size, input_padding, kernel_size, output_size, input, grad_output, grad_kernel, workspace_buffer, nnp_hwinfo.transforms.fft16x16_with_offset_and_stream, nnp_hwinfo.transforms.fft16x16_with_offset_and_stream, nnp_hwinfo.transforms.ifft16x16_with_offset, profile);
			break;

		case nnp_convolution_algorithm_wt8x8:
			/*
			 * Winograd transform is not supported for this operation:
			 * it needs F(5x5, 4x4) transform and presently we implement only F(3x3, 6x6)
			 */
			status = nnp_status_unsupported_algorithm;
			break;
			
		default:
			status = nnp_status_invalid_algorithm;
			break;
	}

cleanup:
	NNP_TOTAL_END(profile)
	return status;
}
