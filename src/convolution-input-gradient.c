#include <nnpack.h>
#include <macros.h>
#include <utils.h>
#include <system.h>
#include <hwinfo.h>
#include <validation.h>


struct NNP_CACHE_ALIGN kernel_transform_context 
{
	const nnp_transform_2d_with_offset transform_function;
	const float* kernel;
	float* kernel_transform;

	const size_t tuple_elements;
	const size_t input_channels;
	const size_t output_channels;
	const size_t output_channels_block_max;
	const struct nnp_size kernel_size;
};

static void compute_kernel_transform(
	const struct kernel_transform_context* context,
	const size_t output_channel,
	const size_t input_channels_subblock_start,
	const size_t output_channel_range,
	const size_t input_channels_subblock_size)
{
	const nnp_transform_2d_with_offset transform_function = context->transform_function;
	const float* kernel                                   = context->kernel;
	float* kernel_transform                               = context->kernel_transform;
	const size_t tuple_elements                           = context->tuple_elements;
	const size_t input_channels                           = context->input_channels;
	const size_t output_channels                          = context->output_channels;
	const size_t output_channels_block_max                = context->output_channels_block_max;
	const struct nnp_size kernel_size                     = context->kernel_size;
	
	const size_t output_channels_block_start  = round_down(output_channel, output_channels_block_max);
	const size_t output_channels_block_size   = min(output_channels - output_channels_block_start, output_channels_block_max);
	const size_t output_channels_block_offset = output_channel - output_channels_block_start;

	for (size_t input_channels_subblock_offset = 0; input_channels_subblock_offset < input_channels_subblock_size; input_channels_subblock_offset++) 
	{
		const size_t input_channel = input_channels_subblock_start + input_channels_subblock_offset;
		transform_function(
			kernel + ((input_channel + (output_channel * input_channels)) * kernel_size.width * kernel_size.height),
			kernel_transform + (output_channels_block_start * input_channels + input_channels_subblock_start * output_channels_block_size + output_channels_block_offset * input_channels_subblock_size + input_channels_subblock_offset) * tuple_elements,
			kernel_size.width,
			output_channels * input_channels * tuple_elements * sizeof(float),
			kernel_size.height,	kernel_size.width,
			0, 0);
	}
}

struct NNP_CACHE_ALIGN grad_output_transform_context
{
	const nnp_transform_2d_with_offset transform_function;
	const float* grad_output;
	float* grad_output_transform;

	const size_t tuple_elements;
	const size_t batch_size;
	const size_t output_channels;
	const size_t output_channels_block_max;
	const struct nnp_size output_size;
	const size_t row_offset;
	const size_t row_count;
	const size_t column_offset;
	const size_t column_count;
};

static void compute_grad_output_transform(
	const struct grad_output_transform_context* context,
	const size_t output_channel,
	const size_t batch_subblock_start,
	const size_t output_channel_range,
	const size_t batch_subblock_size)
{
	const nnp_transform_2d_with_offset transform_function = context->transform_function;
	const float* grad_output                              = context->grad_output;
	float* grad_output_transform                          = context->grad_output_transform;
	const size_t tuple_elements                           = context->tuple_elements;
	const size_t batch_size                               = context->batch_size;
	const size_t output_channels                          = context->output_channels;
	const size_t output_channels_block_max                = context->output_channels_block_max;
	const struct nnp_size output_size                     = context->output_size;
	const size_t row_offset                               = context->row_offset;
	const size_t row_count                                = context->row_count;
	const size_t column_offset                            = context->column_offset;
	const size_t column_count                             = context->column_count;

	const size_t output_channels_block_start  = round_down(output_channel, output_channels_block_max);
	const size_t output_channels_block_size   = min(output_channels - output_channels_block_start, output_channels_block_max);
	const size_t output_channels_block_offset = output_channel - output_channels_block_start;

	for (size_t batch_subblock_offset = 0; batch_subblock_offset < batch_subblock_size; batch_subblock_offset++)
	{
		const size_t sample = batch_subblock_start + batch_subblock_offset;
		transform_function(
			grad_output + (sample * output_channels * output_size.width * output_size.height) + (output_channel * output_size.width * output_size.height),
			grad_output_transform +	(output_channels_block_start * batch_size + batch_subblock_start * output_channels_block_size + output_channels_block_offset * batch_subblock_size + batch_subblock_offset) * tuple_elements,
			output_size.width,
			batch_size * output_channels * tuple_elements * sizeof(float),
			row_count, column_count,
			row_offset,	column_offset);
	}
}

struct NNP_CACHE_ALIGN grad_input_transform_context
{
	const nnp_transform_2d_with_offset transform_function;
	float* grad_input;
	const float* grad_input_transform;

	const size_t tuple_elements;
	const size_t input_channels;
	const size_t batch_size;
	const size_t batch_block_max;
	const struct nnp_size input_size;
	const size_t row_offset;
	const size_t row_count;
	const size_t column_offset;
	const size_t column_count;
};

static void compute_grad_input_transform(
	const struct grad_input_transform_context* context,
	const size_t sample,
	const size_t input_channels_subblock_start,
	const size_t sample_range,
	const size_t input_channels_subblock_size)
{
	const size_t tuple_elements                           = context->tuple_elements;
	const size_t input_channels                           = context->input_channels;
	const size_t batch_size                               = context->batch_size;
	const size_t batch_block_max                          = context->batch_block_max;
	const struct nnp_size input_size                      = context->input_size;
	const size_t row_offset                               = context->row_offset;
	const size_t row_count                                = context->row_count;
	const size_t column_offset                            = context->column_offset;
	const size_t column_count                             = context->column_count;
	
	float* grad_input                                     = context->grad_input;
	const float* grad_input_transform                     = context->grad_input_transform;
	const nnp_transform_2d_with_offset transform_function = context->transform_function;

	const size_t batch_block_start  = round_down(sample, batch_block_max);
	const size_t batch_block_size   = min(batch_size - batch_block_start, batch_block_max);
	const size_t batch_block_offset = sample - batch_block_start;

	for (size_t input_channels_subblock_offset = 0; input_channels_subblock_offset < input_channels_subblock_size; input_channels_subblock_offset++) 
	{
		const size_t input_channel = input_channels_subblock_start + input_channels_subblock_offset;
		transform_function(
			grad_input_transform + (batch_block_start * input_channels + input_channels_subblock_start * batch_block_size + batch_block_offset * input_channels_subblock_size + input_channels_subblock_offset) * tuple_elements,
			grad_input + ((sample * input_channels) + input_channel) * input_size.width * input_size.height,
			batch_size * input_channels * tuple_elements * sizeof(float),
			input_size.width,
			row_count, column_count,
			row_offset,	column_offset);
	}
}

struct NNP_CACHE_ALIGN matrix_multiplication_context
{
	const size_t tuple_elements;
	const size_t batch_size;
	const size_t input_channels;
	const size_t batch_block_start;
	const size_t batch_block_size;
	const size_t output_channels_block_start;
	const size_t output_channels_block_size;
	const size_t batch_subblock_max;
	const size_t input_channels_subblock_max;

	const float* grad_output_transform;
	const float* kernel_transform;
	float* grad_input_transform;

	nnp_fast_tuple_gemm_function fast_gemm;
	nnp_full_tuple_gemm_function full_gemm;
};

static void compute_matrix_multiplication(
	const struct matrix_multiplication_context* context,
	const size_t input_channels_block_start,
	const size_t batch_subblock_start,
	size_t input_channels_block_size,
	const size_t batch_subblock_size)
{
	const size_t tuple_elements                  = context->tuple_elements;
	const size_t batch_size                      = context->batch_size;
	const size_t input_channels                  = context->input_channels;
	const size_t batch_block_start               = context->batch_block_start;
	const size_t batch_block_size                = context->batch_block_size;
	const size_t output_channels_block_start     = context->output_channels_block_start;
	const size_t output_channels_block_size      = context->output_channels_block_size;
	const size_t batch_subblock_max              = context->batch_subblock_max;
	const size_t input_channels_subblock_max     = context->input_channels_subblock_max;
	
	const float* grad_output_transform           = context->grad_output_transform + (output_channels_block_start * batch_size + (batch_block_start + batch_subblock_start) * output_channels_block_size) * tuple_elements;
	const float* kernel_transform                = context->kernel_transform + (output_channels_block_start * input_channels + input_channels_block_start * output_channels_block_size) * tuple_elements;
	float* grad_input_transform                  = context->grad_input_transform + (batch_block_start * input_channels + input_channels_block_start * batch_block_size) * tuple_elements;
	
	if (batch_subblock_size == batch_subblock_max)
	{
		const nnp_fast_tuple_gemm_function fast_gemm = context->fast_gemm;
		while (input_channels_block_size >= input_channels_subblock_max)
		{
			input_channels_block_size -= input_channels_subblock_max;

			fast_gemm(
				output_channels_block_size,
				output_channels_block_start,
				grad_output_transform,
				kernel_transform,
				grad_input_transform + batch_subblock_start * input_channels_subblock_max * tuple_elements,
				input_channels_subblock_max * tuple_elements);

			kernel_transform += input_channels_subblock_max * output_channels_block_size * tuple_elements;
			grad_input_transform += input_channels_subblock_max * batch_block_size * tuple_elements;
		}
	}

	const nnp_full_tuple_gemm_function full_gemm = context->full_gemm;
	while (input_channels_block_size != 0) 
	{
		const size_t input_channels_subblock_size = min(input_channels_block_size, input_channels_subblock_max);
		input_channels_block_size -= input_channels_subblock_size;

		full_gemm(
			batch_subblock_size, 
			input_channels_subblock_size,
			output_channels_block_size, 
			output_channels_block_start,
			grad_output_transform,
			kernel_transform,
			grad_input_transform + batch_subblock_start * input_channels_subblock_size * tuple_elements,
			input_channels_subblock_size * tuple_elements);

		kernel_transform += input_channels_subblock_max * output_channels_block_size * tuple_elements;
		grad_input_transform += input_channels_subblock_max * batch_block_size * tuple_elements;
	}
}

static enum nnp_status compute_fast_convolution_input_gradient(
	const bool fourier_transform,
	const size_t batch_size,
	const size_t input_channels,
	const size_t output_channels,
	const struct nnp_size tile_size,
	const struct nnp_size input_size,
	const struct nnp_padding input_padding,
	const struct nnp_size kernel_size,
	const struct nnp_size output_size,
	const float* grad_output,
	const float* kernel,
	float* grad_input,
	struct nnp_workspace_pointers* workspace_buffer,
	const nnp_transform_2d_with_offset grad_output_transform_function,
	const nnp_transform_2d_with_offset kernel_transform_function,
	const nnp_transform_2d_with_offset grad_input_transform_function,
	struct nnp_profile* profile)
{
	const size_t simd_width = nnp_hwinfo.simd_width;
	const size_t tuple_elements = (fourier_transform ? simd_width * 2 : simd_width);
	const size_t tile_elements = tile_size.height * tile_size.width;
	const size_t tuple_count = tile_elements / tuple_elements;

	const struct nnp_size grad_input_tile_size = 
	{ 
		tile_size.width - kernel_size.width + 1, 
		tile_size.height - kernel_size.height + 1 
	};
	
	/* Calculate cache blocking parameters */
	const size_t cache_elements_l1 = nnp_hwinfo.blocking.l1 / (tuple_elements * sizeof(float));
	const size_t cache_elements_l2 = nnp_hwinfo.blocking.l2 / (tuple_elements * sizeof(float));
	const size_t cache_elements_l3 = nnp_hwinfo.blocking.l3 / (tuple_elements * sizeof(float));

	const size_t batch_subblock_max          = (fourier_transform ? nnp_hwinfo.cxgemm.mr : nnp_hwinfo.sxgemm.mr);
	const size_t input_channels_subblock_max = (fourier_transform ? nnp_hwinfo.cxgemm.nr : nnp_hwinfo.sxgemm.nr);

	const size_t output_channels_block_max = round_down(cache_elements_l1 / (batch_subblock_max + input_channels_subblock_max), 2);
	const size_t batch_block_max           = round_down(cache_elements_l3 / output_channels_block_max, batch_subblock_max);
	const size_t input_channels_block_max  = round_down(cache_elements_l2 / output_channels_block_max, input_channels_subblock_max);

	/* Calculate memory footprint and allocate memory */
	const size_t kernel_transform_size      = output_channels * input_channels * tile_elements * sizeof(float);
	const size_t grad_input_transform_size  = batch_size * input_channels * tile_elements * sizeof(float);
	const size_t grad_output_transform_size = batch_size * output_channels * tile_elements * sizeof(float);
	
	void* memory_block_kernel      = NULL;
	void* memory_block_grad_input  = NULL;
	void* memory_block_grad_output = NULL;

	if (workspace_buffer == NULL)
	{
		memory_block_kernel      = allocate_memory(kernel_transform_size);
		memory_block_grad_input  = allocate_memory(grad_input_transform_size);
		memory_block_grad_output = allocate_memory(grad_output_transform_size);

		if (memory_block_kernel == NULL || memory_block_grad_input == NULL || memory_block_grad_output == NULL)
			return nnp_status_out_of_memory;
	}
	else
	{
		if (workspace_buffer->kernel == NULL || workspace_buffer->input == NULL || workspace_buffer->output == NULL)
		{
			memory_block_kernel      = allocate_memory(kernel_transform_size);
			memory_block_grad_input  = allocate_memory(grad_input_transform_size);
			memory_block_grad_output = allocate_memory(grad_output_transform_size);

			if (memory_block_kernel == NULL || memory_block_grad_input == NULL || memory_block_grad_output == NULL)
				return nnp_status_out_of_memory;

			*workspace_buffer = (struct nnp_workspace_pointers)
			{ 
				memory_block_kernel, 
				memory_block_grad_input, 
				memory_block_grad_output 
			};
		}
		else
		{
			memory_block_kernel      = workspace_buffer->kernel;
			memory_block_grad_input  = workspace_buffer->input;
			memory_block_grad_output = workspace_buffer->output;
		}
	}

	float* kernel_transform      = (float*)memory_block_kernel;
	float* grad_input_transform  = (float*)memory_block_grad_input;
	float* grad_output_transform = (float*)memory_block_grad_output;

	NNP_KERNEL_TRANSFORM_START(profile)
	struct kernel_transform_context kernel_transform_context = 
	{
		kernel_transform_function,
		kernel,
		kernel_transform,
		tuple_elements,
		input_channels,
		output_channels,
		output_channels_block_max,
		kernel_size
	};
	pthreadpool_compute_2d_tiled(
		(pthreadpool_function_2d_tiled_t)compute_kernel_transform,
		&kernel_transform_context,
		output_channels, input_channels,
		1, input_channels_subblock_max);
	NNP_KERNEL_TRANSFORM_END(profile)

	for (size_t y = 0; y < input_size.height; y += grad_input_tile_size.height) 
	{
		const size_t grad_output_y = min(doz(y + input_padding.top, kernel_size.height - 1), output_size.height);
		const size_t row_offset = doz(kernel_size.height - 1, y + input_padding.top);

		for (size_t x = 0; x < input_size.width; x += grad_input_tile_size.width) 
		{
			const size_t grad_output_x = min(doz(x + input_padding.left, kernel_size.width - 1), output_size.width);
			const size_t column_offset = doz(kernel_size.width - 1, x + input_padding.left);

			NNP_OUTPUT_TRANSFORM_START(profile)
			struct grad_output_transform_context grad_output_transform_context = 
			{
				grad_output_transform_function,
				grad_output + grad_output_y * output_size.width + grad_output_x,
				grad_output_transform,
				tuple_elements,
				batch_size,
				output_channels,
				output_channels_block_max,
				output_size,
				row_offset,
				min(output_size.height - grad_output_y,	tile_size.height - row_offset),
				column_offset,
				min(output_size.width - grad_output_x, tile_size.width - column_offset)
			};
			pthreadpool_compute_2d_tiled(
				(pthreadpool_function_2d_tiled_t)compute_grad_output_transform,
				&grad_output_transform_context,
				output_channels, batch_size,
				1, batch_subblock_max);
			NNP_OUTPUT_TRANSFORM_END(profile)

			NNP_BLOCK_MULTIPLICATION_START(profile)
			for (size_t tuple_index = 0; tuple_index < tuple_count; tuple_index++)
			{
				for (size_t output_channels_block_start = 0; output_channels_block_start < output_channels; output_channels_block_start += output_channels_block_max) 
				{
					const size_t output_channels_block_size = min(output_channels - output_channels_block_start, output_channels_block_max);
					for (size_t batch_block_start = 0; batch_block_start < batch_size; batch_block_start += batch_block_max) 
					{
						const size_t batch_block_size = min(batch_size - batch_block_start, batch_block_max);
						
						struct matrix_multiplication_context matrix_multiplication_context = 
						{
							tuple_elements,
							batch_size,
							input_channels,
							batch_block_start,
							batch_block_size,
							output_channels_block_start,
							output_channels_block_size,
							batch_subblock_max,
							input_channels_subblock_max,
							grad_output_transform + tuple_index * tuple_elements * batch_size * output_channels,
							kernel_transform + tuple_index * tuple_elements * output_channels * input_channels,
							grad_input_transform + tuple_index * tuple_elements * batch_size * input_channels,
							nnp_hwinfo.sxgemm.only_mr_x_nr,
							nnp_hwinfo.sxgemm.upto_mr_x_nr
						};
						if (fourier_transform)
						{
							if (tuple_index < NNP_COMPLEX_TUPLE_INDEX)
							{
								matrix_multiplication_context.fast_gemm = nnp_hwinfo.cxgemm.s4cX_only_mr_x_nr;
								matrix_multiplication_context.full_gemm = nnp_hwinfo.cxgemm.s4cX_upto_mr_x_nr;
							}
							else
							{
								matrix_multiplication_context.fast_gemm = nnp_hwinfo.cxgemm.cX_only_mr_x_nr;
								matrix_multiplication_context.full_gemm = nnp_hwinfo.cxgemm.cX_upto_mr_x_nr;
							}
						}
						pthreadpool_compute_2d_tiled(
							(pthreadpool_function_2d_tiled_t)compute_matrix_multiplication,
							&matrix_multiplication_context,
							input_channels,	batch_block_size,
							input_channels_block_max, batch_subblock_max);
					}
				}
			}
			NNP_BLOCK_MULTIPLICATION_END(profile)

			NNP_INPUT_TRANSFORM_START(profile)
			struct grad_input_transform_context grad_input_transform_context = 
			{
				grad_input_transform_function,
				grad_input + y * input_size.width + x,
				grad_input_transform,
				tuple_elements,
				input_channels,
				batch_size,
				batch_block_max,
				input_size,
				fourier_transform ? kernel_size.height - 1 : 0,
				min(input_size.height - y, grad_input_tile_size.height),
				fourier_transform ? kernel_size.width - 1 : 0,
				min(input_size.width - x, grad_input_tile_size.width)
			};
			pthreadpool_compute_2d_tiled(
				(pthreadpool_function_2d_tiled_t)compute_grad_input_transform,
				&grad_input_transform_context,
				batch_size, input_channels,
				1, input_channels_subblock_max);
			NNP_INPUT_TRANSFORM_END(profile)
		}
	}
	
	if (workspace_buffer == NULL)
	{
		release_memory(memory_block_kernel, kernel_transform_size);
		release_memory(memory_block_grad_input, grad_input_transform_size);
		release_memory(memory_block_grad_output, grad_output_transform_size);
	}
	else
	{
		if (memory_block_kernel != workspace_buffer->kernel || memory_block_grad_input != workspace_buffer->input || memory_block_grad_output != workspace_buffer->output)
		{
			release_memory(memory_block_kernel, kernel_transform_size);
			release_memory(memory_block_grad_input, grad_input_transform_size);
			release_memory(memory_block_grad_output, grad_output_transform_size);
		}
	}

	return nnp_status_success;
}

enum nnp_status nnp_convolution_input_gradient(
	enum nnp_convolution_algorithm algorithm,
	const size_t batch_size,
	const size_t input_channels,
	const size_t output_channels,
	const struct nnp_size input_size,
	const struct nnp_padding input_padding,
	const struct nnp_size kernel_size,
	const float* grad_output,
	const float* kernel,
	float* grad_input,
	struct nnp_workspace_pointers* workspace_buffer,
	const enum nnp_activation activation,
	const void* activation_parameters,
	struct nnp_profile* profile)
{
	NNP_TOTAL_START(profile)

	const struct nnp_size output_size =
	{
		input_padding.left + input_size.width + input_padding.right - kernel_size.width + 1,
		input_padding.top + input_size.height + input_padding.bottom - kernel_size.height + 1
	};

	/* Basic validation of parameters. This check detects invalid, but not unsupported parameters. */
	enum nnp_status status = validate_convolution_arguments(batch_size, input_channels, output_channels, input_size, input_padding, kernel_size, (struct nnp_size) { 1, 1 }, activation, activation_parameters);
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

	if (algorithm == nnp_convolution_algorithm_auto) 
	{
		if (max(kernel_size.width, kernel_size.height) > 8) 
			algorithm = nnp_convolution_algorithm_ft16x16;
		else 
		{
			const size_t tile_count_8x8	  = divide_round_up(input_size.height, 8 - kernel_size.height + 1) * 
											divide_round_up(input_size.width, 8 - kernel_size.width + 1);

			const size_t tile_count_16x16 =	divide_round_up(input_size.height, 16 - kernel_size.height + 1) *
											divide_round_up(input_size.width, 16 - kernel_size.width + 1);

			if (tile_count_8x8 <= 4 * tile_count_16x16) 
			{
				/* 8x8 tiles are more efficient */
				if (kernel_size.height == 3 && kernel_size.width == 3)
					algorithm = nnp_convolution_algorithm_wt8x8;
				else 
					algorithm = nnp_convolution_algorithm_ft8x8;
			} 
			else 
				algorithm = nnp_convolution_algorithm_ft16x16;
		}
	}
	
	/* Choose tiling parameters and transform functions depending on convolution algorithm */
	switch (algorithm) 
	{
	case nnp_convolution_algorithm_wt8x8:
		if (kernel_size.height != 3 || kernel_size.width != 3)
		{
			status = nnp_status_unsupported_algorithm;
			goto cleanup;
		}
		status = compute_fast_convolution_input_gradient(false, batch_size, input_channels, output_channels, (struct nnp_size) { 8, 8 }, input_size, input_padding, kernel_size, output_size, grad_output, kernel, grad_input, workspace_buffer, nnp_hwinfo.transforms.iwt_f6x6_3x3_with_offset_and_stream, nnp_hwinfo.transforms.kwt_f6x6_3Rx3R, nnp_hwinfo.transforms.owt_f6x6_3x3, profile);
		break;

	case nnp_convolution_algorithm_ft8x8:
		status = compute_fast_convolution_input_gradient(true, batch_size, input_channels, output_channels, (struct nnp_size) { 8, 8 }, input_size, input_padding, kernel_size, output_size, grad_output, kernel, grad_input, workspace_buffer, nnp_hwinfo.transforms.fft8x8_with_offset_and_stream, nnp_hwinfo.transforms.fft8x8_with_offset_and_stream, nnp_hwinfo.transforms.ifft8x8_with_offset, profile);
		break;

	case nnp_convolution_algorithm_ft16x16:
		status = compute_fast_convolution_input_gradient(true, batch_size, input_channels, output_channels, (struct nnp_size) { 16, 16 }, input_size, input_padding, kernel_size, output_size, grad_output, kernel, grad_input, workspace_buffer, nnp_hwinfo.transforms.fft16x16_with_offset_and_stream, nnp_hwinfo.transforms.fft16x16_with_offset_and_stream, nnp_hwinfo.transforms.ifft16x16_with_offset, profile);
	    break;

	default:
		status = nnp_status_invalid_algorithm;
		break;
	}

cleanup:
	NNP_TOTAL_END(profile)
	return status;
}

