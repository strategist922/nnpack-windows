#ifdef __cplusplus
#include <cstddef>
#include <cstdint>
#include <cstdbool>
#else
#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>
#endif

#include <nnpack.h>
#include <utils.h>
#include <hwinfo.h>
#include <validation.h>

struct __declspec(align(64)) kernel_transform_context
{
	const nnp_transform_2d_with_offset transform_function;
	const float* kernel;
	float* kernel_transform;
	const size_t tuple_elements;
	const size_t output_channels;
	const size_t input_channels;
	const size_t input_channels_block_max;
	const struct nnp_size kernel_size;
};

static void compute_kernel_transform(
	const struct kernel_transform_context* context,
	const size_t input_channel, 
	const size_t output_channels_subblock_start,
	const size_t input_channel_range, 
	const size_t output_channels_subblock_size)
{
	const nnp_transform_2d_with_offset transform_function	= context->transform_function;
	const float* kernel										= context->kernel;
	float* kernel_transform									= context->kernel_transform;
	const size_t tuple_elements								= context->tuple_elements;
	const size_t output_channels							= context->output_channels;
	const size_t input_channels								= context->input_channels;
	const size_t input_channels_block_max					= context->input_channels_block_max;
	const struct nnp_size kernel_size						= context->kernel_size;

	const size_t input_channels_block_start		= round_down(input_channel, input_channels_block_max);
	const size_t input_channels_block_size		= min(input_channels - input_channels_block_start, input_channels_block_max);
	const size_t input_channels_block_offset	= input_channel - input_channels_block_start;

	for (size_t output_channels_subblock_offset = 0ull; output_channels_subblock_offset < output_channels_subblock_size; output_channels_subblock_offset++) 
	{
		const size_t output_channel = output_channels_subblock_start + output_channels_subblock_offset;
		transform_function(
			kernel + (input_channel + (output_channel * input_channels)) * kernel_size.width * kernel_size.height,
			kernel_transform + (input_channels_block_start * output_channels + output_channels_subblock_start * input_channels_block_size + input_channels_block_offset * output_channels_subblock_size + output_channels_subblock_offset) * tuple_elements,
			kernel_size.width,
			output_channels * input_channels * tuple_elements * sizeof(float),
			uint32_t(kernel_size.height),
			uint32_t(kernel_size.width),
			0u,
			0u);
	}
}

struct __declspec(align(64)) input_transform_context
{
	const nnp_transform_2d_with_offset transform_function;
	const float* input;
	float* input_transform;

	const size_t tuple_elements;
	const size_t batch_size;
	const size_t input_channels;
	const size_t input_channels_block_max;
	const struct nnp_size input_size;
	const size_t row_offset;
	const size_t row_count;
	const size_t column_offset;
	const size_t column_count;
};

static void compute_input_transform(
	const struct input_transform_context* context,
	const size_t input_channel, 
	const size_t batch_subblock_start,
	const size_t input_channel_range, 
	const size_t batch_subblock_size)
{
	const nnp_transform_2d_with_offset transform_function	= context->transform_function;
	const float* input										= context->input;
	float* input_transform									= context->input_transform;
	const size_t tuple_elements								= context->tuple_elements;
	const size_t batch_size									= context->batch_size;
	const size_t input_channels								= context->input_channels;
	const size_t input_channels_block_max					= context->input_channels_block_max;
	const struct nnp_size input_size						= context->input_size;
	const size_t row_offset									= context->row_offset;
	const size_t row_count									= context->row_count;
	const size_t column_offset								= context->column_offset;
	const size_t column_count								= context->column_count;
	
	const size_t input_channels_block_start		= round_down(input_channel, input_channels_block_max);
	const size_t input_channels_block_size		= min(input_channels - input_channels_block_start, input_channels_block_max);
	const size_t input_channels_block_offset	= input_channel - input_channels_block_start;

	for (size_t batch_subblock_offset = 0ull; batch_subblock_offset < batch_subblock_size; batch_subblock_offset++) 
	{
		const size_t sample = batch_subblock_start + batch_subblock_offset;
		transform_function(
			input + (sample * input_channels * input_size.width * input_size.height) + (input_channel * input_size.width * input_size.height),
			input_transform + (input_channels_block_start * batch_size + batch_subblock_start * input_channels_block_size + input_channels_block_offset * batch_subblock_size + batch_subblock_offset) * tuple_elements,
			input_size.width,
			batch_size * input_channels * tuple_elements * sizeof(float),
			uint32_t(row_count),
			uint32_t(column_count),
			uint32_t(row_offset),
			uint32_t(column_offset));
	}
}

struct __declspec(align(64)) output_transform_context
{
	const nnp_transform_2d_with_bias transform_function;
	float* output;
	const float* output_transform;
	const float* bias;
	const size_t tuple_elements;
	const size_t output_channels;
	const size_t batch_size;
	const size_t batch_block_max;
	const struct nnp_size output_size;
	const size_t row_offset;
	const size_t row_count;
	const size_t column_offset;
	const size_t column_count;
};

static void compute_output_transform(
	const struct output_transform_context* context,
	const size_t sample, 
	const size_t output_channels_subblock_start,
	const size_t sample_range, 
	const size_t output_channels_subblock_size)
{
	const nnp_transform_2d_with_bias transform_function	= context->transform_function;
	float* output										= context->output;
	const float* output_transform						= context->output_transform;
	const float* bias									= context->bias;
	const size_t tuple_elements							= context->tuple_elements;
	const size_t batch_size								= context->batch_size;
	const size_t output_channels						= context->output_channels;
	const size_t batch_block_max						= context->batch_block_max;
	const struct nnp_size output_size					= context->output_size;
	const size_t row_offset								= context->row_offset;
	const size_t row_count								= context->row_count;
	const size_t column_offset							= context->column_offset;
	const size_t column_count							= context->column_count;

	const size_t batch_block_start	= round_down(sample, batch_block_max);
	const size_t batch_block_size	= min(batch_size - batch_block_start, batch_block_max);
	const size_t batch_block_offset	= sample - batch_block_start;

	for (size_t output_channels_subblock_offset = 0ull; output_channels_subblock_offset < output_channels_subblock_size; output_channels_subblock_offset++) 
	{
		const size_t output_channel = output_channels_subblock_start + output_channels_subblock_offset;
		transform_function(
			output_transform + (batch_block_start * output_channels + output_channels_subblock_start * batch_block_size + batch_block_offset * output_channels_subblock_size + output_channels_subblock_offset) * tuple_elements,
			output + (sample * output_channels * output_size.width * output_size.height) + (output_channel * output_size.width * output_size.height),
			bias + output_channel,
			batch_size * output_channels * tuple_elements * sizeof(float),
			output_size.width,
			uint32_t(row_count),
			uint32_t(column_count));
	}
}

struct __declspec(align(64)) matrix_multiplication_context
{
	const size_t tuple_elements;
	const size_t batch_block_size;
	const size_t input_channels_block_start;
	const size_t input_channels_block_size;
	const size_t batch_subblock_max;
	const size_t output_channels_subblock_max;
	const float* input_transform;
	const float* kernel_transform;
	float* output_transform;
	nnp_fast_tuple_gemm_function fast_gemm;
	nnp_full_tuple_gemm_function full_gemm;
};

static void compute_matrix_multiplication(
	const struct matrix_multiplication_context* context,
	const size_t output_channels_block_start, 
	const size_t batch_subblock_start,
	size_t output_channels_block_size, 
	const size_t batch_subblock_size)
{
	const size_t tuple_elements					= context->tuple_elements;
	const size_t batch_block_size				= context->batch_block_size;
	const size_t input_channels_block_start		= context->input_channels_block_start;
	const size_t input_channels_block_size		= context->input_channels_block_size;
	const size_t batch_subblock_max				= context->batch_subblock_max;
	const size_t output_channels_subblock_max	= context->output_channels_subblock_max;
	const float* input_transform				= context->input_transform + (batch_subblock_start * input_channels_block_size * tuple_elements);
	const float* kernel_transform				= context->kernel_transform + (output_channels_block_start * input_channels_block_size * tuple_elements);
	float* output_transform						= context->output_transform + (output_channels_block_start * batch_block_size * tuple_elements);
	nnp_fast_sgemm_function fast_gemm			= context->fast_gemm;
	nnp_full_sgemm_function full_gemm			= context->full_gemm;

	if (batch_subblock_size == batch_subblock_max) 
	{
		while (output_channels_block_size >= output_channels_subblock_max) 
		{
			output_channels_block_size -= output_channels_subblock_max;

			fast_gemm(
				input_channels_block_size, 
				input_channels_block_start,
				input_transform,
				kernel_transform,
				output_transform + (batch_subblock_start * output_channels_subblock_max * tuple_elements),
				output_channels_subblock_max * tuple_elements);

			kernel_transform += input_channels_block_size * output_channels_subblock_max * tuple_elements;
			output_transform += batch_block_size          * output_channels_subblock_max * tuple_elements;
		}
	}
	
	while (output_channels_block_size != 0ull) 
	{
		const size_t output_channels_subblock_size = min(output_channels_block_size, output_channels_subblock_max);
		output_channels_block_size -= output_channels_subblock_size;

		full_gemm(
			uint32_t(batch_subblock_size),
			uint32_t(output_channels_subblock_size),
			input_channels_block_size,
			input_channels_block_start,
			input_transform,
			kernel_transform,
			output_transform + (batch_subblock_start * output_channels_subblock_size * tuple_elements),
			output_channels_subblock_size * tuple_elements);

		kernel_transform += input_channels_block_size * output_channels_subblock_max * tuple_elements;
		output_transform += batch_block_size          * output_channels_subblock_max * tuple_elements;
	}
}


static enum nnp_status compute_fast_convolution_output(
	const bool fourier_transform,
	const size_t batch_size,
	const size_t input_channels,
	const size_t output_channels,
	const struct nnp_size tile_size,
	const struct nnp_size input_size,
	const struct nnp_padding input_padding,
	const struct nnp_size kernel_size,
	const struct nnp_size output_size,
	const float* input,
	const float* kernel,
	const float* bias,
	float* output,
	struct nnp_workspace_pointers* workspace_buffer,
	const nnp_transform_2d_with_offset input_transform_function,
	const nnp_transform_2d_with_offset kernel_transform_function,
	const nnp_transform_2d_with_bias output_transform_function)
{
	const size_t simd_width = nnp_hwinfo.simd_width;
	const size_t tuple_elements = (fourier_transform ? simd_width << 1ull : simd_width);
	const size_t tile_elements = tile_size.height * tile_size.width;
	const size_t tuple_count = tile_elements / tuple_elements;

	const struct nnp_size output_tile_size = { tile_size.width - kernel_size.width + 1ull, tile_size.height - kernel_size.height + 1ull };

	/* Calculate cache blocking parameters */
	const size_t cache_elements_l1 = nnp_hwinfo.blocking.l1 / (tuple_elements * sizeof(float));
	const size_t cache_elements_l2 = nnp_hwinfo.blocking.l2 / (tuple_elements * sizeof(float));
	const size_t cache_elements_l3 = nnp_hwinfo.blocking.l3 / (tuple_elements * sizeof(float));

	const size_t batch_subblock_max = (fourier_transform ? nnp_hwinfo.cxgemm.mr : nnp_hwinfo.sxgemm.mr);
	const size_t output_channels_subblock_max = (fourier_transform ? nnp_hwinfo.cxgemm.nr : nnp_hwinfo.sxgemm.nr);

	const size_t input_channels_block_max =	round_down(cache_elements_l1 / (batch_subblock_max + output_channels_subblock_max), 2ull);
	const size_t batch_block_max = round_down(cache_elements_l3 / input_channels_block_max, batch_subblock_max);
	const size_t output_channels_block_max = round_down(cache_elements_l2 / input_channels_block_max, output_channels_subblock_max);

	/* Calculate memory footprint and allocate memory */
	const size_t kernel_transform_size = output_channels * input_channels * tile_elements * sizeof(float);
	const size_t input_transform_size = batch_size * input_channels * tile_elements * sizeof(float);
	const size_t output_transform_size = batch_size * output_channels * tile_elements * sizeof(float);
	
	void* memory_block_kernel = NULL;
	void* memory_block_input = NULL;
	void* memory_block_output = NULL;

	if (workspace_buffer == NULL)
	{
		memory_block_kernel = _aligned_malloc(kernel_transform_size, 64ull);
		memory_block_input = _aligned_malloc(input_transform_size, 64ull);
		memory_block_output = _aligned_malloc(output_transform_size, 64ull);

		if (memory_block_kernel == NULL || memory_block_input == NULL || memory_block_output == NULL)
			return nnp_status_out_of_memory;
	}
	else
	{
		if (workspace_buffer->kernel == NULL || workspace_buffer->input == NULL || workspace_buffer->output == NULL)
		{
			memory_block_kernel = _aligned_malloc(kernel_transform_size, 64ull);
			memory_block_input = _aligned_malloc(input_transform_size, 64ull);
			memory_block_output = _aligned_malloc(output_transform_size, 64ull);

			if (memory_block_kernel == NULL || memory_block_input == NULL || memory_block_output == NULL)
				return nnp_status_out_of_memory;

			*workspace_buffer = nnp_workspace_pointers{ memory_block_kernel, memory_block_input, memory_block_output };
		}
		else
		{
			memory_block_kernel = workspace_buffer->kernel;
			memory_block_input = workspace_buffer->input;
			memory_block_output = workspace_buffer->output;
		}
	}

	float* kernel_transform = static_cast<float*>(memory_block_kernel);
	float* input_transform = static_cast<float*>(memory_block_input);
	float* output_transform = static_cast<float*>(memory_block_output);
	
	struct kernel_transform_context kernel_transform_contex =
	{
		kernel_transform_function,
		kernel,
		kernel_transform,
		tuple_elements,
		output_channels,
		input_channels,
		input_channels_block_max,
		kernel_size
	};
	
	pthreadpool_compute_2d_tiled(
		(pthreadpool_function_2d_tiled_t)compute_kernel_transform,
		&kernel_transform_contex,
		input_channels,
		output_channels,
		1ull,
		output_channels_subblock_max);
	
	for (size_t y = 0ull; y < output_size.height; y += output_tile_size.height) 
	{
		const size_t input_y = min(doz(y, input_padding.top), input_size.height);
		const size_t row_offset = doz(input_padding.top, y);

		for (size_t x = 0ull; x < output_size.width; x += output_tile_size.width) 
		{
			const size_t input_x = min(doz(x, input_padding.left), input_size.width);
			const size_t column_offset = doz(input_padding.left, x);

			struct input_transform_context input_transform_ctx =
			{
				input_transform_function,
				input + input_y * input_size.width + input_x,
				input_transform,
				tuple_elements,
				batch_size,
				input_channels,
				input_channels_block_max,
				input_size,
				row_offset,
				min(input_size.height - input_y, tile_size.height - row_offset),
				column_offset,
				min(input_size.width - input_x,tile_size.width - column_offset)
			};
			pthreadpool_compute_2d_tiled(
				(pthreadpool_function_2d_tiled_t)compute_input_transform,
				&input_transform_ctx,
				input_channels,
				batch_size,
				1ull,
				batch_subblock_max);
			
			for (size_t tuple_index = 0ull; tuple_index < tuple_count; tuple_index++) 
			{
				for (size_t input_channels_block_start = 0ull; input_channels_block_start < input_channels; input_channels_block_start += input_channels_block_max) 
				{
					const size_t input_channels_block_size = min(input_channels - input_channels_block_start, input_channels_block_max);
					for (size_t batch_block_start = 0ull; batch_block_start < batch_size; batch_block_start += batch_block_max) 
					{
						const size_t batch_block_size = min(batch_size - batch_block_start, batch_block_max);

						struct matrix_multiplication_context matrix_multiplication_contex =
						{
							tuple_elements,
							batch_block_size,
							input_channels_block_start,
							input_channels_block_size,
							batch_subblock_max,
							output_channels_subblock_max,
							input_transform + tuple_index * tuple_elements * batch_size * input_channels + input_channels_block_start * batch_size * tuple_elements + batch_block_start * input_channels_block_size * tuple_elements,
							kernel_transform + tuple_index * tuple_elements * output_channels * input_channels + input_channels_block_start * output_channels * tuple_elements,
							output_transform + tuple_index * tuple_elements * batch_size * output_channels + batch_block_start * output_channels * tuple_elements,
							nnp_hwinfo.sxgemm.only_mr_x_nr,
							nnp_hwinfo.sxgemm.upto_mr_x_nr
						};

						if (fourier_transform) 
						{
							if (tuple_index < NNP_COMPLEX_TUPLE_INDEX) 
							{
								matrix_multiplication_contex.fast_gemm = nnp_hwinfo.cxgemm.s4cX_conjb_only_mr_x_nr;
								matrix_multiplication_contex.full_gemm = nnp_hwinfo.cxgemm.s4cX_conjb_upto_mr_x_nr;
							}
							else 
							{
								matrix_multiplication_contex.fast_gemm = nnp_hwinfo.cxgemm.cX_conjb_only_mr_x_nr;
								matrix_multiplication_contex.full_gemm = nnp_hwinfo.cxgemm.cX_conjb_upto_mr_x_nr;
							}
						}
						
						pthreadpool_compute_2d_tiled(
							(pthreadpool_function_2d_tiled_t)compute_matrix_multiplication,
							&matrix_multiplication_contex,
							output_channels,
							batch_block_size,
							output_channels_block_max,
							batch_subblock_max);
					}
				}
			}

			struct output_transform_context output_transform_contex =
			{
				output_transform_function,
				output + y * output_size.width + x,
				output_transform,
				bias,
				tuple_elements,
				output_channels,
				batch_size,
				batch_block_max,
				output_size,
				0ull,
				min(output_tile_size.height, output_size.height - y),
				0ull,
				min(output_tile_size.width, output_size.width - x)
			};
				
			pthreadpool_compute_2d_tiled(
				(pthreadpool_function_2d_tiled_t)compute_output_transform,
				&output_transform_contex,
				batch_size,
				output_channels,
				1ull,
				output_channels_subblock_max);
		}
	}
	
	if (workspace_buffer == NULL)
	{
		_aligned_free(memory_block_kernel);
		_aligned_free(memory_block_input);
		_aligned_free(memory_block_output);
	}
	else
	{
		if (memory_block_kernel != workspace_buffer->kernel || memory_block_input != workspace_buffer->input || memory_block_output != workspace_buffer->output)
		{
			_aligned_free(memory_block_kernel);
			_aligned_free(memory_block_input);
			_aligned_free(memory_block_output);
		}
	}
	
	return nnp_status_success;
}

enum nnp_status nnp_convolution_output(
	enum nnp_convolution_algorithm algorithm,
	const size_t batch_size,
	const size_t input_channels,
	const size_t output_channels,
	const struct nnp_size input_size,
	const struct nnp_padding input_padding,
	const struct nnp_size kernel_size,
	const float* input,
	const float* kernel,
	const float* bias,
	float* output,
	struct nnp_workspace_pointers* workspace_buffer,
	const enum nnp_activation activation,
	const void* activation_parameters
	)
{
	const struct nnp_size output_size = 
	{ 
		input_padding.left + input_size.width + input_padding.right - kernel_size.width + 1ull, 
		input_padding.top + input_size.height + input_padding.bottom - kernel_size.height + 1ull 
	};

	if (activation_parameters != NULL)
		return nnp_status_unsupported_activation_parameters;

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
			{
				/* 8x8 tiles are more efficient */
				if (kernel_size.height == 3ull && kernel_size.width == 3ull) 
					algorithm = nnp_convolution_algorithm_wt8x8;
				else 
					algorithm = nnp_convolution_algorithm_ft8x8;
			}
			else 
				algorithm = nnp_convolution_algorithm_ft16x16;
		}
	}

	/* Choose tiling parameters and transform functions depending on convolution algorithm */
	enum nnp_status status = nnp_status_success;
	nnp_transform_2d_with_bias output_transform_function;
	
	switch (algorithm) 
	{
	case nnp_convolution_algorithm_wt8x8:
		if (kernel_size.height > 8ull || kernel_size.width > 8ull)
			status = nnp_status_unsupported_algorithm;
		else
			if (activation == nnp_activation_relu)
				output_transform_function = nnp_hwinfo.transforms.owt_f6x6_3x3_with_bias_with_relu;
			else
				output_transform_function = nnp_hwinfo.transforms.owt_f6x6_3x3_with_bias;
			status = compute_fast_convolution_output(false, batch_size, input_channels, output_channels, nnp_size{ 8ull, 8ull }, input_size, input_padding, kernel_size, output_size, input, kernel, bias, output, workspace_buffer, nnp_hwinfo.transforms.iwt_f6x6_3x3_with_offset_and_stream, nnp_hwinfo.transforms.kwt_f6x6_3x3, output_transform_function);
		break;

	case nnp_convolution_algorithm_ft8x8:
		if (kernel_size.height > 8ull || kernel_size.width > 8ull)
			status = nnp_status_unsupported_algorithm;
		else
			if (activation == nnp_activation_relu)
				output_transform_function = nnp_hwinfo.transforms.ifft8x8_with_bias_with_relu;
			else
				output_transform_function = nnp_hwinfo.transforms.ifft8x8_with_bias;

			status = compute_fast_convolution_output(true, batch_size, input_channels, output_channels, nnp_size{ 8ull, 8ull }, input_size, input_padding, kernel_size, output_size, input, kernel, bias, output, workspace_buffer, nnp_hwinfo.transforms.fft8x8_with_offset_and_stream, nnp_hwinfo.transforms.fft8x8_with_offset_and_stream, output_transform_function);
		break;

	case nnp_convolution_algorithm_ft16x16:
		if (kernel_size.height > 16ull || kernel_size.width > 16ull)
			status = nnp_status_unsupported_algorithm;
		else
			if (activation == nnp_activation_relu)
				output_transform_function = nnp_hwinfo.transforms.ifft16x16_with_bias_with_relu;
			else
				output_transform_function = nnp_hwinfo.transforms.ifft16x16_with_bias;
			status = compute_fast_convolution_output(true, batch_size, input_channels, output_channels, nnp_size{ 16ull, 16ull }, input_size, input_padding, kernel_size, output_size, input, kernel, bias, output, workspace_buffer, nnp_hwinfo.transforms.fft16x16_with_offset_and_stream, nnp_hwinfo.transforms.fft16x16_with_offset_and_stream, output_transform_function);
		break;

	default:
		status = nnp_status_invalid_algorithm;
		break;
	}

	return status;
}