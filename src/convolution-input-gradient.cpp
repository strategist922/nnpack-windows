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
	const size_t input_channels;
	const size_t output_channels;
	const size_t output_channels_block_max;
	const struct nnp_size kernel_size;
};

static void compute_kernel_transform(
	const struct kernel_transform_context* context,
	size_t output_channel,      
	size_t input_channels_subblock_start,
	size_t /* output_channel_range */, 
	size_t input_channels_subblock_size)
{
	const nnp_transform_2d_with_offset transform_function	= context->transform_function;
	const float* kernel										= context->kernel;
	float* kernel_transform									= context->kernel_transform;
	const size_t tuple_elements								= context->tuple_elements;
	const size_t input_channels								= context->input_channels;
	const size_t output_channels							= context->output_channels;
	const size_t output_channels_block_max					= context->output_channels_block_max;
	const struct nnp_size kernel_size						= context->kernel_size;
	
	const size_t output_channels_block_start  = round_down(output_channel, output_channels_block_max);
	const size_t output_channels_block_size   = min(output_channels - output_channels_block_start, output_channels_block_max);
	const size_t output_channels_block_offset = output_channel - output_channels_block_start;

	for (size_t input_channels_subblock_offset = 0ull; input_channels_subblock_offset < input_channels_subblock_size; input_channels_subblock_offset++) 
	{
		const size_t input_channel = input_channels_subblock_start + input_channels_subblock_offset;
		transform_function(
			kernel + ((input_channel + (output_channel * input_channels)) * kernel_size.width * kernel_size.height),
			kernel_transform + (output_channels_block_start * input_channels + input_channels_subblock_start * output_channels_block_size + output_channels_block_offset * input_channels_subblock_size + input_channels_subblock_offset) * tuple_elements,
			kernel_size.width,
			output_channels * input_channels * tuple_elements * sizeof(float),
			uint32_t(kernel_size.height),
			uint32_t(kernel_size.width),
			0u,
			0u);
	}
}

struct __declspec(align(64)) grad_output_transform_context
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
	const size_t /* output_channel_range */,
	const size_t batch_subblock_size)
{
	const nnp_transform_2d_with_offset transform_function	= context->transform_function;
	const float* grad_output								= context->grad_output;
	float* grad_output_transform							= context->grad_output_transform;
	const size_t tuple_elements								= context->tuple_elements;
	const size_t batch_size									= context->batch_size;
	const size_t output_channels							= context->output_channels;
	const size_t output_channels_block_max					= context->output_channels_block_max;
	const struct nnp_size output_size						= context->output_size;
	const size_t row_offset									= context->row_offset;
	const size_t row_count									= context->row_count;
	const size_t column_offset								= context->column_offset;
	const size_t column_count								= context->column_count;

	const size_t output_channels_block_start  = round_down(output_channel, output_channels_block_max);
	const size_t output_channels_block_size   = min(output_channels - output_channels_block_start, output_channels_block_max);
	const size_t output_channels_block_offset = output_channel - output_channels_block_start;

	for (size_t batch_subblock_offset = 0ull; batch_subblock_offset < batch_subblock_size; batch_subblock_offset++)
	{
		const size_t sample = batch_subblock_start + batch_subblock_offset;
		transform_function(
			grad_output + (sample * output_channels * output_size.width * output_size.height) + (output_channel * output_size.width * output_size.height),
			grad_output_transform +	(output_channels_block_start * batch_size + batch_subblock_start * output_channels_block_size + output_channels_block_offset * batch_subblock_size + batch_subblock_offset) * tuple_elements,
			output_size.width,
			batch_size * output_channels * tuple_elements * sizeof(float),
			uint32_t(row_count),
			uint32_t(column_count),
			uint32_t(row_offset),
			uint32_t(column_offset));
	}
}

struct __declspec(align(64)) grad_input_transform_context
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
	const size_t /* sample_range */,
	const size_t input_channels_subblock_size)
{
	const nnp_transform_2d_with_offset transform_function	= context->transform_function;
	float* grad_input										= context->grad_input;
	const float* grad_input_transform						= context->grad_input_transform;
	const size_t tuple_elements								= context->tuple_elements;
	const size_t input_channels								= context->input_channels;
	const size_t batch_size									= context->batch_size;
	const size_t batch_block_max							= context->batch_block_max;
	const struct nnp_size input_size						= context->input_size;
	const size_t row_offset									= context->row_offset;
	const size_t row_count									= context->row_count;
	const size_t column_offset								= context->column_offset;
	const size_t column_count								= context->column_count;
	
	const size_t batch_block_start  = round_down(sample, batch_block_max);
	const size_t batch_block_size   = min(batch_size - batch_block_start, batch_block_max);
	const size_t batch_block_offset = sample - batch_block_start;

	for (size_t input_channels_subblock_offset = 0ull; input_channels_subblock_offset < input_channels_subblock_size; input_channels_subblock_offset++) 
	{
		const size_t input_channel = input_channels_subblock_start + input_channels_subblock_offset;
		transform_function(
			grad_input_transform + (batch_block_start * input_channels + input_channels_subblock_start * batch_block_size + batch_block_offset * input_channels_subblock_size + input_channels_subblock_offset) * tuple_elements,
			grad_input + (sample * input_channels * input_size.width * input_size.height) + (input_channel * input_size.width * input_size.height),
			batch_size * input_channels * tuple_elements * sizeof(float),
			input_size.width,
			uint32_t(row_count), 
			uint32_t(column_count), 
			uint32_t(row_offset), 
			uint32_t(column_offset));
	}
}

struct __declspec(align(64)) matrix_multiplication_context
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

	float* grad_output_transform;
	float* kernel_transform;
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
	const size_t tuple_elements					= context->tuple_elements;
	const size_t batch_size						= context->batch_size;
	const size_t input_channels					= context->input_channels;
	const size_t batch_block_start				= context->batch_block_start;
	const size_t batch_block_size				= context->batch_block_size;
	const size_t output_channels_block_start	= context->output_channels_block_start;
	const size_t output_channels_block_size		= context->output_channels_block_size;
	const size_t batch_subblock_max				= context->batch_subblock_max;
	const size_t input_channels_subblock_max	= context->input_channels_subblock_max;
	float* grad_output_transform				= context->grad_output_transform + (output_channels_block_start * batch_size + (batch_block_start + batch_subblock_start) * output_channels_block_size) * tuple_elements;
	float* kernel_transform						= context->kernel_transform + (output_channels_block_start * input_channels + input_channels_block_start * output_channels_block_size) * tuple_elements;
	float* grad_input_transform					= context->grad_input_transform + (batch_block_start * input_channels + input_channels_block_start * batch_block_size) * tuple_elements;
	nnp_fast_sgemm_function fast_gemm			= context->fast_gemm;
	nnp_full_sgemm_function full_gemm			= context->full_gemm;

	if (batch_subblock_size == batch_subblock_max) 
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
		
	while (input_channels_block_size != 0ull) 
	{
		const size_t input_channels_subblock_size = min(input_channels_block_size, input_channels_subblock_max);
		input_channels_block_size -= input_channels_subblock_size;

		full_gemm(
			uint32_t(batch_subblock_size), 
			uint32_t(input_channels_subblock_size),
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
	nnp_transform_2d_with_offset grad_output_transform_function,
	nnp_transform_2d_with_offset kernel_transform_function,
	nnp_transform_2d_with_offset grad_input_transform_function)
{
	const size_t simd_width = nnp_hwinfo.simd_width;
	const size_t tuple_elements = (fourier_transform ? simd_width << 1ull : simd_width);
	const size_t tile_elements = tile_size.height * tile_size.width;
	const size_t tuple_count = tile_elements / tuple_elements;

	const struct nnp_size grad_input_tile_size = { tile_size.width - kernel_size.width + 1ull, tile_size.height - kernel_size.height + 1ull };
	
	/* Calculate cache blocking parameters */
	const size_t cache_elements_l1 = nnp_hwinfo.blocking.l1 / (tuple_elements * sizeof(float));
	const size_t cache_elements_l2 = nnp_hwinfo.blocking.l2 / (tuple_elements * sizeof(float));
	const size_t cache_elements_l3 = nnp_hwinfo.blocking.l3 / (tuple_elements * sizeof(float));

	const size_t batch_subblock_max = (fourier_transform ? nnp_hwinfo.cxgemm.mr : nnp_hwinfo.sxgemm.mr);
	const size_t input_channels_subblock_max = (fourier_transform ? nnp_hwinfo.cxgemm.nr : nnp_hwinfo.sxgemm.nr);

	const size_t output_channels_block_max = round_down(cache_elements_l1 / (batch_subblock_max + input_channels_subblock_max), 2ull);
	const size_t batch_block_max = round_down(cache_elements_l3 / output_channels_block_max, batch_subblock_max);
	const size_t input_channels_block_max =	round_down(cache_elements_l2 / output_channels_block_max, input_channels_subblock_max);

	/* Calculate memory footprint and allocate memory */
	const size_t kernel_transform_size = output_channels * input_channels * tile_elements * sizeof(float);
	const size_t grad_input_transform_size = batch_size * input_channels * tile_elements * sizeof(float);
	const size_t grad_output_transform_size = batch_size * output_channels * tile_elements * sizeof(float);
	
	void* memory_block_kernel = NULL;
	void* memory_block_grad_input = NULL;
	void* memory_block_grad_output = NULL;

	if (workspace_buffer == NULL)
	{
		memory_block_kernel = _aligned_malloc(kernel_transform_size, 64ull);
		memory_block_grad_input = _aligned_malloc(grad_input_transform_size, 64ull);
		memory_block_grad_output = _aligned_malloc(grad_output_transform_size, 64ull);

		if (memory_block_kernel == NULL || memory_block_grad_input == NULL || memory_block_grad_output == NULL)
			return nnp_status_out_of_memory;
	}
	else
	{
		if (workspace_buffer->kernel == NULL || workspace_buffer->input == NULL || workspace_buffer->output == NULL)
		{
			memory_block_kernel = _aligned_malloc(kernel_transform_size, 64ull);
			memory_block_grad_input = _aligned_malloc(grad_input_transform_size, 64ull);
			memory_block_grad_output = _aligned_malloc(grad_output_transform_size, 64ull);

			if (memory_block_kernel == NULL || memory_block_grad_input == NULL || memory_block_grad_output == NULL)
				return nnp_status_out_of_memory;

			*workspace_buffer = nnp_workspace_pointers{ memory_block_kernel, memory_block_grad_input, memory_block_grad_output };
		}
		else
		{
			memory_block_kernel = workspace_buffer->kernel;
			memory_block_grad_input = workspace_buffer->input;
			memory_block_grad_output = workspace_buffer->output;
		}
	}

	float* kernel_transform = static_cast<float*>(memory_block_kernel);
	float* grad_output_transform = static_cast<float*>(memory_block_grad_output);
	float* grad_input_transform = static_cast<float*>(memory_block_grad_input);

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
		1u, input_channels_subblock_max);
	
	for (size_t y = 0ull; y < input_size.height; y += grad_input_tile_size.height) 
	{
		const size_t grad_output_y = min(doz(y + input_padding.top, kernel_size.height - 1ull), output_size.height);
		for (size_t x = 0ull; x < input_size.width; x += grad_input_tile_size.width) 
		{
			const size_t grad_output_x = min(doz(x + input_padding.left, kernel_size.width - 1ull), output_size.width);

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
				doz(kernel_size.height - 1ull, y + input_padding.top),
				min(output_size.height - grad_output_y,	tile_size.height - grad_output_transform_context.row_offset),
				doz(kernel_size.width - 1ull, x + input_padding.left),
				min(output_size.width - grad_output_x, tile_size.width - grad_output_transform_context.column_offset)
			};
			pthreadpool_compute_2d_tiled(
				(pthreadpool_function_2d_tiled_t)compute_grad_output_transform,
				&grad_output_transform_context,
				output_channels,
				batch_size,
				1u,
				batch_subblock_max);
			
			for (size_t tuple_index = 0ull; tuple_index < tuple_count; tuple_index++)
			{
				for (size_t output_channels_block_start = 0ull; output_channels_block_start < output_channels; output_channels_block_start += output_channels_block_max) 
				{
					const size_t output_channels_block_size = min(output_channels - output_channels_block_start, output_channels_block_max);
					for (size_t batch_block_start = 0ull; batch_block_start < batch_size; batch_block_start += batch_block_max) 
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
							input_channels,
							batch_block_size,
							input_channels_block_max,
							batch_subblock_max);
					}
				}
			}
		
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
				fourier_transform ? kernel_size.height - 1ull : 0ull,
				min(input_size.height - y, grad_input_tile_size.height),
				fourier_transform ? kernel_size.width - 1ull : 0ull,
				min(input_size.width - x, grad_input_tile_size.width)
			};
			pthreadpool_compute_2d_tiled(
				(pthreadpool_function_2d_tiled_t)compute_grad_input_transform,
				&grad_input_transform_context,
				batch_size,
				input_channels,
				1ull,
				input_channels_subblock_max);
		}
	}
	
	if (workspace_buffer == NULL)
	{
		_aligned_free(memory_block_kernel);
		_aligned_free(memory_block_grad_input);
		_aligned_free(memory_block_grad_output);
	}
	else
	{
		if (memory_block_kernel != workspace_buffer->kernel || memory_block_grad_input != workspace_buffer->input || memory_block_grad_output != workspace_buffer->output)
		{
			_aligned_free(memory_block_kernel);
			_aligned_free(memory_block_grad_input);
			_aligned_free(memory_block_grad_output);
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
	struct nnp_workspace_pointers* workspace_buffer)
{
	const struct nnp_size output_size =
	{
		input_padding.left + input_size.width + input_padding.right - kernel_size.width + 1ull,
		input_padding.top + input_size.height + input_padding.bottom - kernel_size.height + 1ull
	};

	if (algorithm == nnp_convolution_algorithm_auto) 
	{
		if (max(kernel_size.width, kernel_size.height) > 8ull) 
			algorithm = nnp_convolution_algorithm_ft16x16;
		else 
		{
			const size_t tile_count_8x8 = divide_round_up(input_size.height, 8ull - kernel_size.height + 1ull) * divide_round_up(input_size.width, 8ull - kernel_size.width + 1ull);
			const size_t tile_count_16x16 =	divide_round_up(input_size.height, 16ull - kernel_size.height + 1ull) * divide_round_up(input_size.width, 16ull - kernel_size.width + 1ull);
			if (tile_count_8x8 <= 4 * tile_count_16x16) 
			{
				/* 8x8 tiles are more efficient */
				if ((kernel_size.height == 3ull) && (kernel_size.width == 3ull)) 
					algorithm = nnp_convolution_algorithm_wt8x8;
				else 
					algorithm = nnp_convolution_algorithm_ft8x8;
			} 
			else 
				algorithm = nnp_convolution_algorithm_ft16x16;
		}
	}

	/* Choose tiling parameters and transform functions depending on convolution algorithm */
	struct nnp_size tile_size;
	bool fourier_transform;
	nnp_transform_2d_with_offset grad_output_transform_function;
	nnp_transform_2d_with_offset kernel_transform_function;
	nnp_transform_2d_with_offset grad_input_transform_function;
	switch (algorithm) 
	{
	case nnp_convolution_algorithm_wt8x8:
	{
		if ((kernel_size.height != 3ull) || (kernel_size.width != 3ull))
			return nnp_status_unsupported_algorithm;

		grad_output_transform_function = nnp_hwinfo.transforms.iwt_f6x6_3x3_with_offset_and_stream;
		kernel_transform_function = nnp_hwinfo.transforms.kwt_f6x6_3Rx3R;
		grad_input_transform_function = nnp_hwinfo.transforms.owt_f6x6_3x3;
		tile_size = nnp_size{ 8ull, 8ull };
		fourier_transform = false;
	}
	break;

	case nnp_convolution_algorithm_ft8x8:
	{
		grad_output_transform_function = nnp_hwinfo.transforms.fft8x8_with_offset_and_stream;
		kernel_transform_function = nnp_hwinfo.transforms.fft8x8_with_offset_and_stream;
		grad_input_transform_function = nnp_hwinfo.transforms.ifft8x8_with_offset;
		tile_size = nnp_size{ 8ull, 8ull };
		fourier_transform = true;
	}
	break;

	case nnp_convolution_algorithm_ft16x16:
	{
		grad_output_transform_function = nnp_hwinfo.transforms.fft16x16_with_offset_and_stream;
		kernel_transform_function = nnp_hwinfo.transforms.fft16x16_with_offset_and_stream;
		grad_input_transform_function = nnp_hwinfo.transforms.ifft16x16_with_offset;
		tile_size = nnp_size{ 16ull, 16ull };
		fourier_transform = true;
	}
	break;

	default:
		return nnp_status_invalid_algorithm;
		break;
	}

	return compute_fast_convolution_input_gradient(fourier_transform, batch_size, input_channels, output_channels,	tile_size, input_size, input_padding, kernel_size, output_size,	grad_output, kernel, grad_input, workspace_buffer, grad_output_transform_function, kernel_transform_function, grad_input_transform_function);
}
