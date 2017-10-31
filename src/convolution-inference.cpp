#include <fxdiv.h>

#include <nnpack.h>
#include <utils.h>

#include <hwinfo.h>
#include <activations.h>
#include <system.h>
#include <validation.h>


struct NNP_CACHE_ALIGN kernel_transform_context 
{
	nnp_transform_2d_with_offset transform_function;
	const float* kernel;
	float* kernel_transform;
	
	size_t tuple_size;
	size_t input_channels;
	size_t input_channels_block_size;
	size_t output_channels;
	nnp_size kernel_size;
};

static void compute_kernel_transform(
	const kernel_transform_context* context,
	const size_t output_channels_subblock_start,
	const size_t input_channels_block_offset,
	const size_t output_channels_subblock_size,
	const size_t input_channels_block_increment)
{
	const nnp_transform_2d_with_offset transform_function = context->transform_function;
	const float* kernel                                   = context->kernel;
	float* kernel_transform                               = context->kernel_transform;
	const size_t tuple_size                               = context->tuple_size;
	const size_t input_channels                           = context->input_channels;
	const size_t input_channels_block_size                = context->input_channels_block_size;
	const size_t output_channels                          = context->output_channels;
	const nnp_size kernel_size                            = context->kernel_size;
	
	for (size_t output_channels_subblock_offset = 0ull; output_channels_subblock_offset < output_channels_subblock_size; output_channels_subblock_offset++) 
	{
		const size_t output_channel = output_channels_subblock_start + output_channels_subblock_offset;

		transform_function(
			kernel + (output_channel * input_channels * kernel_size.width * kernel_size.height) + (input_channels_block_offset * kernel_size.width * kernel_size.height),
			kernel_transform + (output_channels_subblock_start * input_channels_block_size + input_channels_block_offset * output_channels_subblock_size + output_channels_subblock_offset) * tuple_size,
			kernel_size.width,
			input_channels_block_size * output_channels * tuple_size,
			uint32_t(kernel_size.height),
			uint32_t(kernel_size.width),
			0u, 
			0u);
	}
}

struct NNP_CACHE_ALIGN input_transform_context
{
	const float* input;
	float* input_transform;
	nnp_transform_2d_with_offset transform_function;

	size_t tuple_size;
	size_t tiles_count;
	fxdiv_divisor_size_t tiles_x_count;
	size_t input_channels_block_start;
	size_t input_channels_block_size;
	nnp_size input_size;
	size_t input_padding_left;
	size_t input_padding_top;
	nnp_size input_tile;
	nnp_size output_tile;
};

static void compute_input_transform(
	const input_transform_context* context,
	const size_t input_channels_block_offset,
	const size_t tiles_subblock_start,
	const size_t input_channels_block_range,
	const size_t tiles_subblock_size)
{
	const nnp_transform_2d_with_offset transform_function = context->transform_function;
	const float* input                                    = context->input;
	float* input_transform                                = context->input_transform;
	const size_t tuple_size                               = context->tuple_size;
	const size_t tiles_count                              = context->tiles_count;
	const fxdiv_divisor_size_t tiles_x_count              = context->tiles_x_count;
	const size_t input_channels_block_start               = context->input_channels_block_start;
	const size_t input_channels_block_size                = context->input_channels_block_size;
	const nnp_size input_size                             = context->input_size;
	const size_t input_padding_left                       = context->input_padding_left;
	const size_t input_padding_top                        = context->input_padding_top;
	const nnp_size input_tile                             = context->input_tile;
	const nnp_size output_tile                            = context->output_tile;
	
	const size_t input_channel = input_channels_block_start + input_channels_block_offset;
	for (size_t tiles_subblock_offset = 0ull; tiles_subblock_offset < tiles_subblock_size; tiles_subblock_offset++)
	{
		const size_t tile = tiles_subblock_start + tiles_subblock_offset;
		const fxdiv_result_size_t tile_xy = fxdiv_divide_size_t(tile, tiles_x_count);
		const size_t tile_x = tile_xy.remainder;
		const size_t tile_y = tile_xy.quotient;

		const size_t output_x = tile_x * output_tile.width;
		const size_t output_y = tile_y * output_tile.height;

		const size_t input_x = min(doz(output_x, input_padding_left), input_size.width);
		const size_t input_y = min(doz(output_y, input_padding_top), input_size.height);

		const size_t row_offset = doz(input_padding_top, output_y);
		const size_t row_count = min(input_size.height - input_y, input_tile.height - row_offset);
		const size_t column_offset = doz(input_padding_left, output_x);
		const size_t column_count = min(input_size.width - input_x, input_tile.width - column_offset);
		
		transform_function(
			input + (input_channel * input_size.width * input_size.height) + (input_y * input_size.width) + input_x,
			input_transform + (tiles_subblock_start * input_channels_block_size + input_channels_block_offset * tiles_subblock_size + tiles_subblock_offset) * tuple_size,
			input_size.width,
			input_channels_block_size * tiles_count * tuple_size,
			uint32_t(row_count),
			uint32_t(column_count),
			uint32_t(row_offset),
			uint32_t(column_offset));
	}
}

struct NNP_CACHE_ALIGN output_transform_context
{
	nnp_transform_2d_with_bias transform_function;
	float* output;
	const float* output_transform;
	const float* bias;

	size_t tuple_size;
	size_t tiles_count;
	fxdiv_divisor_size_t tiles_x_count;
	fxdiv_divisor_size_t tiles_block_max;
	size_t output_channels;
	nnp_size output_size;
	nnp_size output_tile;
};

static void compute_output_transform(
	const output_transform_context* context,
	const size_t output_channels_subblock_start,
	const size_t tiles_subblock_start,
	const size_t output_channels_subblock_size,
	const size_t tiles_subblock_size)
{
	const nnp_transform_2d_with_bias transform_function = context->transform_function;
	float* output                                       = context->output;
	const float* output_transform                       = context->output_transform;
	const float* bias                                   = context->bias;
	const size_t tuple_size                             = context->tuple_size;
	const size_t tiles_count                            = context->tiles_count;
	const fxdiv_divisor_size_t tiles_x_count            = context->tiles_x_count;
	const fxdiv_divisor_size_t tiles_block_max          = context->tiles_block_max;
	const size_t output_channels                        = context->output_channels;
	const nnp_size output_size                          = context->output_size;
	const nnp_size output_tile                          = context->output_tile;

	const size_t tiles_block_start = fxdiv_round_down_size_t(tiles_subblock_start, tiles_block_max);
	const size_t tiles_block_size = min(tiles_count - tiles_block_start, tiles_block_max.value);
	
	for (size_t tiles_subblock_offset = 0ull; tiles_subblock_offset < tiles_subblock_size; tiles_subblock_offset++)
	{
		const size_t tile = tiles_subblock_start + tiles_subblock_offset;
		const fxdiv_result_size_t tile_xy = fxdiv_divide_size_t(tile, tiles_x_count);

		const size_t tile_x = tile_xy.remainder;
		const size_t tile_y = tile_xy.quotient;

		const size_t output_x = tile_x * output_tile.width;
		const size_t output_y = tile_y * output_tile.height;

		for (size_t output_channels_subblock_offset = 0ull; output_channels_subblock_offset < output_channels_subblock_size; output_channels_subblock_offset++) 
		{
			const size_t output_channel = output_channels_subblock_start + output_channels_subblock_offset;
			transform_function(
				output_transform + (tiles_block_start * output_channels + output_channels_subblock_start * tiles_block_size + ((tiles_subblock_start - tiles_block_start) + tiles_subblock_offset) * output_channels_subblock_size + output_channels_subblock_offset) * tuple_size,
				output + (output_channel * output_size.width * output_size.height) + (output_y * output_size.width) + output_x,
				bias + output_channel,
				tiles_count * output_channels * tuple_size,
				output_size.width,
				uint32_t(min(output_tile.height, output_size.height - output_y)),
				uint32_t(min(output_tile.width, output_size.width - output_x)));
		}
	}
}

struct NNP_CACHE_ALIGN tuple_multiplication_context
{
	size_t tuple_elements;
	size_t tuple_size;
	size_t tiles_subblock_max;
	size_t input_channels_block_size;
	size_t input_channels_block_start;
	size_t output_channels;
	size_t output_channels_subblock_max;
	size_t output_channels_block_start;

	const float* input_transform;
	const float* kernel_transform;
	float* output_transform;

	nnp_fast_tuple_gemm_function fast_gemm;
	nnp_full_tuple_gemm_function full_gemm;
};

static void compute_tuple_multiplication(
	const tuple_multiplication_context* context,
	const size_t tiles_block_start,
	const size_t output_channels_subblock_start,
	size_t tiles_block_size,
	const size_t output_channels_subblock_size)
{
	const size_t tuple_elements               = context->tuple_elements;
	const size_t tuple_size                   = context->tuple_size;
	const size_t tiles_subblock_max           = context->tiles_subblock_max;
	const size_t input_channels_block_size    = context->input_channels_block_size;
	const size_t input_channels_block_start   = context->input_channels_block_start;
	const size_t output_channels              = context->output_channels;
	const size_t output_channels_subblock_max = context->output_channels_subblock_max;
	const size_t output_channels_block_start  = context->output_channels_block_start;

	const float* input_transform              = context->input_transform + tiles_block_start * input_channels_block_size * tuple_size;
	const float* kernel_transform             = context->kernel_transform +	(output_channels_block_start + output_channels_subblock_start) * input_channels_block_size * tuple_size;
	float* output_transform                   = context->output_transform + (tiles_block_start * output_channels + (output_channels_block_start + output_channels_subblock_start) * tiles_block_size) * tuple_size;

	if (output_channels_subblock_size == output_channels_subblock_max) 
	{
		const nnp_fast_tuple_gemm_function fast_gemm = context->fast_gemm;
		while (tiles_block_size >= tiles_subblock_max) 
		{
			tiles_block_size -= tiles_subblock_max;

			fast_gemm(
				input_channels_block_size,
				input_channels_block_start,
				input_transform,
				kernel_transform,
				output_transform,
				output_channels_subblock_size * tuple_elements);

			input_transform  += tiles_subblock_max * input_channels_block_size * tuple_size;
			output_transform += tiles_subblock_max * output_channels_subblock_size * tuple_size;
		}
	}

	const nnp_full_tuple_gemm_function full_gemm = context->full_gemm;
	while (tiles_block_size != 0ull) 
	{
		const size_t tiles_subblock_size = min(tiles_block_size, tiles_subblock_max);
		tiles_block_size -= tiles_subblock_size;

		full_gemm(
			uint32_t(tiles_subblock_size),
			uint32_t(output_channels_subblock_size),
			input_channels_block_size,
			input_channels_block_start,
			input_transform,
			kernel_transform,
			output_transform,
			output_channels_subblock_size * tuple_elements);

		input_transform  += tiles_subblock_max * input_channels_block_size * tuple_size;
		output_transform += tiles_subblock_max * output_channels_subblock_size * tuple_size;
	}
}

struct NNP_CACHE_ALIGN kernel_packing_context
{
	const float* kernel;
	float* packed_kernel;

	size_t reduction_size;
	size_t reduction_block_start;
	size_t reduction_block_size;
};

static void compute_kernel_packing(
	const kernel_packing_context* context,
	const size_t output_channels_subblock_start,
	const size_t reduction_block_offset,
	const size_t output_channels_subblock_size,
	const size_t reduction_block_range)
{
	const size_t reduction_size        = context->reduction_size;
	const size_t reduction_block_start = context->reduction_block_start;
	const size_t reduction_block_size  = context->reduction_block_size;
	const float* kernel                = context->kernel + output_channels_subblock_start * reduction_size + reduction_block_offset;
	float* packed_kernel               = context->packed_kernel + output_channels_subblock_start * reduction_block_size + reduction_block_offset * output_channels_subblock_size;
		
	for (size_t output_channels_subblock_offset = 0ull; output_channels_subblock_offset < output_channels_subblock_size; output_channels_subblock_offset++) 
		packed_kernel[output_channels_subblock_offset] = kernel[output_channels_subblock_offset * reduction_size];
}

struct NNP_CACHE_ALIGN input_packing_context
{
	const float* input;
	float* packed_input;

	size_t simd_width;
	size_t reduction_block_start;
	size_t reduction_block_size;
	size_t output_image_block_start;
	nnp_size input_size;
	size_t input_padding_top;
	size_t input_padding_left;
	fxdiv_divisor_size_t kernel_elements;
	fxdiv_divisor_size_t kernel_width;
	fxdiv_divisor_size_t output_width;
	nnp_size output_subsampling;
};

static void compute_input_packing(
	const input_packing_context* context,
	const size_t reduction_block_offset,
	const size_t output_image_subblock_start,
	const size_t reduction_block_range,
	const size_t output_image_subblock_size)
{
	const float* input                          = context->input;
	float* packed_input                         = context->packed_input;
	const size_t simd_width                     = context->simd_width;
	const size_t reduction_block_start          = context->reduction_block_start;
	const size_t reduction_block_size           = context->reduction_block_size;
	const size_t output_image_block_start       = context->output_image_block_start;
	const nnp_size input_size                   = context->input_size;
	const size_t input_padding_top              = context->input_padding_top;
	const size_t input_padding_left             = context->input_padding_left;
	const fxdiv_divisor_size_t kernel_elements  = context->kernel_elements;
	const fxdiv_divisor_size_t kernel_width     = context->kernel_width;
	const fxdiv_divisor_size_t output_width     = context->output_width;
	const nnp_size output_subsampling           = context->output_subsampling;

	const size_t output_image_subblock_stride = round_up_by_power_of_2(output_image_subblock_size, simd_width);

	const size_t reduction_index = reduction_block_start + reduction_block_offset;
	const fxdiv_result_size_t reduction_index_divmod = fxdiv_divide_size_t(reduction_index, kernel_elements);
	const size_t input_channel = reduction_index_divmod.quotient;
	const fxdiv_result_size_t kernel_xy = fxdiv_divide_size_t(reduction_index_divmod.remainder, kernel_width);
	const size_t kernel_y = kernel_xy.quotient;
	const size_t kernel_x = kernel_xy.remainder;

	for (size_t output_image_subblock_offset = 0ull; output_image_subblock_offset < output_image_subblock_size; output_image_subblock_offset++) 
	{
		const size_t output_image_index = output_image_block_start + output_image_subblock_start + output_image_subblock_offset;
		const fxdiv_result_size_t output_xy = fxdiv_divide_size_t(output_image_index, output_width);
		const size_t output_y = output_xy.quotient;
		const size_t output_x = output_xy.remainder;

		const size_t input_y = output_y * output_subsampling.height + kernel_y - input_padding_top;
		const size_t input_x = output_x * output_subsampling.width  + kernel_x - input_padding_left;

		const size_t packed_index = output_image_subblock_start * reduction_block_size + reduction_block_offset * output_image_subblock_stride + output_image_subblock_offset;
		if (input_x < input_size.width && input_y < input_size.height)
			packed_input[packed_index] = input[(input_channel * input_size.width * input_size.height) + (input_y * input_size.width) + input_x];
		else 
			packed_input[packed_index] = 0.0f;
	}
}

struct NNP_CACHE_ALIGN matrix_multiplication_context
{
	const float* packed_kernel;
	const float* packed_input;
	float* output;

	size_t reduction_block_start;
	size_t reduction_block_size;
	size_t output_image_size;
	size_t output_image_block_start;
	size_t output_image_subblock_max;
	size_t output_channels_subblock_max;
};

static void compute_matrix_multiplication(
	const matrix_multiplication_context* context,
	const size_t output_channels_block_start,
	const size_t output_image_subblock_start,
	size_t output_channels_block_size,
	const size_t output_image_subblock_size)
{
	const size_t reduction_block_start        = context->reduction_block_start;
	const size_t reduction_block_size         = context->reduction_block_size;
	const size_t output_image_size            = context->output_image_size;
	const size_t output_image_block_start     = context->output_image_block_start;
	const size_t output_image_subblock_max    = context->output_image_subblock_max;
	const size_t output_channels_subblock_max = context->output_channels_subblock_max;
	const float* packed_kernel                = context->packed_kernel + output_channels_block_start * reduction_block_size;
	const float* packed_input                 = context->packed_input + output_image_subblock_start * reduction_block_size;
	float* output                             = context->output + output_channels_block_start * output_image_size + output_image_block_start + output_image_subblock_start;
		
	if (output_image_subblock_size == output_image_subblock_max) 
	{
		const nnp_fast_sgemm_function fast_gemm = nnp_hwinfo.sgemm.only_mr_x_nr;
		while (output_channels_block_size >= output_channels_subblock_max) 
		{
			output_channels_block_size -= output_channels_subblock_max;

			fast_gemm(
				reduction_block_size,
				reduction_block_start,
				packed_kernel,
				packed_input,
				output,
				output_image_size);

			packed_kernel += reduction_block_size * output_channels_subblock_max;
			output        += output_image_size    * output_channels_subblock_max;
		}
	}

	const nnp_full_sgemm_function full_gemm = nnp_hwinfo.sgemm.upto_mr_x_nr;
	while (output_channels_block_size != 0ull) 
	{
		const size_t output_channels_subblock_size = min(output_channels_block_size, output_channels_subblock_max);
		output_channels_block_size -= output_channels_subblock_size;

		full_gemm(
			uint32_t(output_channels_subblock_size),
			uint32_t(output_image_subblock_size),
			reduction_block_size,
			reduction_block_start,
			packed_kernel,
			packed_input,
			output,
			output_image_size);

		packed_kernel += reduction_block_size * output_channels_subblock_max;
		output        += output_image_size    * output_channels_subblock_max;
	}
}

struct NNP_CACHE_ALIGN direct_convolution_context
{
	const float* input;
	const float* kernel;
	float* output;

	size_t image_elements;
	size_t input_channels;
	size_t input_channels_block_max;
	size_t output_channels_block_max;

	nnp_fast_conv_function fast_conv;
	nnp_full_conv_function full_conv;
};

static void compute_direct_convolution(
	const direct_convolution_context* context,
	const size_t output_channels_block_start, 
	const size_t output_channels_block_size)
{
	
	const size_t image_elements            = context->image_elements;
	const size_t input_channels            = context->input_channels;
	const size_t input_channels_block_max  = context->input_channels_block_max;
	const size_t output_channels_block_max = context->output_channels_block_max;
	const float* input                     = context->input;
	const float* kernel                    = context->kernel + output_channels_block_start * input_channels;
	float* output                          = context->output + output_channels_block_start * image_elements;

	memset(output, 0, sizeof(float) * output_channels_block_size * image_elements);

	size_t input_channels_unprocessed = input_channels;
	if (output_channels_block_size == output_channels_block_max) 
	{
		const nnp_fast_conv_function fast_conv = context->fast_conv;
		while (input_channels_unprocessed >= input_channels_block_max) 
		{
			input_channels_unprocessed -= input_channels_block_max;

			fast_conv(
				input_channels,
				image_elements,
				input,
				kernel,
				output);

			input  += input_channels_block_max * image_elements;
			kernel += input_channels_block_max;
		}
	}

	const nnp_full_conv_function full_conv = context->full_conv;
	while (input_channels_unprocessed != 0ull) 
	{
		const size_t input_channels_block_size = min(input_channels_unprocessed, input_channels_block_max);
		input_channels_unprocessed -= input_channels_block_size;

		full_conv(
			uint32_t(input_channels_block_size),
			uint32_t(output_channels_block_size),
			input_channels,
			image_elements,
			input,
			kernel,
			output);

		input  += input_channels_block_max * image_elements;
		kernel += input_channels_block_max;
	}
}

static nnp_status compute_fast_convolution_inference(
	const bool fourier_transform,
	const nnp_convolution_transform_strategy transform_strategy,
	const size_t transform_element_size,
	const size_t input_channels,
	const size_t output_channels,
	const nnp_size tile_size,
	const nnp_size input_size,
	const nnp_padding input_padding,
	const nnp_size kernel_size,
	const nnp_size output_size,
	const float* input,
	const float* kernel,
	const float* bias,
	float* output,
	nnp_workspace_pointers* workspace_buffer,
	const nnp_transform_2d_with_offset input_transform_function,
	const nnp_transform_2d_with_offset kernel_transform_function,
	const nnp_transform_2d_with_bias output_transform_function,
	nnp_profile* profile)
{
	const size_t simd_width = nnp_hwinfo.simd_width;
	const size_t tuple_elements = (fourier_transform ? simd_width * 2ull : simd_width);
	const size_t tuple_size = tuple_elements * transform_element_size;
	const size_t tile_elements = tile_size.height * tile_size.width;
	const size_t tuple_count = tile_elements / tuple_elements;

	const nnp_size output_tile_size = { tile_size.width - kernel_size.width + 1ull, tile_size.height - kernel_size.height + 1ull };

	const size_t tiles_y_count = divide_round_up(output_size.height, output_tile_size.height);
	const size_t tiles_x_count = divide_round_up(output_size.width, output_tile_size.width);
	const size_t tiles_count = tiles_x_count * tiles_y_count;

	/* Calculate cache blocking parameters */
	const size_t cache_elements_l1 = nnp_hwinfo.blocking.l1 / tuple_size;
	const size_t cache_elements_l2 = nnp_hwinfo.blocking.l2 / tuple_size;
	const size_t cache_elements_l3 = nnp_hwinfo.blocking.l3 / tuple_size;

	const size_t tiles_subblock_max = (fourier_transform ? nnp_hwinfo.cxgemm.mr : nnp_hwinfo.sxgemm.mr);
	const size_t output_channels_subblock_max = (fourier_transform ? nnp_hwinfo.cxgemm.nr : nnp_hwinfo.sxgemm.nr);

	const size_t input_channels_block_max = round_down(cache_elements_l1 / (tiles_subblock_max + output_channels_subblock_max), 2ull);
	const size_t tiles_block_max = round_down(cache_elements_l2 / input_channels_block_max, tiles_subblock_max);
	const size_t output_channels_block_max = round_down(cache_elements_l3 / input_channels_block_max, output_channels_subblock_max);

	const size_t transform_tile_size = tile_elements * transform_element_size;
	const size_t input_transform_size = tiles_count * min(input_channels, input_channels_block_max) * transform_tile_size;
	const size_t output_transform_size = tiles_count * output_channels * transform_tile_size;
	const size_t kernel_transform_size = output_channels * min(input_channels, input_channels_block_max) * transform_tile_size;

	void* memory_block_input = NULL;
	void* memory_block_output = NULL;
	void* memory_block_kernel = NULL;
	
	if (workspace_buffer == NULL)
	{
		memory_block_input = allocate_memory(input_transform_size);
		memory_block_output = allocate_memory(output_transform_size);
		memory_block_kernel = allocate_memory(kernel_transform_size);
		
		if (memory_block_input == NULL || memory_block_output == NULL || memory_block_kernel == NULL)
			return nnp_status_out_of_memory;
	}
	else
	{
		if (workspace_buffer->kernel == NULL || workspace_buffer->input == NULL || workspace_buffer->output == NULL)
		{
			memory_block_input = allocate_memory(input_transform_size);
			memory_block_output = allocate_memory(output_transform_size);
			memory_block_kernel = allocate_memory(kernel_transform_size);
			
			if (memory_block_input == NULL || memory_block_output == NULL || memory_block_kernel == NULL)
				return nnp_status_out_of_memory;

			*workspace_buffer = nnp_workspace_pointers{ memory_block_kernel, memory_block_input, memory_block_output };
		}
		else
		{
			memory_block_input = workspace_buffer->input;
			memory_block_output = workspace_buffer->output;
			memory_block_kernel = workspace_buffer->kernel;
		}
	}

	float* input_transform = static_cast<float*>(memory_block_input);
	float* output_transform = static_cast<float*>(memory_block_output);
	float* kernel_transform = static_cast<float*>(memory_block_kernel);

	for (size_t input_channels_block_start = 0ull; input_channels_block_start < input_channels; input_channels_block_start += input_channels_block_max)
	{
		const size_t input_channels_block_size = min(input_channels - input_channels_block_start, input_channels_block_max);
			
		NNP_KERNEL_TRANSFORM_START(profile)
		kernel_transform_context kernel_transform_context =
		{
			kernel_transform_function,
			kernel + input_channels_block_start * kernel_size.height * kernel_size.width,
			kernel_transform,
			tuple_size,
			input_channels,
			input_channels_block_size,
			output_channels,
			kernel_size
		};
		pthreadpool_compute_2d_tiled(
			(pthreadpool_function_2d_tiled_t)compute_kernel_transform,
			&kernel_transform_context,
			output_channels,
			input_channels_block_size,
			output_channels_subblock_max,
			1ull);
		NNP_KERNEL_TRANSFORM_END(profile)

		NNP_INPUT_TRANSFORM_START(profile)
		input_transform_context input_transform_context =
		{
			input,
			input_transform,
			input_transform_function,
			tuple_size,
			tiles_count,
			fxdiv_init_size_t(tiles_x_count),
			input_channels_block_start,
			input_channels_block_size,
			input_size,
			input_padding.left,
			input_padding.top,
			tile_size,
			output_tile_size
		};
		pthreadpool_compute_2d_tiled(
			(pthreadpool_function_2d_tiled_t)compute_input_transform,
			&input_transform_context,
			input_channels_block_size,
			tiles_count,
			1ull,
			tiles_subblock_max);
		NNP_INPUT_TRANSFORM_END(profile)

		NNP_BLOCK_MULTIPLICATION_START(profile)
		for (size_t tuple_index = 0ull; tuple_index < tuple_count; tuple_index++)
		{
			nnp_fast_tuple_gemm_function fast_gemm_function;
			nnp_full_tuple_gemm_function full_gemm_function;
			if (fourier_transform)
			{
				if (tuple_index < NNP_COMPLEX_TUPLE_INDEX)
				{
					fast_gemm_function = nnp_hwinfo.cxgemm.s4cX_conjb_only_mr_x_nr;
					full_gemm_function = nnp_hwinfo.cxgemm.s4cX_conjb_upto_mr_x_nr;
				}
				else
				{
					fast_gemm_function = nnp_hwinfo.cxgemm.cX_conjb_only_mr_x_nr;
					full_gemm_function = nnp_hwinfo.cxgemm.cX_conjb_upto_mr_x_nr;
				}
			}
			else
			{
				fast_gemm_function = nnp_hwinfo.sxgemm.only_mr_x_nr;
				full_gemm_function = nnp_hwinfo.sxgemm.upto_mr_x_nr;
			}

			for (size_t output_channels_block_start = 0ull; output_channels_block_start < output_channels; output_channels_block_start += output_channels_block_max)
			{
				const size_t output_channels_block_size = min(output_channels - output_channels_block_start, output_channels_block_max);
				tuple_multiplication_context tuple_multiplication_context =
				{
					tuple_elements,
					tuple_size,
					tiles_subblock_max,
					input_channels_block_size,
					input_channels_block_start,
					output_channels,
					output_channels_subblock_max,
					output_channels_block_start,
					input_transform + tuple_index * tiles_count * input_channels_block_size * tuple_size,
					kernel_transform + tuple_index * output_channels * input_channels_block_size * tuple_size,
					output_transform + tuple_index * tiles_count * output_channels * tuple_size,
					fast_gemm_function,
					full_gemm_function
				};
				pthreadpool_compute_2d_tiled(
					(pthreadpool_function_2d_tiled_t)compute_tuple_multiplication,
					&tuple_multiplication_context,
					tiles_count,
					output_channels_block_size,
					tiles_block_max,
					output_channels_subblock_max);
			}
		}
		NNP_BLOCK_MULTIPLICATION_END(profile)
	}

	NNP_OUTPUT_TRANSFORM_START(profile)
	output_transform_context output_transform_context =
	{
		output_transform_function,
		output,
		output_transform,
		bias,
		tuple_size,
		tiles_count,
		fxdiv_init_size_t(tiles_x_count),
		fxdiv_init_size_t(tiles_block_max),
		output_channels,
		output_size,
		output_tile_size
	};
	pthreadpool_compute_2d_tiled(
		(pthreadpool_function_2d_tiled_t)compute_output_transform,
		&output_transform_context,
		output_channels,
		tiles_count,
		output_channels_subblock_max,
		tiles_subblock_max);
	NNP_OUTPUT_TRANSFORM_END(profile)

	if (workspace_buffer == NULL)
	{
		release_memory(memory_block_input, input_transform_size);
		release_memory(memory_block_output, output_transform_size);
		release_memory(memory_block_kernel, kernel_transform_size);
	}
	else
	{
		if (memory_block_kernel != workspace_buffer->kernel || memory_block_input != workspace_buffer->input || memory_block_output != workspace_buffer->output)
		{
			release_memory(memory_block_input, input_transform_size);
			release_memory(memory_block_output, output_transform_size);
			release_memory(memory_block_kernel, kernel_transform_size);
		}
	}

	return nnp_status_success;
}

static nnp_status compute_gemm_convolution_inference(
	const size_t input_channels,
	const size_t output_channels,
	const nnp_size input_size,
	const nnp_padding input_padding,
	const nnp_size kernel_size,
	const nnp_size output_size,
	const nnp_size output_subsampling,
	const float* input,
	const float* kernel,
	const float* bias,
	float* output,
	nnp_workspace_pointers* workspace_buffer,
	const nnp_activation activation,
	nnp_profile* profile)
{
	nnp_status status = nnp_status_success;
	const size_t simd_width = nnp_hwinfo.simd_width;

	/* Calculate cache blocking parameters */
	const size_t cache_elements_l1 = nnp_hwinfo.blocking.l1 / sizeof(float);
	const size_t cache_elements_l2 = nnp_hwinfo.blocking.l2 / sizeof(float);
	const size_t cache_elements_l3 = nnp_hwinfo.blocking.l3 / sizeof(float);

	const size_t output_channels_subblock_max = nnp_hwinfo.sgemm.mr;
	const size_t output_image_subblock_max = nnp_hwinfo.sgemm.nr;

	const size_t reduction_size = input_channels * kernel_size.height * kernel_size.width;
	const size_t output_image_size = output_size.height * output_size.width;
	const size_t reduction_block_max = round_down(cache_elements_l1 / (output_channels_subblock_max + output_image_subblock_max), 2ull);
	const size_t output_channels_block_max = round_down(cache_elements_l2 / reduction_block_max, output_channels_subblock_max);
	const size_t output_image_block_max = round_down(cache_elements_l3 / reduction_block_max, output_image_subblock_max);

	const size_t packed_kernel_size = output_channels *	min(reduction_block_max, reduction_size) * sizeof(float);
	const size_t packed_input_size = min(output_image_block_max, round_up(output_image_size, simd_width)) *	min(reduction_block_max, reduction_size) * sizeof(float);

	void* memory_packed_input = allocate_memory(packed_input_size);
	void* memory_packed_kernel = allocate_memory(packed_kernel_size);
		
	if (memory_packed_kernel == NULL || memory_packed_input == NULL)
		return nnp_status_out_of_memory;

	float* packed_input = static_cast<float*>(memory_packed_input);
	float* packed_kernel = static_cast<float*>(memory_packed_kernel);

	for (size_t reduction_block_start = 0ull; reduction_block_start < reduction_size; reduction_block_start += reduction_block_max) 
	{
		const size_t reduction_block_size = min(reduction_size - reduction_block_start, reduction_block_max);

		/* Pack kernel into memory block */
		NNP_KERNEL_TRANSFORM_START(profile)
		kernel_packing_context kernel_packing_context = 
		{
			kernel + reduction_block_start,
			packed_kernel,
			reduction_size,
			reduction_block_start,
			reduction_block_size
		};
		pthreadpool_compute_2d_tiled(
			(pthreadpool_function_2d_tiled_t)compute_kernel_packing,
			&kernel_packing_context,
			output_channels,
			reduction_block_size,
			output_channels_subblock_max,
			1ull);
		NNP_KERNEL_TRANSFORM_END(profile)

		const fxdiv_divisor_size_t kernel_elements_divisor = fxdiv_init_size_t(kernel_size.height * kernel_size.width);
		const fxdiv_divisor_size_t kernel_width_divisor = fxdiv_init_size_t(kernel_size.width);
		const fxdiv_divisor_size_t output_width_divisor = fxdiv_init_size_t(output_size.width);
		for (size_t output_image_block_start = 0ull; output_image_block_start < output_image_size; output_image_block_start += output_image_block_max) 
		{
			const size_t output_image_block_size = min(output_image_size - output_image_block_start, output_image_block_max);

			/* Pack image into L3 block */
			NNP_INPUT_TRANSFORM_START(profile)
			input_packing_context input_packing_context = 
			{
				input,
				packed_input,
				simd_width,
				reduction_block_start,
				reduction_block_size,
				output_image_block_start,
				input_size,
				input_padding.top,
				input_padding.left,
				kernel_elements_divisor,
				kernel_width_divisor,
				output_width_divisor,
				output_subsampling
			};
			pthreadpool_compute_2d_tiled(
				(pthreadpool_function_2d_tiled_t)compute_input_packing,
				&input_packing_context,
				reduction_block_size,
				output_image_block_size,
				1ull,
				output_image_subblock_max);
			NNP_INPUT_TRANSFORM_END(profile)

			NNP_BLOCK_MULTIPLICATION_START(profile)
			matrix_multiplication_context matrix_multiplication_context = 
			{
				packed_kernel,
				packed_input,
				output,
				reduction_block_start,
				reduction_block_size,
				output_image_size,
				output_image_block_start,
				output_image_subblock_max,
				output_channels_subblock_max
			};
			pthreadpool_compute_2d_tiled(
				(pthreadpool_function_2d_tiled_t)compute_matrix_multiplication,
				&matrix_multiplication_context,
				output_channels,
				output_image_block_size,
				output_channels_block_max,
				output_image_subblock_max);
			NNP_BLOCK_MULTIPLICATION_END(profile)
		}
	}
	
	/* Add bias */
	NNP_OUTPUT_TRANSFORM_START(profile)
	switch (activation) 
	{
	case nnp_activation_identity:
		for (size_t output_channel = 0ull; output_channel < output_channels; output_channel++)
		{
			const float bias_value = bias[output_channel];
			for (size_t index = 0ull; index < output_image_size; index++)
				output[output_channel * output_image_size + index] += bias_value;
		}
		break;

	case nnp_activation_relu:
		for (size_t output_channel = 0ull; output_channel < output_channels; output_channel++)
		{
			const float bias_value = bias[output_channel];
			for (size_t index = 0ull; index < output_image_size; index++)
				output[output_channel * output_image_size + index] = relu(output[output_channel * output_image_size + index] + bias_value, 0.0f);
		}
		break;
	}
	NNP_OUTPUT_TRANSFORM_END(profile)

	release_memory(packed_input, packed_input_size);
	release_memory(packed_kernel, packed_kernel_size);
	
	return status;
}

static nnp_status compute_direct_convolution_inference(
	const size_t input_channels,
	const size_t output_channels,
	const nnp_size image_size,
	const nnp_size kernel_size,
	const float* input,
	const float* kernel,
	const float* bias,
	float* output,
	nnp_workspace_pointers* workspace_buffer,
	const nnp_activation activation,
	nnp_profile* profile)
{
	const size_t image_elements = image_size.height * image_size.width;

	NNP_BLOCK_MULTIPLICATION_START(profile)
	direct_convolution_context direct_convolution_context = 
	{
		input,
		kernel,
		output,
		image_elements,
		input_channels,
		nnp_hwinfo.conv1x1.mr,
		nnp_hwinfo.conv1x1.nr,
		nnp_hwinfo.conv1x1.only_mr_x_nr,
		nnp_hwinfo.conv1x1.upto_mr_x_nr
	};
	pthreadpool_compute_1d_tiled(
		(pthreadpool_function_1d_tiled_t)compute_direct_convolution,
		&direct_convolution_context,
		output_channels,
		nnp_hwinfo.conv1x1.nr);
	NNP_BLOCK_MULTIPLICATION_END(profile)

	/* Add bias */
	NNP_OUTPUT_TRANSFORM_START(profile)
	switch (activation) 
	{
	case nnp_activation_identity:
		for (size_t output_channel = 0ull; output_channel < output_channels; output_channel++)
		{
			const float bias_value = bias[output_channel];
			for (size_t index = 0ull; index < image_elements; index++)
				output[output_channel * image_elements + index] += bias_value;
		}
		break;

	case nnp_activation_relu:
		for (size_t output_channel = 0ull; output_channel < output_channels; output_channel++) 
		{
			const float bias_value = bias[output_channel];
			for (size_t index = 0ull; index < image_elements; index ++) 
				output[output_channel * image_elements + index] = relu(output[output_channel * image_elements + index] + bias_value, 0.0f);
		}
		break;
	}
	NNP_OUTPUT_TRANSFORM_END(profile)

	return nnp_status_success;
}

nnp_status nnp_convolution_inference(
	nnp_convolution_algorithm algorithm,
	const nnp_convolution_transform_strategy transform_strategy,
	const size_t input_channels,
	const size_t output_channels,
	const nnp_size input_size,
	const nnp_padding input_padding,
	const nnp_size kernel_size,
	const nnp_size output_subsampling,
	const float* input,
	const float* kernel,
	const float* bias,
	float* output,
	nnp_workspace_pointers* workspace_buffer,
	const nnp_activation activation,
	const void* activation_parameters,
	nnp_profile* profile)
{
	const nnp_size output_size = 
	{ 
		(input_padding.left + input_size.width + input_padding.right - kernel_size.width) / output_subsampling.width + 1ull, 
		(input_padding.top + input_size.height + input_padding.bottom - kernel_size.height) / output_subsampling.height + 1ull 
	};

	if (activation_parameters != NULL)
		return nnp_status_unsupported_activation_parameters;

	if (algorithm == nnp_convolution_algorithm_auto) 
	{
		if ((max(kernel_size.width, kernel_size.height) > 16ull) || (max(output_subsampling.width, output_subsampling.height) > 1ull))
			algorithm = nnp_convolution_algorithm_implicit_gemm;
		else if (max(kernel_size.width, kernel_size.height) > 8ull) 
			algorithm = nnp_convolution_algorithm_ft16x16;
		else if (max(kernel_size.width, kernel_size.height) == 1ull) 
			algorithm = nnp_convolution_algorithm_direct;
		else 
		{
			const size_t tile_count_8x8 = divide_round_up(output_size.height, 8ull - kernel_size.height + 1ull) * divide_round_up(output_size.width, 8ull - kernel_size.width + 1ull);
			const size_t tile_count_16x16 = divide_round_up(output_size.height, 16ull - kernel_size.height + 1ull) * divide_round_up(output_size.width, 16ull - kernel_size.width + 1ull);
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

	const size_t transform_element_size = sizeof(float);
	nnp_status status = nnp_status_success;
	nnp_size tile_size = nnp_size{ 8ull, 8ull };
	bool fourier_transform = false;
	nnp_transform_2d_with_offset input_transform_function = NULL;
	nnp_transform_2d_with_offset kernel_transform_function = NULL;
	nnp_transform_2d_with_bias output_transform_function = NULL;

	switch (algorithm) 
	{
		case nnp_convolution_algorithm_wt8x8:
		{
			input_transform_function = nnp_hwinfo.transforms.iwt_f6x6_3x3_with_offset_and_stream;
			kernel_transform_function = nnp_hwinfo.transforms.kwt_f6x6_3x3;
			if (activation == nnp_activation_relu)
				output_transform_function = nnp_hwinfo.transforms.owt_f6x6_3x3_with_bias_with_relu;
			else
				output_transform_function = nnp_hwinfo.transforms.owt_f6x6_3x3_with_bias;
		}
		break;

		case nnp_convolution_algorithm_ft8x8:
		{
			input_transform_function = nnp_hwinfo.transforms.fft8x8_with_offset_and_stream;
			kernel_transform_function = nnp_hwinfo.transforms.fft8x8_with_offset_and_stream;
			fourier_transform = true;
			if (activation == nnp_activation_relu)
				output_transform_function = nnp_hwinfo.transforms.ifft8x8_with_bias_with_relu;
			else
				output_transform_function = nnp_hwinfo.transforms.ifft8x8_with_bias;
		}
		break;

		case nnp_convolution_algorithm_ft16x16:
		{
			tile_size = nnp_size{ 16ull, 16ull };
			input_transform_function = nnp_hwinfo.transforms.fft16x16_with_offset_and_stream;
			kernel_transform_function = nnp_hwinfo.transforms.fft16x16_with_offset_and_stream;
			fourier_transform = true;
			if (activation == nnp_activation_relu)
				output_transform_function = nnp_hwinfo.transforms.ifft16x16_with_bias_with_relu;
			else
				output_transform_function = nnp_hwinfo.transforms.ifft16x16_with_bias;
		}
		break;
		
		case nnp_convolution_algorithm_implicit_gemm:
		case nnp_convolution_algorithm_direct:
			break;

		default:
			return nnp_status_invalid_algorithm;
			break;
	}

	switch (algorithm) 
	{
		case nnp_convolution_algorithm_wt8x8:
		case nnp_convolution_algorithm_ft8x8:
		case nnp_convolution_algorithm_ft16x16:
		{
			if (transform_strategy != nnp_convolution_transform_strategy_compute)
				return nnp_status_unsupported_transform_strategy;
			if (max(output_subsampling.height, output_subsampling.width) != 1ull)
				return nnp_status_unsupported_algorithm;
			if (kernel_size.height > tile_size.height || kernel_size.width > tile_size.width)
				return nnp_status_unsupported_algorithm;

			status = compute_fast_convolution_inference(
				fourier_transform, transform_strategy, transform_element_size,
				input_channels, output_channels,
				tile_size, input_size, input_padding, kernel_size, output_size,
				input, kernel, bias, output, workspace_buffer,
				input_transform_function, kernel_transform_function, output_transform_function, profile);
		}
		break;

		case nnp_convolution_algorithm_implicit_gemm:
		{
			if (transform_strategy != nnp_convolution_transform_strategy_compute)
				return nnp_status_unsupported_transform_strategy;

			status = compute_gemm_convolution_inference(
				input_channels, output_channels,
				input_size, input_padding, kernel_size, output_size, output_subsampling,
				input, kernel, bias, output, workspace_buffer, activation, profile);
		}
		break;

		case nnp_convolution_algorithm_direct:
		{
			if (transform_strategy != nnp_convolution_transform_strategy_compute)
				return nnp_status_unsupported_transform_strategy;

			if (max(output_subsampling.height, output_subsampling.width) != 1ull)
				return nnp_status_unsupported_algorithm;

			if (max(kernel_size.height, kernel_size.width) != 1ull)
				return nnp_status_unsupported_algorithm;

			status = compute_direct_convolution_inference(
				input_channels, output_channels, input_size, kernel_size,
				input, kernel, bias, output, workspace_buffer, activation, profile);
		}
		break;
	}

	return status;
}
