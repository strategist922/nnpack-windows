#include <math.h>

#include <nnpack.h>
#include <macros.h>
#include <utils.h>
#include <pooling.h>
#include <validation.h>
#include <macros.h>

struct NNP_CACHE_ALIGN pooling_context 
{
	nnp_pooling_function pooling_function;
	const float* input_pointer;
	float* output_pointer;
	size_t channels;
	struct nnp_size input_size;
	struct nnp_padding input_padding;
	struct nnp_size output_size;
	struct nnp_size pooling_size;
	struct nnp_size pooling_stride;
};

static void compute_max_pooling_forward__generic(
	const float* input_pointer,
	float* output_pointer,
	const size_t input_height,
	const size_t input_width,
	const size_t padding_top,
	const size_t padding_left,
	const size_t output_height,
	const size_t output_width,
	const uint32_t stride_height,
	const uint32_t stride_width,
	const uint32_t pooling_height,
	const uint32_t pooling_width)
{
	const float* input = input_pointer;
	float* output = output_pointer;

	for (size_t y = 0; y < output_height; y++) 
		for (size_t x = 0; x < output_width; x++) 
		{
			float v = -INFINITY;
			for (size_t i = 0; i < pooling_height; i++) 
			{
				const size_t s = y * stride_height + i - padding_top;
				if (s < input_height) 
					for (size_t j = 0; j < pooling_width; j++) 
					{
						const size_t t = x * stride_width + j - padding_left;
						if (t < input_width) 
							v = maxf(input[s * input_width + t], v);
					}
			}
			output[y * output_width + x] = v;
		}
}


static void compute_max_pooling_forward_2x2_2x2__avx2(
	const float* input_pointer,
	float* output_pointer,
	const size_t input_height,
	const size_t input_width,
	const size_t padding_top,
	const size_t padding_left,
	const size_t output_height,
	const size_t output_width,
	const uint32_t stride_height,
	const uint32_t stride_width,
	const uint32_t pooling_height,
	const uint32_t pooling_width)
{
	const struct nnp_size input_tile = { 16 , 2 };
	const struct nnp_size output_tile = { 8, 1 };

	const float* input = input_pointer;
	float* output = output_pointer;

	for (size_t y = 0; y < output_height; y += output_tile.height) 
	{
		const size_t input_y = min(doz(y * stride_height, padding_top), input_height);
		const size_t input_row_offset = doz(padding_top, y);
		const size_t input_row_count = min(input_tile.height, doz(input_height, input_y));
		const size_t output_row_count = min(output_tile.height, output_height - y);
		for (size_t x = 0; x < output_width; x += output_tile.width) 
		{
			const size_t input_x = min(doz(x * stride_width, padding_left), input_width);
			const size_t input_column_offset = doz(padding_left, x);
			const size_t input_column_count = min(input_tile.width, doz(input_width, input_x));
			const size_t output_column_count = min(output_tile.width, output_width - x);
			nnp_maxpool_2x2_2x2__avx2(
				input + input_y * input_width + input_x,
				output + y * output_width + x,
				input_width,
				input_row_offset,
				input_row_count,
				input_column_offset,
				input_column_count,
				output_column_count);
		}
	}
}


static void compute_pooling_output(
	const struct pooling_context* context,
	size_t sample,
	size_t channel)
{
	const size_t channels                       = context->channels;
	const struct nnp_size input_size            = context->input_size;
	const struct nnp_padding input_padding      = context->input_padding;
	const struct nnp_size output_size           = context->output_size;
	const struct nnp_size pooling_stride        = context->pooling_stride;
	const struct nnp_size pooling_size          = context->pooling_size;
	const nnp_pooling_function pooling_function = context->pooling_function;

	const float* input = context->input_pointer;
	float* output = context->output_pointer;

	pooling_function(
		input + (sample * channels * input_size.height * input_size.width) + (channel * input_size.height * input_size.width),
		output + (sample * channels * output_size.height * output_size.width) + (channel * output_size.height * output_size.width),
		input_size.height, input_size.width,
		input_padding.top, input_padding.left,
		output_size.height, output_size.width,
		pooling_stride.height, pooling_stride.width,
		pooling_size.height, pooling_size.width);
}

enum nnp_status nnp_max_pooling_output(
	const size_t batch_size,
	const size_t channels,
	const struct nnp_size input_size,
	const struct nnp_padding input_padding,
	const struct nnp_size pooling_size,
	const struct nnp_size pooling_stride,
	const float* input,
	float* output)
{
	
	const struct nnp_size output_size = 
	{ 
		divide_round_up(doz(input_padding.left + input_size.width + input_padding.right, pooling_size.width), pooling_stride.width) + 1, 
		divide_round_up(doz(input_padding.top + input_size.height + input_padding.bottom, pooling_size.height), pooling_stride.height) + 1 
	};

	struct pooling_context pooling_context = 
	{
		compute_max_pooling_forward__generic,
		input,
		output,
		channels,
		input_size,
		input_padding,
		output_size,
		pooling_size,
		pooling_stride
	};
	
	if ((pooling_stride.height == 2) && (pooling_stride.width == 2) && (pooling_size.height == 2) && (pooling_size.width == 2)) 
	    pooling_context.pooling_function = compute_max_pooling_forward_2x2_2x2__avx2;
	
	pthreadpool_compute_2d((pthreadpool_function_2d_t)compute_pooling_output,
		&pooling_context,
		batch_size, channels);

	return nnp_status_success;
}
