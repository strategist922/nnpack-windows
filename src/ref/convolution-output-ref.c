#include <nnpack.h>
#include <nnpack/reference.h>

struct convolution_output_context 
{
	const size_t input_channels;
	const size_t output_channels;
	const struct nnp_size input_size;
	const struct nnp_padding input_padding;
	const struct nnp_size kernel_size;
	const struct nnp_size output_size;
	const struct nnp_size output_subsampling;
	const float* input_pointer;
	const float* kernel_pointer;
	const float* bias;
	float* output_pointer;
};

static void compute_convolution_output(
	const struct convolution_output_context* context,
	const size_t sample, 
	const size_t output_channel)
{
	const size_t input_channels              = context->input_channels;
	const size_t output_channels             = context->output_channels;
	const struct nnp_size input_size         = context->input_size;
	const struct nnp_padding input_padding   = context->input_padding;
	const struct nnp_size kernel_size        = context->kernel_size;
	const struct nnp_size output_size        = context->output_size;
	const struct nnp_size output_subsampling = context->output_subsampling;

	const float* input = context->input_pointer;
	const float* kernel = context->kernel_pointer;
	float* output = context->output_pointer;

	for (size_t y = 0; y < output_size.height; y++) 
		for (size_t x = 0; x < output_size.width; x++) 
		{
			double v = 0.0;
			for (size_t input_channel = 0; input_channel < input_channels; input_channel++) 
				for (size_t i = 0; i < kernel_size.height; i++) 
				{
					const size_t s = y * output_subsampling.height + i - input_padding.top;
					if (s < input_size.height) 
						for (size_t j = 0; j < kernel_size.width; j++) 
						{
							const size_t t = x * output_subsampling.width + j - input_padding.left;
							if (t < input_size.width) 
								v += input[(sample * input_channels * input_size.width * input_size.height) + (input_channel * input_size.width * input_size.height) + (s * input_size.width) + t] * kernel[(output_channel * input_channels * kernel_size.width * kernel_size.height) + (input_channel * kernel_size.width * kernel_size.height) + (i * kernel_size.width) + j];
						}
				}

			output[(sample * output_channels * output_size.width * output_size.height) + (output_channel * output_size.width * output_size.height) + (y  * output_size.width) + x] = (float)(v + context->bias[output_channel]);
		}
}

void nnp_convolution_output__reference(
	const size_t batch_size,
	const size_t input_channels,
	const size_t output_channels,
	const struct nnp_size input_size,
	const struct nnp_padding input_padding,
	const struct nnp_size kernel_size,
	const struct nnp_size output_subsampling,
	const float* input_pointer,
	const float* kernel_pointer,
	const float* bias,
	float* output_pointer)
{
	const struct nnp_size output_size =
	{
		.width = (input_padding.left + input_size.width + input_padding.right - kernel_size.width) / output_subsampling.width + 1,
		.height = (input_padding.top + input_size.height + input_padding.bottom - kernel_size.height) / output_subsampling.height + 1
	};

	struct convolution_output_context convolution_output_context =
	{
		.input_channels = input_channels,
		.output_channels = output_channels,
		.input_size = input_size,
		.input_padding = input_padding,
		.kernel_size = kernel_size,
		.output_size = output_size,
		.output_subsampling = output_subsampling,
		.input_pointer = input_pointer,
		.kernel_pointer = kernel_pointer,
		.bias = bias,
		.output_pointer = output_pointer
	};
	pthreadpool_compute_2d(
		(pthreadpool_function_2d_t)compute_convolution_output,
		&convolution_output_context,
		batch_size, output_channels);
}