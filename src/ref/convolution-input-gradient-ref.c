#include <nnpack.h>
#include <nnpack/reference.h>


struct convolution_input_gradient_context 
{
	const size_t input_channels;
	const size_t output_channels;
	const struct nnp_size input_size;
	const struct nnp_padding input_padding;
	const struct nnp_size kernel_size;
	const struct nnp_size output_size;
	const float* grad_output_pointer;
	const float* kernel_pointer;
	float* grad_input_pointer;
};

static void compute_convolution_input_gradient(
	const struct convolution_input_gradient_context* context,
	const size_t sample, 
	const size_t input_channel)
{
	const size_t input_channels            = context->input_channels;
	const size_t output_channels           = context->output_channels;
	const struct nnp_size input_size       = context->input_size;
	const struct nnp_padding input_padding = context->input_padding;
	const struct nnp_size kernel_size      = context->kernel_size;
	const struct nnp_size output_size      = context->output_size;

	const float* grad_output = context->grad_output_pointer;
	const float* kernel = context->kernel_pointer;

	float* grad_input = context->grad_input_pointer;

	for (size_t y = 0; y < input_size.height; y++) 
		for (size_t x = 0; x < input_size.width; x++) 
		{
			double v = 0.0;
			for (size_t output_channel = 0; output_channel < output_channels; output_channel++) 
				for (size_t i = 0; i < kernel_size.height; i++) 
				{
					const size_t s = y - i + input_padding.top;
					if (s < output_size.height) 
						for (size_t j = 0; j < kernel_size.width; j++) 
						{
							const size_t t = x - j + input_padding.left;
							if (t < output_size.width) 
								v += grad_output[(sample * output_channels * output_size.width * output_size.height) + (output_channel * output_size.width * output_size.height) + (s * output_size.width) + t] * kernel[(output_channel * input_channels * kernel_size.width * kernel_size.height) + (input_channel * kernel_size.width * kernel_size.height) + (i * kernel_size.width) +j];
						}
				}
			
			grad_input[(sample * input_channels * input_size.width * input_size.height) + (input_channel * input_size.width * input_size.height) + (y * input_size.width) + x] = (float)v;
		}
	
}

void nnp_convolution_input_gradient__reference(
	const size_t batch_size,
	const size_t input_channels,
	const size_t output_channels,
	const struct nnp_size input_size,
	const struct nnp_padding input_padding,
	const struct nnp_size kernel_size,
	const float* grad_output_pointer,
	const float* kernel_pointer,
	float* grad_input_pointer)
{
	const struct nnp_size output_size = 
	{ 
		.width = input_padding.left + input_size.width + input_padding.right - kernel_size.width + 1,
		.height = input_padding.top + input_size.height + input_padding.bottom - kernel_size.height + 1
	};

	struct convolution_input_gradient_context convolution_input_gradient_context = 
	{
		.input_channels = input_channels,
		.output_channels = output_channels,
		.input_size = input_size,
		.input_padding = input_padding,
		.kernel_size = kernel_size,
		.output_size = output_size,
		.grad_output_pointer = grad_output_pointer,
		.kernel_pointer = kernel_pointer,
		.grad_input_pointer = grad_input_pointer
	};

	pthreadpool_compute_2d(
		(pthreadpool_function_2d_t)compute_convolution_input_gradient,
		&convolution_input_gradient_context,
		batch_size, input_channels);
}
