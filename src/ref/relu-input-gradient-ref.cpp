#include <nnpack.h>
#include <reference.h>
#include <activations.h>

struct relu_input_gradient_context 
{
	size_t channels;
	const float* grad_output;
	const float* input;
	float* grad_input;
	float negative_slope;
};

static void compute_relu_input_gradient(
	const struct relu_input_gradient_context* context,
	const size_t sample)
{
	const size_t channels    = context->channels;
	const float* grad_output = context->grad_output + sample * channels;
	const float* input       = context->input       + sample * channels;
	float* grad_input        = context->grad_input  + sample * channels;
	float negative_slope     = context->negative_slope;

	for (size_t channel = 0ull; channel < channels; channel++)
		grad_input[channel] = grad_relu(grad_output[channel], input[channel], negative_slope);
}

void nnp_relu_input_gradient__reference(
	const size_t batch_size,
	const size_t channels,
	const float* grad_output,
	const float* input,
	float* grad_input,
	const float negative_slope)
{
	struct relu_input_gradient_context relu_input_gradient_context = 
	{
		channels,
		grad_output,
		input,
		grad_input,
		negative_slope
	};

	pthreadpool_compute_1d(
		(pthreadpool_function_1d_t)compute_relu_input_gradient,
		&relu_input_gradient_context,
		batch_size);
}
