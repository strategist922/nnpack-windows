#include "../../include/nnpack.h"
#include "../../include/reference.h"

struct fully_connected_output_context 
{
	const size_t input_channels;
	const size_t output_channels;
	const float* input_pointer;
	const float* kernel_pointer;
	float* output_pointer;
};

static void compute_fully_connected_output_f32(
	const struct fully_connected_output_context* context,
	const size_t sample, 
	const size_t output_channel)
{
	const size_t input_channels = context->input_channels;
	const size_t output_channels = context->output_channels;

	const float* input = context->input_pointer;
	const float* kernel = context->kernel_pointer;
	float* output = context->output_pointer;

	double v = 0.0;
	for (size_t input_channel = 0ull; input_channel < input_channels; input_channel++)
		v += double(input[sample * input_channels + input_channel]) * double(kernel[output_channel * output_channels + input_channel]);
	
	output[sample * output_channels + output_channel] = float(v);
}

void nnp_fully_connected_output_f32__reference(
	const size_t batch_size,
	const size_t input_channels,
	const size_t output_channels,
	const float* input,
	const float* kernel,
	float* output)
{
	struct fully_connected_output_context fully_connected_output_context = 
	{
		input_channels,
		output_channels,
		input,
		kernel,
		output
	};

	pthreadpool_compute_2d(
		(pthreadpool_function_2d_t)compute_fully_connected_output_f32,
		&fully_connected_output_context,
		batch_size, output_channels);
}
