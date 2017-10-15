#include "../../include/nnpack.h"
#include "../../include/reference.h"
#include "../../include/activations.h"

struct relu_output_context 
{
	const size_t channels;
	const float* input;
	float* output;
	const float negative_slope;
};

static void compute_relu_output(
	const struct relu_output_context* context,
	const size_t sample)
{
	const size_t channels = context->channels;
	const float* input    = context->input  + sample * channels;
	float* output         = context->output + sample * channels;
	const float negative_slope  = context->negative_slope;

	for (size_t channel = 0ull; channel < channels; channel++)
		output[channel] = relu(input[channel], negative_slope);
}

void nnp_relu_output__reference(
	const size_t batch_size,
	const size_t channels,
	const float* input,
	float* output,
	const float negative_slope)
{
	struct relu_output_context relu_output_context = 
	{
		channels,
		input,
		output,
		negative_slope
	};

	pthreadpool_compute_1d(
		(pthreadpool_function_1d_t)compute_relu_output,
		&relu_output_context,
		batch_size);
}
