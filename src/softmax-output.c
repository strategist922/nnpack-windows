#include <nnpack.h>
#include <utils.h>
#include <hwinfo.h>
#include <softmax.h>
#include <validation.h>
#include <macros.h>


struct NNP_CACHE_ALIGN softmax_context 
{
	const nnp_softmax_function softmax_function;
	const size_t channels;
	const float* input;
	float* output;
};

static void compute_softmax_output(const struct softmax_context* context, const size_t sample)
{
	const nnp_softmax_function softmax = context->softmax_function;
	const size_t channels              = context->channels;

	const float* input = context->input;
	float* output = context->output;

	softmax(channels, input + sample * channels, output + sample * channels);
}

struct NNP_CACHE_ALIGN inplace_softmax_context 
{
	const nnp_inplace_softmax_function softmax_function;
	const size_t channels;
	float* data;
};

static void compute_inplace_softmax_output(
	const struct inplace_softmax_context* context,
	const size_t sample)
{
	const nnp_inplace_softmax_function softmax = context->softmax_function;
	const size_t channels                      = context->channels;

	float* data = context->data;

	softmax(channels, data + sample * channels);
}

enum nnp_status nnp_softmax_output(
	const size_t batch_size,
	const size_t channels,
	const float* input,
	float* output)
{
	enum nnp_status status = validate_softmax_arguments(batch_size, channels);
	if (status != nnp_status_success)
		return status;
	
	if (input != output) 
	{
		/* Out-of-place softmax */
		struct softmax_context softmax_context = 
		{
			nnp_hwinfo.activations.softmax,
			channels,
			input,
			output
		};
		pthreadpool_compute_1d(
			(pthreadpool_function_1d_t)compute_softmax_output,
			&softmax_context,
			batch_size);
	} 
	else 
	{
		/* In-place softmax */
		struct inplace_softmax_context inplace_softmax_context = 
		{
			nnp_hwinfo.activations.inplace_softmax,
			channels,
			output
		};
		pthreadpool_compute_1d(
			(pthreadpool_function_1d_t)compute_inplace_softmax_output,
			&inplace_softmax_context,
			batch_size);
	}

	return nnp_status_success;
}
