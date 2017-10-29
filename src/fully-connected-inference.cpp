#include <stddef.h>

#include <nnpack.h>
#include <utils.h>
#include <hwinfo.h>
#include <system.h>
#include <validation.h>


struct NNP_CACHE_ALIGN fully_connected_inference_context 
{
	size_t input_channels;
	const float* input;
	const float* kernel;
	float* output;
};

static void compute_fully_connected_inference_f32(
	const fully_connected_inference_context* context,
	const size_t output_channels_subblock_start, 
	const size_t output_channels_subblock_size)
{
	const size_t input_channels      = context->input_channels;
	const float* input               = context->input;
	const float* kernel              = context->kernel;
	float* output                    = context->output;
	const nnp_sdotxf_function sdotxf = nnp_hwinfo.sdotxf.functions[output_channels_subblock_size - 1];

	sdotxf(input, &kernel[output_channels_subblock_start * input_channels],	input_channels, &output[output_channels_subblock_start], input_channels);
}

nnp_status nnp_fully_connected_inference(
	const size_t input_channels,
	const size_t output_channels,
	const float* input,
	const float* kernel,
	float* output)
{
	/* Basic validation of parameters. This check detects invalid, but not unsupported parameters. */
	nnp_status status = validate_fully_connected_arguments(1, input_channels, output_channels);
	if (status != nnp_status_success)
		return status;
	
	/* Do the computation */
	const size_t output_channels_subblock_max = nnp_hwinfo.sdotxf.fusion;
	fully_connected_inference_context fully_connected_inference_context = 
	{
		input_channels,
		input,
		kernel,
		output
	};
	pthreadpool_compute_1d_tiled(
		(pthreadpool_function_1d_tiled_t)compute_fully_connected_inference_f32,
		&fully_connected_inference_context,
		output_channels, output_channels_subblock_max);

	return nnp_status_success;
}