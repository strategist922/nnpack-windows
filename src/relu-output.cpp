#include <stdint.h>
#include <stddef.h>
#include <assert.h>

#include <nnpack.h>
#include <utils.h>
#include <hwinfo.h>
#include <activations.h>
#include <validation.h>


struct __declspec(align(64)) relu_context
{
	nnp_relu_function relu_function;
	const float* input;
	float* output;
	float negative_slope;
};

static void compute_relu_output(
	const struct relu_context* context,
	const size_t block_start, const size_t block_size)
{
	nnp_relu_function relu = context->relu_function;
	const float* input     = context->input;
	float* output          = context->output;
	float negative_slope   = context->negative_slope;

	relu(input + block_start, output + block_start, block_size, negative_slope);
}

struct __declspec(align(64)) inplace_relu_context 
{
	nnp_inplace_relu_function relu_function;
	float* data;
	float negative_slope;
};

static void compute_inplace_relu_output(
	const struct inplace_relu_context* context,
	const size_t block_start, const size_t block_size)
{
	nnp_inplace_relu_function relu = context->relu_function;
	float* data                    = context->data;
	float negative_slope           = context->negative_slope;

	relu(data + block_start, block_size, negative_slope);
}

enum nnp_status nnp_relu_output(
	const size_t batch_size,
	const size_t channels,
	const float* input,
	float* output,
	float negative_slope)
{
	enum nnp_status status = validate_relu_arguments(batch_size, channels);
	if (status != nnp_status_success) 
		return status;
	
	size_t elements = batch_size * channels;
	const size_t simd_width = nnp_hwinfo.simd_width;

	assert(((uintptr_t) input) % sizeof(float) == 0);
	assert(((uintptr_t) output) % sizeof(float) == 0);

	const size_t prologue_elements = min((size_t)(-(long long)(((uintptr_t) output) / sizeof(float)) % simd_width), elements);
	for (size_t i = 0ull; i < prologue_elements; i++) 
		output[i] = relu(input[i], negative_slope);
	
	elements -= prologue_elements;
	input += prologue_elements;
	output += prologue_elements;

	const size_t epilogue_elements = elements % simd_width;
	for (size_t i = 0ull; i < epilogue_elements; i++) {
		output[elements - epilogue_elements + i] =
			relu(input[elements - epilogue_elements + i], negative_slope);
	}
	elements -= epilogue_elements;

	if (input != output) 
	{
		/* Out-of-place transformation */
		struct relu_context relu_context = 
		{
			nnp_hwinfo.activations.relu,
			input,
			output,
			negative_slope
		};
		pthreadpool_compute_1d_tiled(
			(pthreadpool_function_1d_tiled_t) compute_relu_output,
			&relu_context,
			elements, round_down(nnp_hwinfo.blocking.l1 / sizeof(float), simd_width));
	} 
	else 
	{
		/* In-place transformation */
		struct inplace_relu_context inplace_relu_context = 
		{
			nnp_hwinfo.activations.inplace_relu,
			output,
			negative_slope
		};
		pthreadpool_compute_1d_tiled(
			(pthreadpool_function_1d_tiled_t) compute_inplace_relu_output,
			&inplace_relu_context,
			elements, round_down(nnp_hwinfo.blocking.l1 / sizeof(float), simd_width));
	}

	return nnp_status_success;
}
