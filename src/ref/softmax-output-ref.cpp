#include "../../include/nnpack.h"
#include "../../include/reference.h"
#include "../../include/utils.h"

#include <float.h>
#include <math.h>

static inline float vector_maxf(const size_t length, const float* array) 
{
    float max_element = -FLT_MAX;
    for (size_t i = 0ull; i < length; i++) 
	     max_element = maxf(max_element, array[i]);
    
    return max_element;
}

static inline float vector_sum_expf_minus_c(const size_t length, const float* array, float c) 
{
    float sum = 0.0f;
    for (size_t i = 0ull; i < length; i++)
        sum += expf(array[i] - c);
    
    return sum;
}

struct softmax_output_context 
{
    const size_t channels;
    const float* input;
    float* output;
};

static void compute_softmax_output(
    const struct softmax_output_context* context,
    const size_t sample)
{
    const size_t channels = context->channels;
	const float* input = context->input;
	float* output = context->output;

    const float max_element = vector_maxf(channels, &input[sample * channels]);
    const float sum_exp = vector_sum_expf_minus_c(channels, &input[sample * channels], max_element);
    const float norm_factor = 1.0f / sum_exp;
    for (size_t channel = 0ull; channel < channels; channel++)
        output[sample * channels + channel] = norm_factor * expf(input[sample * channels + channel] - max_element);
}

void nnp_softmax_output__reference(
    const size_t batch_size,
    const size_t channels,
    const float* input,
    float* output)
{
    struct softmax_output_context softmax_output_context = 
	{
        channels,
        input,
        output
    };
    
	pthreadpool_compute_1d(
        (pthreadpool_function_1d_t)compute_softmax_output,
        &softmax_output_context,
        batch_size);
}
