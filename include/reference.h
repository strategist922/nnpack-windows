#pragma once

#ifdef __cplusplus
extern "C" {
#endif

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
	float* output_pointer);

void nnp_convolution_input_gradient__reference(
	const size_t batch_size,
	const size_t input_channels,
	const size_t output_channels,
	const struct nnp_size input_size,
	const struct nnp_padding input_padding,
	const struct nnp_size kernel_size,
	const float* grad_output,
	const float* kernel,
	float* grad_input);

void nnp_convolution_kernel_gradient__reference(
	const size_t batch_size,
	const size_t input_channels,
	const size_t output_channels,
	const struct nnp_size input_size,
	const struct nnp_padding input_padding,
	const struct nnp_size kernel_size,
	const float* input,
	const float* grad_output,
	float* grad_kernel);

void nnp_fully_connected_output_f32__reference(
	const size_t batch_size,
	const size_t input_channels,
	const size_t output_channels,
	const float* input,
	const float* kernel,
	float* output);

void nnp_max_pooling_output__reference(
	const size_t batch_size,
	const size_t channels,
	const struct nnp_size input_size,
	const struct nnp_padding input_padding,
	const struct nnp_size pooling_size,
	const struct nnp_size pooling_stride,
	const float* input,
	float* output);

void nnp_relu_output__reference(
	const size_t batch_size,
	const size_t channels,
	const float* input,
	float* output,
	const float negative_slope);

void nnp_relu_input_gradient__reference(
	const size_t batch_size,
	const size_t channels,
	const float* grad_output,
	const float* input,
	float* grad_input,
	const float negative_slope);

void nnp_softmax_output__reference(
	const size_t batch_size,
	const size_t channels,
	const float* input,
	float* output);

#ifdef __cplusplus
} /* extern "C" */
#endif
