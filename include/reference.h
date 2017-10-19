#pragma once

#include <stddef.h>

#include "pthreadpool.h"

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

void nnp_fft2_aos__ref(const float* t, size_t t_stride, float* f, size_t f_stride);
void nnp_fft4_aos__ref(const float* t, size_t t_stride, float* f, size_t f_stride);
void nnp_fft8_aos__ref(const float* t, size_t t_stride, float* f, size_t f_stride);
void nnp_fft16_aos__ref(const float* t, size_t t_stride, float* f, size_t f_stride);
void nnp_fft32_aos__ref(const float* t, size_t t_stride, float* f, size_t f_stride);
void nnp_fft2_soa__ref(const float* t, size_t t_stride, float* f, size_t f_stride);
void nnp_fft4_soa__ref(const float* t, size_t t_stride, float* f, size_t f_stride);
void nnp_fft8_soa__ref(const float* t, size_t t_stride, float* f, size_t f_stride);
void nnp_fft16_soa__ref(const float* t, size_t t_stride, float* f, size_t f_stride);
void nnp_fft32_soa__ref(const float* t, size_t t_stride, float* f, size_t f_stride);
void nnp_ifft2_aos__ref(const float* f, size_t f_stride, float* t, size_t t_stride);
void nnp_ifft4_aos__ref(const float* f, size_t f_stride, float* t, size_t t_stride);
void nnp_ifft8_aos__ref(const float* f, size_t f_stride, float* t, size_t t_stride);
void nnp_ifft16_aos__ref(const float* f, size_t f_stride, float* t, size_t t_stride);
void nnp_ifft32_aos__ref(const float* f, size_t f_stride, float* t, size_t t_stride);
void nnp_ifft2_soa__ref(const float* f, size_t f_stride, float* t, size_t t_stride);
void nnp_ifft4_soa__ref(const float* f, size_t f_stride, float* t, size_t t_stride);
void nnp_ifft8_soa__ref(const float* f, size_t f_stride, float* t, size_t t_stride);
void nnp_ifft16_soa__ref(const float* f, size_t f_stride, float* t, size_t t_stride);
void nnp_ifft32_soa__ref(const float* f, size_t f_stride, float* t, size_t t_stride);
void nnp_fft8_real__ref(const float* t, size_t f_stride, float* f, size_t t_stride);
void nnp_fft16_real__ref(const float* t, size_t f_stride, float* f, size_t t_stride);
void nnp_fft32_real__ref(const float* t, size_t f_stride, float* f, size_t t_stride);
void nnp_ifft8_real__ref(const float* t, size_t f_stride, float* f, size_t t_stride);
void nnp_ifft16_real__ref(const float* t, size_t f_stride, float* f, size_t t_stride);
void nnp_ifft32_real__ref(const float* t, size_t f_stride, float* f, size_t t_stride);
void nnp_fft8_dualreal__ref(const float* t, float* f);
void nnp_fft16_dualreal__ref(const float* t, float* f);
void nnp_fft32_dualreal__ref(const float* t, float* f);
void nnp_ifft8_dualreal__ref(const float* f, float* t);
void nnp_ifft16_dualreal__ref(const float* f, float* t);
void nnp_ifft32_dualreal__ref(const float* f, float* t);

#ifdef __cplusplus
} /* extern "C" */
#endif
