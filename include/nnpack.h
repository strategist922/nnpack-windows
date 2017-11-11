#pragma once

#if defined(__cplusplus)
#include <cstddef>
#include <cstdint>
#include <cstdbool>
#else
#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>
#endif

#include <pthreadpool.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
* @brief Status code for any NNPACK function call.
*/
enum nnp_status
{
	/** The call succeeded, and all output arguments now contain valid data. */
	nnp_status_success = 0,
	/** NNPACK function was called with batch_size == 0. */
	nnp_status_invalid_batch_size = 2,
	/** NNPACK function was called with channels == 0. */
	nnp_status_invalid_channels = 3,
	/** NNPACK function was called with input_channels == 0. */
	nnp_status_invalid_input_channels = 4,
	/** NNPACK function was called with output_channels == 0. */
	nnp_status_invalid_output_channels = 5,
	/** NNPACK function was called with input_size.height == 0 or input_size.width == 0 */
	nnp_status_invalid_input_size = 10,
	/** NNPACK function was called with input_stride.height == 0 or input_stride.width == 0 */
	nnp_status_invalid_input_stride = 11,
	/** NNPACK function was called with input_padding not less than respective kernel (or pooling) size, i.e.:
	*
	*  - input_padding.left   >= kernel_size.width  (>= pooling_size.width)
	*  - input_padding.right  >= kernel_size.width  (>= pooling_size.width)
	*  - input_padding.top    >= kernel_size.height (>= pooling_size.height)
	*  - input_padding.bottom >= kernel_size.height (>= pooling_size.height)
	*/
	nnp_status_invalid_input_padding = 12,
	/** NNPACK function was called with kernel_size.height == 0 or kernel_size.width == 0 */
	nnp_status_invalid_kernel_size = 13,
	/** NNPACK function was called with pooling_size.height == 0 or pooling_size.width == 0 */
	nnp_status_invalid_pooling_size = 14,
	/** NNPACK function was called with pooling_stride.height == 0 or pooling_stride.width == 0 */
	nnp_status_invalid_pooling_stride = 15,
	/** NNPACK function was called with convolution algorithm not in nnp_convolution_algorithm enumeration */
	nnp_status_invalid_algorithm = 16,
	/** NNPACK function was called with convolution transform strategy not in nnp_convolution_transform_strategy enum */
	nnp_status_invalid_transform_strategy = 17,
	/** NNPACK function was called with output_subsampling.height == 0 or output_subsampling.width == 0 */
	nnp_status_invalid_output_subsampling = 13,
	/** NNPACK function was called with activation not in nnp_activation enum */
	nnp_status_invalid_activation = 14,
	/** NNPACK function was called with invalid activation parameters */
	nnp_status_invalid_activation_parameters = 15,

	/** NNPACK does not support the particular input size for the function */
	nnp_status_unsupported_input_size = 20,
	/** NNPACK does not support the particular input stride for the function */
	nnp_status_unsupported_input_stride = 21,
	/** NNPACK does not support the particular input padding for the function */
	nnp_status_unsupported_input_padding = 22,
	/** NNPACK does not support the particular kernel size for the function */
	nnp_status_unsupported_kernel_size = 23,
	/** NNPACK does not support the particular pooling size for the function */
	nnp_status_unsupported_pooling_size = 24,
	/** NNPACK does not support the particular pooling stride for the function */
	nnp_status_unsupported_pooling_stride = 25,
	/** NNPACK does not support the particular convolution algorithm for the function */
	nnp_status_unsupported_algorithm = 26,
	/** NNPACK does not support the particular convolution transform strategy for the algorithm */
	nnp_status_unsupported_transform_strategy = 27,
	/** NNPACK does not support the particular activation function for the function */
	nnp_status_unsupported_activation = 28,
	/** NNPACK does not support the particular activation function parameters for the function */
	nnp_status_unsupported_activation_parameters = 29,

	/** NNPACK function was called before the library was initialized */
	nnp_status_uninitialized = 50,
	/** NNPACK does not implement this function for the host CPU */
	nnp_status_unsupported_hardware = 51,
	/** NNPACK failed to allocate memory for temporary buffers */
	nnp_status_out_of_memory = 52,
	/** Scratch space buffer is too small */
	nnp_status_insufficient_buffer = 53,
	/** Scratch space buffer is not properly aligned */
	nnp_status_misaligned_buffer = 54
};


/**
* @brief Activation applied applied after a convolutional or fully-connected layer.
*/
enum nnp_activation {
	/** Identity activation f(x) := x, i.e. no transformation */
	nnp_activation_identity = 0,
	/** ReLU activation f(x) := max(0, x) */
	nnp_activation_relu = 1,
};

/**
* @brief Algorithm for computing convolutional layers.
*/
enum nnp_convolution_algorithm {
	/** Let NNPACK choose the algorithm depending on layer parameters */
	nnp_convolution_algorithm_auto = 0,
	/** Tiled convolution based on 2D Fourier transform with 8x8 blocks. Supports kernels up to 8x8. */
	nnp_convolution_algorithm_ft8x8 = 1,
	/** Tiled convolution based on 2D Fourier transform with 16x16 blocks. Supports kernels up to 16x16. */
	nnp_convolution_algorithm_ft16x16 = 2,
	/** Tiled convolution based on 2D Winograd transform F(3x3, 6x6) with 8x8 blocks. Supports only 3x3 kernels. */
	nnp_convolution_algorithm_wt8x8 = 3,
	/** Direct convolution via implicit GEMM. */
	nnp_convolution_algorithm_implicit_gemm = 4,
	/** Direct convolution implementation. */
	nnp_convolution_algorithm_direct = 5,
	/**
	* Tiled convolution based on 2D Winograd transform F(3x3, 6x6) with 8x8 blocks in FP16.
	* Supports only 3x3 kernels. Implemented only for new ARM processors (with NEON-HP),
	* on non-supported processors falls back to nnp_convolution_algorithm_wt8x8.
	*/
	nnp_convolution_algorithm_wt8x8_fp16 = 6,
};

enum nnp_convolution_transform_strategy {
	nnp_convolution_transform_strategy_compute = 1,
	nnp_convolution_transform_strategy_precompute = 2,
	nnp_convolution_transform_strategy_reuse = 3
};

/* For backward compatibility */
#define nnp_convolution_transform_strategy_block_based nnp_convolution_transform_strategy_compute
#define nnp_convolution_transform_strategy_tuple_based nnp_convolution_transform_strategy_compute


/**
* @brief Size of images, kernels, and pooling filters in NNPACK.
*/
struct nnp_size {
	/** Width (horizontal size) of an image, kernel, or pooling filter. */
	size_t width;
	/** Height (vertical size) of an image, kernel, or pooling filter. */
	size_t height;
};

/**
* @brief Padding of images in NNPACK.
*/
struct nnp_padding {
	/** Padding above the image data */
	size_t top;
	/** Padding on the right of image data */
	size_t right;
	/** Padding below the image data */
	size_t bottom;
	/** Padding on the left of image data */
	size_t left;
};

/**
* @brief Profiling information about time spent in different phases of a function call.
*/
struct nnp_profile {
	/** Time spent inside the function call, in seconds. */
	double total;
	/** Time spend on transformation of the input or input gradient tensor, in seconds. */
	double input_transform;
	/** Time spend on transformation of the kernel or kernel gradient tensor, in seconds. */
	double kernel_transform;
	/** Time spend on transformation of the output or output gradient tensor, in seconds. */
	double output_transform;
	/** Time spend on multiplication-accumulation of transformed coefficients, in seconds. */
	double block_multiplication;
};

struct nnp_workspace_pointers {
	void* kernel;
	void* input;
	void* output;
};

enum nnp_status nnp_initialize();

enum nnp_status nnp_deinitialize();

enum nnp_status nnp_convolution_output(
	enum nnp_convolution_algorithm algorithm,
	const size_t batch_size,
	const size_t input_channels,
	const size_t output_channels,
	const struct nnp_size input_size,
	const struct nnp_padding input_padding,
	const struct nnp_size kernel_size,
	const float* input,
	const float* kernel,
	const float* bias,
	float* output,
	struct nnp_workspace_pointers* workspace_buffer,
	const enum nnp_activation activation,
	const void* activation_parameters,
	struct nnp_profile* profile);

enum nnp_status nnp_convolution_input_gradient(
	enum nnp_convolution_algorithm algorithm,
	const size_t batch_size,
	const size_t input_channels,
	const size_t output_channels,
	const struct nnp_size input_size,
	const struct nnp_padding input_padding,
	const struct nnp_size kernel_size,
	const float* grad_output,
	const float* kernel,
	float* grad_input,
	struct nnp_workspace_pointers* workspace_buffer,
	const enum nnp_activation activation,
	const void* activation_parameters,
	struct nnp_profile* profile);

enum nnp_status nnp_convolution_kernel_gradient(
	const enum nnp_convolution_algorithm algorithm,
	const size_t batch_size,
	const size_t input_channels,
	const size_t output_channels,
	const struct nnp_size input_size,
	const struct nnp_padding input_padding,
	const struct nnp_size kernel_size,
	const float* input,
	const float* grad_output,
	float* grad_kernel,
	struct nnp_workspace_pointers* workspace_buffer,
	const enum nnp_activation activation,
	const void* activation_parameters,
	struct nnp_profile* profile);

enum nnp_status nnp_convolution_inference(
	enum nnp_convolution_algorithm algorithm,
	const enum nnp_convolution_transform_strategy transform_strategy,
	const size_t input_channels,
	const size_t output_channels,
	const struct nnp_size input_size,
	const struct nnp_padding input_padding,
	const struct nnp_size kernel_size,
	const struct nnp_size output_subsampling,
	const float* input,
	const float* kernel,
	const float* bias,
	float* output,
	struct nnp_workspace_pointers* workspace_buffer,
	const enum nnp_activation activation,
	const void* activation_parameters,
	struct nnp_profile* profile);

enum nnp_status nnp_fully_connected_output(
	const size_t batch_size,
	const size_t input_channels,
	const size_t output_channels,
	const float* input,
	const float* kernel,
	float* output,
	struct nnp_profile* profile);

enum nnp_status nnp_fully_connected_inference(
	const size_t input_channels,
	const size_t output_channels,
	const float* input,
	const float* kernel,
	float* output);

enum nnp_status nnp_max_pooling_output(
	const size_t batch_size,
	const size_t channels,
	const struct nnp_size input_size,
	const struct nnp_padding input_padding,
	const struct nnp_size pooling_size,
	const struct nnp_size pooling_stride,
	const float* input,
	float* output);

enum nnp_status nnp_softmax_output(
	const size_t batch_size,
	const size_t channels,
	const float* input,
	float* output);

enum nnp_status nnp_relu_output(
	const size_t batch_size,
	const size_t channels,
	const float* input,
	float* output,
	const float negative_slope);

enum nnp_status nnp_relu_input_gradient(
	const size_t batch_size,
	const size_t channels,
	const float* grad_output,
	const float* input,
	float* grad_input,
	const float negative_slope);
#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
inline nnp_status nnp_convolution_output(
	nnp_convolution_algorithm algorithm,
	const size_t batch_size,
	const size_t input_channels,
	const size_t output_channels,
	const nnp_size input_size,
	const nnp_padding input_padding,
	const nnp_size kernel_size,
	const float input[],
	const float kernel[],
	const float bias[],
	float output[],
	nnp_profile* profile)
{
	return nnp_convolution_output(
		algorithm,
		batch_size, input_channels, output_channels,
		input_size, input_padding, kernel_size,
		input, kernel, bias, output,
		NULL, nnp_activation_identity, NULL, profile);
}

inline nnp_status nnp_convolution_input_gradient(
	nnp_convolution_algorithm algorithm,
	const size_t batch_size,
	const size_t input_channels,
	const size_t output_channels,
	const nnp_size input_size,
	const nnp_padding input_padding,
	const nnp_size kernel_size,
	const float grad_output[],
	const float kernel[],
	float grad_input[],
	nnp_profile* profile)
{
	return nnp_convolution_input_gradient(
		algorithm,
		batch_size, input_channels, output_channels,
		input_size, input_padding, kernel_size,
		grad_output, kernel, grad_input,
		NULL, nnp_activation_identity, NULL, profile);
}

inline nnp_status nnp_convolution_kernel_gradient(
	nnp_convolution_algorithm algorithm,
	const size_t batch_size,
	const size_t input_channels,
	const size_t output_channels,
	const nnp_size input_size,
	const nnp_padding input_padding,
	const nnp_size kernel_size,
	const float input[],
	const float grad_output[],
	float grad_kernel[],
	nnp_profile* profile)
{
	return nnp_convolution_kernel_gradient(
		algorithm,
		batch_size, input_channels, output_channels,
		input_size, input_padding, kernel_size,
		input, grad_output, grad_kernel,
		NULL, nnp_activation_identity, NULL, profile);
}

inline nnp_status nnp_convolution_inference(
	nnp_convolution_algorithm algorithm,
	const nnp_convolution_transform_strategy transform_strategy,
	const size_t input_channels,
	const size_t output_channels,
	const nnp_size input_size,
	const nnp_padding input_padding,
	const nnp_size kernel_size,
	const nnp_size output_subsampling,
	const float input[],
	const float kernel[],
	const float bias[],
	float output[],
	nnp_profile* profile)
{
	return nnp_convolution_inference(
		algorithm, transform_strategy,
		input_channels, output_channels,
		input_size, input_padding, kernel_size, output_subsampling,
		input, kernel, bias, output, NULL, 
		nnp_activation_identity, NULL, profile);
}
#endif // __cplusplus
