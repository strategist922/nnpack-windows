#pragma once

#ifdef __cplusplus
#include <cstddef>
#include <cstdint>
#include <cstdbool>
#else
#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

	typedef void(*nnp_exp_function)(size_t, const float*, float*);

	void nnp_softmax__avx2(size_t n, const float* x, float* y);
	void nnp_inplace_softmax__avx2(size_t n, float* v);

	void nnp_softmax__scalar(size_t n, const float* x, float* y);
	void nnp_inplace_softmax__scalar(size_t n, float* v);

#ifdef __cplusplus
} /* extern "C" */
#endif



