
#include <math.h>

#include "../../include/utils.h"
#include "../../include/softmax.h"

#ifdef __cplusplus
extern "C" {
#endif
	float max__avx(size_t n, const float* v);
	float sum_exp_minus_c__avx2(size_t n, const float* v, float c);
	void scaled_exp_minus_c__avx2(size_t n, const float* x, float* y, float scale, float c);
	void inplace_scaled_exp_minus_c__avx2(size_t n, const float* v, float scale, float c);

#ifdef __cplusplus
} /* extern "C" */
#endif

void nnp_softmax__avx2(size_t n, const float* x, float* y)
{
	const float c = max__avx(n, x);
	const float sum = sum_exp_minus_c__avx2(n, x, c);
	const float scale = 1.0f / sum;
	scaled_exp_minus_c__avx2(n, x, y, scale, c);
}

void nnp_inplace_softmax__avx2(size_t n, float* v)
{
	const float c = max__avx(n, v);
	const float sum = sum_exp_minus_c__avx2(n, v, c);
	const float scale = 1.0f / sum;
	inplace_scaled_exp_minus_c__avx2(n, v, scale, c);
}
