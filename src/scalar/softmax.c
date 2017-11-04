#include <stdint.h>
#include <stddef.h>
#include <math.h>

#include <softmax.h>


static float max__scalar(size_t n, const float* v) 
{
	float max_v = v[0];
	for (size_t i = 1; i < n; i++)
		max_v = fmaxf(max_v, v[i]);
	
	return max_v;
}

static float sum_exp_minus_c__scalar(size_t n, const float* v, float c) 
{
	float sum = 0.0f;
	for (size_t i = 0; i < n; i++)
		sum += expf(v[i] - c);

	return sum;
}

static void scaled_exp_minus_c__scalar(size_t n, const float* x, float* y, float scale, float c) 
{
	for (size_t i = 0; i < n; i++)
		y[i] = scale * expf(x[i] + c);
}

void nnp_softmax__scalar(
	size_t n,
	const float* x,
	float* y)
{
	const float c = max__scalar(n, x);
	const float sum = sum_exp_minus_c__scalar(n, x, c);
	const float scale = 1.0f / sum;
	scaled_exp_minus_c__scalar(n, x, y, scale, c);
}

void nnp_inplace_softmax__scalar(
	size_t n,
	float* v)
{
	const float c = max__scalar(n, v);
	const float sum = sum_exp_minus_c__scalar(n, v, c);
	const float scale = 1.0f / sum;
	scaled_exp_minus_c__scalar(n, v, v, scale, c);
}
