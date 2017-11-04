#include <stddef.h>

#include <fp16.h>


void nnp_shdotxf1__scalar(
	const float x[1],
	const uint16_t y[1],
	size_t stride_y,
	float sum[1],
	size_t n)
{
	float acc0 = 0.0f;
	const uint16_t* y0 = y;
	while (n--) {
		const float vx = *x++;
		acc0 += vx * fp16_alt_to_fp32_value(*y0++);
	}
	sum[0] = acc0;
}

void nnp_shdotxf2__scalar(
	const float x[1],
	const uint16_t y[2],
	size_t stride_y,
	float sum[2],
	size_t n)
{
	float acc0 = 0.0f;
	float acc1 = 0.0f;

	const uint16_t* y0 = y;
	const uint16_t* y1 = y0 + stride_y;
	while (n--) {
		const float vx = *x++;
		acc0 += vx * fp16_alt_to_fp32_value(*y0++);
		acc1 += vx * fp16_alt_to_fp32_value(*y1++);
	}
	sum[0] = acc0;
	sum[1] = acc1;
}

void nnp_shdotxf3__scalar(
	const float x[1],
	const uint16_t y[3],
	size_t stride_y,
	float sum[3],
	size_t n)
{
	float acc0 = 0.0f;
	float acc1 = 0.0f;
	float acc2 = 0.0f;
	
	const uint16_t* y0 = y;
	const uint16_t* y1 = y0 + stride_y;
	const uint16_t* y2 = y1 + stride_y;
	while (n--) {
		const float vx = *x++;
		acc0 += vx * fp16_alt_to_fp32_value(*y0++);
		acc1 += vx * fp16_alt_to_fp32_value(*y1++);
		acc2 += vx * fp16_alt_to_fp32_value(*y2++);
	}
	sum[0] = acc0;
	sum[1] = acc1;
	sum[2] = acc2;
}

void nnp_shdotxf4__scalar(
	const float x[1],
	const uint16_t y[4],
	size_t stride_y,
	float sum[4],
	size_t n)
{
	float acc0 = 0.0f;
	float acc1 = 0.0f;
	float acc2 = 0.0f;
	float acc3 = 0.0f;

	const uint16_t* y0 = y;
	const uint16_t* y1 = y0 + stride_y;
	const uint16_t* y2 = y1 + stride_y;
	const uint16_t* y3 = y2 + stride_y;
	while (n--) {
		const float vx = *x++;
		acc0 += vx * fp16_alt_to_fp32_value(*y0++);
		acc1 += vx * fp16_alt_to_fp32_value(*y1++);
		acc2 += vx * fp16_alt_to_fp32_value(*y2++);
		acc3 += vx * fp16_alt_to_fp32_value(*y3++);
	}
	sum[0] = acc0;
	sum[1] = acc1;
	sum[2] = acc2;
	sum[3] = acc3;
}

void nnp_shdotxf5__scalar(
	const float x[1],
	const uint16_t y[5],
	size_t stride_y,
	float sum[5],
	size_t n)
{
	float acc0 = 0.0f;
	float acc1 = 0.0f;
	float acc2 = 0.0f;
	float acc3 = 0.0f;
	float acc4 = 0.0f;
	const uint16_t* y0 = y;
	const uint16_t* y1 = y0 + stride_y;
	const uint16_t* y2 = y1 + stride_y;
	const uint16_t* y3 = y2 + stride_y;
	const uint16_t* y4 = y3 + stride_y;
	while (n--) 
	{
		const float vx = *x++;
		acc0 += vx * fp16_alt_to_fp32_value(*y0++);
		acc1 += vx * fp16_alt_to_fp32_value(*y1++);
		acc2 += vx * fp16_alt_to_fp32_value(*y2++);
		acc3 += vx * fp16_alt_to_fp32_value(*y3++);
		acc4 += vx * fp16_alt_to_fp32_value(*y4++);
	}
	sum[0] = acc0;
	sum[1] = acc1;
	sum[2] = acc2;
	sum[3] = acc3;
	sum[4] = acc4;
}

void nnp_shdotxf6__scalar(
	const float x[1],
	const uint16_t y[6],
	size_t stride_y,
	float sum[6],
	size_t n)
{
	float acc0 = 0.0f;
	float acc1 = 0.0f;
	float acc2 = 0.0f;
	float acc3 = 0.0f;
	float acc4 = 0.0f;
	float acc5 = 0.0f;

	const uint16_t* y0 = y;
	const uint16_t* y1 = y0 + stride_y;
	const uint16_t* y2 = y1 + stride_y;
	const uint16_t* y3 = y2 + stride_y;
	const uint16_t* y4 = y3 + stride_y;
	const uint16_t* y5 = y4 + stride_y;
	while (n--) 
	{
		const float vx = *x++;
		acc0 += vx * fp16_alt_to_fp32_value(*y0++);
		acc1 += vx * fp16_alt_to_fp32_value(*y1++);
		acc2 += vx * fp16_alt_to_fp32_value(*y2++);
		acc3 += vx * fp16_alt_to_fp32_value(*y3++);
		acc4 += vx * fp16_alt_to_fp32_value(*y4++);
		acc5 += vx * fp16_alt_to_fp32_value(*y5++);
	}
	sum[0] = acc0;
	sum[1] = acc1;
	sum[2] = acc2;
	sum[3] = acc3;
	sum[4] = acc4;
	sum[5] = acc5;
}

void nnp_shdotxf7__scalar(
	const float x[1],
	const uint16_t y[7],
	size_t stride_y,
	float sum[7],
	size_t n)
{
	float acc0 = 0.0f;
	float acc1 = 0.0f;
	float acc2 = 0.0f;
	float acc3 = 0.0f;
	float acc4 = 0.0f;
	float acc5 = 0.0f;
	float acc6 = 0.0f;
	const uint16_t* y0 = y;
	const uint16_t* y1 = y0 + stride_y;
	const uint16_t* y2 = y1 + stride_y;
	const uint16_t* y3 = y2 + stride_y;
	const uint16_t* y4 = y3 + stride_y;
	const uint16_t* y5 = y4 + stride_y;
	const uint16_t* y6 = y5 + stride_y;
	while (n--) 
	{
		const float vx = *x++;
		acc0 += vx * fp16_alt_to_fp32_value(*y0++);
		acc1 += vx * fp16_alt_to_fp32_value(*y1++);
		acc2 += vx * fp16_alt_to_fp32_value(*y2++);
		acc3 += vx * fp16_alt_to_fp32_value(*y3++);
		acc4 += vx * fp16_alt_to_fp32_value(*y4++);
		acc5 += vx * fp16_alt_to_fp32_value(*y5++);
		acc6 += vx * fp16_alt_to_fp32_value(*y6++);
	}
	sum[0] = acc0;
	sum[1] = acc1;
	sum[2] = acc2;
	sum[3] = acc3;
	sum[4] = acc4;
	sum[5] = acc5;
	sum[6] = acc6;
}

void nnp_shdotxf8__scalar(
	const float x[1],
	const uint16_t y[8],
	size_t stride_y,
	float sum[8],
	size_t n)
{
	float acc0 = 0.0f;
	float acc1 = 0.0f;
	float acc2 = 0.0f;
	float acc3 = 0.0f;
	float acc4 = 0.0f;
	float acc5 = 0.0f;
	float acc6 = 0.0f;
	float acc7 = 0.0f;

	const uint16_t* y0 = y;
	const uint16_t* y1 = y0 + stride_y;
	const uint16_t* y2 = y1 + stride_y;
	const uint16_t* y3 = y2 + stride_y;
	const uint16_t* y4 = y3 + stride_y;
	const uint16_t* y5 = y4 + stride_y;
	const uint16_t* y6 = y5 + stride_y;
	const uint16_t* y7 = y6 + stride_y;
	while (n--) {
		const float vx = *x++;
		acc0 += vx * fp16_alt_to_fp32_value(*y0++);
		acc1 += vx * fp16_alt_to_fp32_value(*y1++);
		acc2 += vx * fp16_alt_to_fp32_value(*y2++);
		acc3 += vx * fp16_alt_to_fp32_value(*y3++);
		acc4 += vx * fp16_alt_to_fp32_value(*y4++);
		acc5 += vx * fp16_alt_to_fp32_value(*y5++);
		acc6 += vx * fp16_alt_to_fp32_value(*y6++);
		acc7 += vx * fp16_alt_to_fp32_value(*y7++);
	}
	sum[0] = acc0;
	sum[1] = acc1;
	sum[2] = acc2;
	sum[3] = acc3;
	sum[4] = acc4;
	sum[5] = acc5;
	sum[6] = acc6;
	sum[7] = acc7;
}


