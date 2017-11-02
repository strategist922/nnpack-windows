#include <stddef.h>


void nnp_sdotxf1__scalar(
	const float x[1],
	const float y[1],
	size_t stride_y,
	float sum[1],
	size_t n)
{
	float acc0 = 0.0f;
	const float* y0 = y;
	while (n--) {
		const float vx = *x++;
		acc0 += vx * (*y0++);
	}
	sum[0] = acc0;
}

void nnp_sdotxf2__scalar(
	const float x[1],
	const float y[2],
	size_t stride_y,
	float sum[2],
	size_t n)
{
	float acc0, acc1;
	acc0 = acc1 = 0.0f;
	const float* y0 = y;
	const float* y1 = y0 + stride_y;
	while (n--) {
		const float vx = *x++;
		acc0 += vx * (*y0++);
		acc1 += vx * (*y1++);
	}
	sum[0] = acc0;
	sum[1] = acc1;
}

void nnp_sdotxf3__scalar(
	const float x[1],
	const float y[3],
	size_t stride_y,
	float sum[3],
	size_t n)
{
	float acc0, acc1, acc2;
	acc0 = acc1 = acc2 = 0.0f;
	const float* y0 = y;
	const float* y1 = y0 + stride_y;
	const float* y2 = y1 + stride_y;
	while (n--) {
		const float vx = *x++;
		acc0 += vx * (*y0++);
		acc1 += vx * (*y1++);
		acc2 += vx * (*y2++);
	}
	sum[0] = acc0;
	sum[1] = acc1;
	sum[2] = acc2;
}

void nnp_sdotxf4__scalar(
	const float x[1],
	const float y[4],
	size_t stride_y,
	float sum[4],
	size_t n)
{
	float acc0, acc1, acc2, acc3;
	acc0 = acc1 = acc2 = acc3 = 0.0f;
	const float* y0 = y;
	const float* y1 = y0 + stride_y;
	const float* y2 = y1 + stride_y;
	const float* y3 = y2 + stride_y;
	while (n--) {
		const float vx = *x++;
		acc0 += vx * (*y0++);
		acc1 += vx * (*y1++);
		acc2 += vx * (*y2++);
		acc3 += vx * (*y3++);
	}
	sum[0] = acc0;
	sum[1] = acc1;
	sum[2] = acc2;
	sum[3] = acc3;
}

void nnp_sdotxf5__scalar(
	const float x[1],
	const float y[5],
	size_t stride_y,
	float sum[5],
	size_t n)
{
	float acc0, acc1, acc2, acc3, acc4;
	acc0 = acc1 = acc2 = acc3 = acc4 = 0.0f;
	const float* y0 = y;
	const float* y1 = y0 + stride_y;
	const float* y2 = y1 + stride_y;
	const float* y3 = y2 + stride_y;
	const float* y4 = y3 + stride_y;
	while (n--) {
		const float vx = *x++;
		acc0 += vx * (*y0++);
		acc1 += vx * (*y1++);
		acc2 += vx * (*y2++);
		acc3 += vx * (*y3++);
		acc4 += vx * (*y4++);
	}
	sum[0] = acc0;
	sum[1] = acc1;
	sum[2] = acc2;
	sum[3] = acc3;
	sum[4] = acc4;
}

void nnp_sdotxf6__scalar(
	const float x[1],
	const float y[6],
	size_t stride_y,
	float sum[6],
	size_t n)
{
	float acc0, acc1, acc2, acc3, acc4, acc5;
	acc0 = acc1 = acc2 = acc3 = acc4 = acc5 = 0.0f;
	const float* y0 = y;
	const float* y1 = y0 + stride_y;
	const float* y2 = y1 + stride_y;
	const float* y3 = y2 + stride_y;
	const float* y4 = y3 + stride_y;
	const float* y5 = y4 + stride_y;
	while (n--) {
		const float vx = *x++;
		acc0 += vx * (*y0++);
		acc1 += vx * (*y1++);
		acc2 += vx * (*y2++);
		acc3 += vx * (*y3++);
		acc4 += vx * (*y4++);
		acc5 += vx * (*y5++);
	}
	sum[0] = acc0;
	sum[1] = acc1;
	sum[2] = acc2;
	sum[3] = acc3;
	sum[4] = acc4;
	sum[5] = acc5;
}

void nnp_sdotxf7__scalar(
	const float x[1],
	const float y[7],
	size_t stride_y,
	float sum[7],
	size_t n)
{
	float acc0, acc1, acc2, acc3, acc4, acc5, acc6;
	acc0 = acc1 = acc2 = acc3 = acc4 = acc5 = acc6 = 0.0f;
	const float* y0 = y;
	const float* y1 = y0 + stride_y;
	const float* y2 = y1 + stride_y;
	const float* y3 = y2 + stride_y;
	const float* y4 = y3 + stride_y;
	const float* y5 = y4 + stride_y;
	const float* y6 = y5 + stride_y;
	while (n--) {
		const float vx = *x++;
		acc0 += vx * (*y0++);
		acc1 += vx * (*y1++);
		acc2 += vx * (*y2++);
		acc3 += vx * (*y3++);
		acc4 += vx * (*y4++);
		acc5 += vx * (*y5++);
		acc6 += vx * (*y6++);
	}
	sum[0] = acc0;
	sum[1] = acc1;
	sum[2] = acc2;
	sum[3] = acc3;
	sum[4] = acc4;
	sum[5] = acc5;
	sum[6] = acc6;
}

void nnp_sdotxf8__scalar(
	const float x[1],
	const float y[8],
	size_t stride_y,
	float sum[8],
	size_t n)
{
	float acc0, acc1, acc2, acc3, acc4, acc5, acc6, acc7;
	acc0 = acc1 = acc2 = acc3 = acc4 = acc5 = acc6 = acc7 = 0.0f;
	const float* y0 = y;
	const float* y1 = y0 + stride_y;
	const float* y2 = y1 + stride_y;
	const float* y3 = y2 + stride_y;
	const float* y4 = y3 + stride_y;
	const float* y5 = y4 + stride_y;
	const float* y6 = y5 + stride_y;
	const float* y7 = y6 + stride_y;
	while (n--) {
		const float vx = *x++;
		acc0 += vx * (*y0++);
		acc1 += vx * (*y1++);
		acc2 += vx * (*y2++);
		acc3 += vx * (*y3++);
		acc4 += vx * (*y4++);
		acc5 += vx * (*y5++);
		acc6 += vx * (*y6++);
		acc7 += vx * (*y7++);
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
