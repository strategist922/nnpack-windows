#include <scalar/fft/real.h>


void nnp_fft8_real__scalar(
	const float* t,
	float* f)
{
	scalar_fft8_real(
		t, t + 4, 1, 0, 8,
		f, 1);
}

void nnp_fft16_real__scalar(
	const float* t,
	float* f)
{
	scalar_fft16_real(
		t, t + 8, 1, 0, 16,
		f, 1);
}

void nnp_ifft8_real__scalar(
	const float* f,
	float* t)
{
	const float f0r = f[0];
	const float f4r = f[1];
	const float f1r = f[2];
	const float f1i = f[3];
	const float f2r = f[4];
	const float f2i = f[5];
	const float f3r = f[6];
	const float f3i = f[7];
	scalar_ifft8_real(
		f0r, f4r, f1r, f1i, f2r, f2i, f3r, f3i,
		t, t + 4, 1);
}

void nnp_ifft16_real__scalar(
	const float* f,
	float* t)
{
	const float f0r = f[ 0];
	const float f8r = f[ 1];
	const float f1r = f[ 2];
	const float f1i = f[ 3];
	const float f2r = f[ 4];
	const float f2i = f[ 5];
	const float f3r = f[ 6];
	const float f3i = f[ 7];
	const float f4r = f[ 8];
	const float f4i = f[ 9];
	const float f5r = f[10];
	const float f5i = f[11];
	const float f6r = f[12];
	const float f6i = f[13];
	const float f7r = f[14];
	const float f7i = f[15];
	scalar_ifft16_real(
		f0r, f8r, f1r, f1i, f2r, f2i, f3r, f3i, f4r, f4i, f5r, f5i, f6r, f6i, f7r, f7i,
		t, t + 8, 1);
}
