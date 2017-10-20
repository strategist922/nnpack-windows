#include <ccomplex>
#include <complex.h>
#include <complex-base.h>
#include <complex-ref.h>
#include <fft-constants.h>


void nnp_fft8_dualreal__ref(const float* t, float* f) 
{
	std::complex<float> w0 = CMPLXF(t[0], t[ 8]);
	std::complex<float> w1 = CMPLXF(t[1], t[ 9]);
	std::complex<float> w2 = CMPLXF(t[2], t[10]);
	std::complex<float> w3 = CMPLXF(t[3], t[11]);
	std::complex<float> w4 = CMPLXF(t[4], t[12]);
	std::complex<float> w5 = CMPLXF(t[5], t[13]);
	std::complex<float> w6 = CMPLXF(t[6], t[14]);
	std::complex<float> w7 = CMPLXF(t[7], t[15]);

	fft8fc(&w0, &w1, &w2, &w3, &w4, &w5, &w6, &w7);

	const float x0 = crealf(w0);
	const float h0 = cimagf(w0);
	

	const std::complex<float>  x1 =  0.5f  * (w1 + std::conj(w7));
	const std::complex<float> h1 = std::complex<float>(0.0f, -0.5f) * (w1 - std::conj(w7));
	const std::complex<float> x2 =  0.5f  * (w2 + std::conj(w6));
	const std::complex<float> h2 = std::complex<float>(0.0f, -0.5f) * (w2 - std::conj(w6));
	const std::complex<float> x3 =  0.5f  * (w3 + std::conj(w5));
	const std::complex<float> h3 = std::complex<float>(0.0f, -0.5f) * (w3 - std::conj(w5));

	const float x4 = crealf(w4);
	const float h4 = cimagf(w4);

	f[0] = x0;
	f[1] = h0;
	f[2] = crealf(x1);
	f[3] = crealf(h1);
	f[4] = crealf(x2);
	f[5] = crealf(h2);
	f[6] = crealf(x3);
	f[7] = crealf(h3);

	f[ 8] = x4;
	f[ 9] = h4;
	f[10] = cimagf(x1);
	f[11] = cimagf(h1);
	f[12] = cimagf(x2);
	f[13] = cimagf(h2);
	f[14] = cimagf(x3);
	f[15] = cimagf(h3);
}

void nnp_fft16_dualreal__ref(const float* t, float* f)
{
	std::complex<float> w0  = CMPLXF(t[ 0], t[16]);
	std::complex<float> w1  = CMPLXF(t[ 1], t[17]);
	std::complex<float> w2  = CMPLXF(t[ 2], t[18]);
	std::complex<float> w3  = CMPLXF(t[ 3], t[19]);
	std::complex<float> w4  = CMPLXF(t[ 4], t[20]);
	std::complex<float> w5  = CMPLXF(t[ 5], t[21]);
	std::complex<float> w6  = CMPLXF(t[ 6], t[22]);
	std::complex<float> w7  = CMPLXF(t[ 7], t[23]);
	std::complex<float> w8  = CMPLXF(t[ 8], t[24]);
	std::complex<float> w9  = CMPLXF(t[ 9], t[25]);
	std::complex<float> w10 = CMPLXF(t[10], t[26]);
	std::complex<float> w11 = CMPLXF(t[11], t[27]);
	std::complex<float> w12 = CMPLXF(t[12], t[28]);
	std::complex<float> w13 = CMPLXF(t[13], t[29]);
	std::complex<float> w14 = CMPLXF(t[14], t[30]);
	std::complex<float> w15 = CMPLXF(t[15], t[31]);

	fft16fc(&w0, &w1, &w2, &w3, &w4, &w5, &w6, &w7, &w8, &w9, &w10, &w11, &w12, &w13, &w14, &w15);

	const float x0 = crealf(w0);
	const float h0 = cimagf(w0);

	const std::complex<float> x1 =  0.5f  * (w1 + std::conj(w15));
	const std::complex<float> h1 = std::complex<float>(0.0f, -0.5f) * (w1 - std::conj(w15));
	const std::complex<float> x2 =  0.5f  * (w2 + std::conj(w14));
	const std::complex<float> h2 = std::complex<float>(0.0f, -0.5f) * (w2 - std::conj(w14));
	const std::complex<float> x3 =  0.5f  * (w3 + std::conj(w13));
	const std::complex<float> h3 = std::complex<float>(0.0f, -0.5f) * (w3 - std::conj(w13));
	const std::complex<float> x4 =  0.5f  * (w4 + std::conj(w12));
	const std::complex<float> h4 = std::complex<float>(0.0f, -0.5f) * (w4 - std::conj(w12));
	const std::complex<float> x5 =  0.5f  * (w5 + std::conj(w11));
	const std::complex<float> h5 = std::complex<float>(0.0f, -0.5f) * (w5 - std::conj(w11));
	const std::complex<float> x6 =  0.5f  * (w6 + std::conj(w10));
	const std::complex<float> h6 = std::complex<float>(0.0f, -0.5f) * (w6 - std::conj(w10));
	const std::complex<float> x7 =  0.5f  * (w7 + std::conj(w9));
	const std::complex<float> h7 = std::complex<float>(0.0f, -0.5f) * (w7 - std::conj(w9));

	const float x8 = crealf(w8);
	const float h8 = cimagf(w8);

	f[ 0] = x0;
	f[ 1] = h0;
	f[ 2] = crealf(x1);
	f[ 3] = crealf(h1);
	f[ 4] = crealf(x2);
	f[ 5] = crealf(h2);
	f[ 6] = crealf(x3);
	f[ 7] = crealf(h3);
	f[ 8] = crealf(x4);
	f[ 9] = crealf(h4);
	f[10] = crealf(x5);
	f[11] = crealf(h5);
	f[12] = crealf(x6);
	f[13] = crealf(h6);
	f[14] = crealf(x7);
	f[15] = crealf(h7);

	f[16] = x8;
	f[17] = h8;
	f[18] = cimagf(x1);
	f[19] = cimagf(h1);
	f[20] = cimagf(x2);
	f[21] = cimagf(h2);
	f[22] = cimagf(x3);
	f[23] = cimagf(h3);
	f[24] = cimagf(x4);
	f[25] = cimagf(h4);
	f[26] = cimagf(x5);
	f[27] = cimagf(h5);
	f[28] = cimagf(x6);
	f[29] = cimagf(h6);
	f[30] = cimagf(x7);
	f[31] = cimagf(h7);
}

void nnp_fft32_dualreal__ref(const float* t, float* f) 
{
	std::complex<float> w0  = CMPLXF(t[ 0], t[32]);
	std::complex<float> w1  = CMPLXF(t[ 1], t[33]);
	std::complex<float> w2  = CMPLXF(t[ 2], t[34]);
	std::complex<float> w3  = CMPLXF(t[ 3], t[35]);
	std::complex<float> w4  = CMPLXF(t[ 4], t[36]);
	std::complex<float> w5  = CMPLXF(t[ 5], t[37]);
	std::complex<float> w6  = CMPLXF(t[ 6], t[38]);
	std::complex<float> w7  = CMPLXF(t[ 7], t[39]);
	std::complex<float> w8  = CMPLXF(t[ 8], t[40]);
	std::complex<float> w9  = CMPLXF(t[ 9], t[41]);
	std::complex<float> w10 = CMPLXF(t[10], t[42]);
	std::complex<float> w11 = CMPLXF(t[11], t[43]);
	std::complex<float> w12 = CMPLXF(t[12], t[44]);
	std::complex<float> w13 = CMPLXF(t[13], t[45]);
	std::complex<float> w14 = CMPLXF(t[14], t[46]);
	std::complex<float> w15 = CMPLXF(t[15], t[47]);
	std::complex<float> w16 = CMPLXF(t[16], t[48]);
	std::complex<float> w17 = CMPLXF(t[17], t[49]);
	std::complex<float> w18 = CMPLXF(t[18], t[50]);
	std::complex<float> w19 = CMPLXF(t[19], t[51]);
	std::complex<float> w20 = CMPLXF(t[20], t[52]);
	std::complex<float> w21 = CMPLXF(t[21], t[53]);
	std::complex<float> w22 = CMPLXF(t[22], t[54]);
	std::complex<float> w23 = CMPLXF(t[23], t[55]);
	std::complex<float> w24 = CMPLXF(t[24], t[56]);
	std::complex<float> w25 = CMPLXF(t[25], t[57]);
	std::complex<float> w26 = CMPLXF(t[26], t[58]);
	std::complex<float> w27 = CMPLXF(t[27], t[59]);
	std::complex<float> w28 = CMPLXF(t[28], t[60]);
	std::complex<float> w29 = CMPLXF(t[29], t[61]);
	std::complex<float> w30 = CMPLXF(t[30], t[62]);
	std::complex<float> w31 = CMPLXF(t[31], t[63]);

	fft32fc(&w0, &w1, &w2, &w3, &w4, &w5, &w6, &w7, &w8, &w9, &w10, &w11, &w12, &w13, &w14, &w15, &w16, &w17, &w18, &w19, &w20, &w21, &w22, &w23, &w24, &w25, &w26, &w27, &w28, &w29, &w30, &w31);

	const float x0 = crealf(w0);
	const float h0 = cimagf(w0);

	const std::complex<float> x1  =  0.5f  * (w1  + std::conj(w31));
	const std::complex<float> h1  = std::complex<float>(0.0f, -0.5f) * (w1  - std::conj(w31));
	const std::complex<float> x2  =  0.5f  * (w2  + std::conj(w30));
	const std::complex<float> h2  = std::complex<float>(0.0f, -0.5f) * (w2  - std::conj(w30));
	const std::complex<float> x3  =  0.5f  * (w3  + std::conj(w29));
	const std::complex<float> h3  = std::complex<float>(0.0f, -0.5f) * (w3  - std::conj(w29));
	const std::complex<float> x4  =  0.5f  * (w4  + std::conj(w28));
	const std::complex<float> h4  = std::complex<float>(0.0f, -0.5f) * (w4  - std::conj(w28));
	const std::complex<float> x5  =  0.5f  * (w5  + std::conj(w27));
	const std::complex<float> h5  = std::complex<float>(0.0f, -0.5f) * (w5  - std::conj(w27));
	const std::complex<float> x6  =  0.5f  * (w6  + std::conj(w26));
	const std::complex<float> h6  = std::complex<float>(0.0f, -0.5f) * (w6  - std::conj(w26));
	const std::complex<float> x7  =  0.5f  * (w7  + std::conj(w25));
	const std::complex<float> h7  = std::complex<float>(0.0f, -0.5f) * (w7  - std::conj(w25));
	const std::complex<float> x8  =  0.5f  * (w8  + std::conj(w24));
	const std::complex<float> h8  = std::complex<float>(0.0f, -0.5f) * (w8  - std::conj(w24));
	const std::complex<float> x9  =  0.5f  * (w9  + std::conj(w23));
	const std::complex<float> h9  = std::complex<float>(0.0f, -0.5f) * (w9  - std::conj(w23));
	const std::complex<float> x10 =  0.5f  * (w10 + std::conj(w22));
	const std::complex<float> h10 = std::complex<float>(0.0f, -0.5f) * (w10 - std::conj(w22));
	const std::complex<float> x11 =  0.5f  * (w11 + std::conj(w21));
	const std::complex<float> h11 = std::complex<float>(0.0f, -0.5f) * (w11 - std::conj(w21));
	const std::complex<float> x12 =  0.5f  * (w12 + std::conj(w20));
	const std::complex<float> h12 = std::complex<float>(0.0f, -0.5f) * (w12 - std::conj(w20));
	const std::complex<float> x13 =  0.5f  * (w13 + std::conj(w19));
	const std::complex<float> h13 = std::complex<float>(0.0f, -0.5f) * (w13 - std::conj(w19));
	const std::complex<float> x14 =  0.5f  * (w14 + std::conj(w18));
	const std::complex<float> h14 = std::complex<float>(0.0f, -0.5f) * (w14 - std::conj(w18));
	const std::complex<float> x15 =  0.5f  * (w15 + std::conj(w17));
	const std::complex<float> h15 = std::complex<float>(0.0f, -0.5f) * (w15 - std::conj(w17));

	const float x16 = crealf(w16);
	const float h16 = cimagf(w16);

	f[ 0] = x0;
	f[ 1] = h0;
	f[ 2] = crealf(x1);
	f[ 3] = crealf(h1);
	f[ 4] = crealf(x2);
	f[ 5] = crealf(h2);
	f[ 6] = crealf(x3);
	f[ 7] = crealf(h3);
	f[ 8] = crealf(x4);
	f[ 9] = crealf(h4);
	f[10] = crealf(x5);
	f[11] = crealf(h5);
	f[12] = crealf(x6);
	f[13] = crealf(h6);
	f[14] = crealf(x7);
	f[15] = crealf(h7);
	f[16] = crealf(x8);
	f[17] = crealf(h8);
	f[18] = crealf(x9);
	f[19] = crealf(h9);
	f[20] = crealf(x10);
	f[21] = crealf(h10);
	f[22] = crealf(x11);
	f[23] = crealf(h11);
	f[24] = crealf(x12);
	f[25] = crealf(h12);
	f[26] = crealf(x13);
	f[27] = crealf(h13);
	f[28] = crealf(x14);
	f[29] = crealf(h14);
	f[30] = crealf(x15);
	f[31] = crealf(h15);
	//f[32] = x8;  bug in original code ???
	//f[33] = h8;
	f[32] = x16;
	f[33] = h16;

	f[34] = cimagf(x1);
	f[35] = cimagf(h1);
	f[36] = cimagf(x2);
	f[37] = cimagf(h2);
	f[38] = cimagf(x3);
	f[39] = cimagf(h3);
	f[40] = cimagf(x4);
	f[41] = cimagf(h4);
	f[42] = cimagf(x5);
	f[43] = cimagf(h5);
	f[44] = cimagf(x6);
	f[45] = cimagf(h6);
	f[46] = cimagf(x7);
	f[47] = cimagf(h7);
	f[48] = cimagf(x8);
	f[49] = cimagf(h8);
	f[50] = cimagf(x9);
	f[51] = cimagf(h9);
	f[52] = cimagf(x10);
	f[53] = cimagf(h10);
	f[54] = cimagf(x11);
	f[55] = cimagf(h11);
	f[56] = cimagf(x12);
	f[57] = cimagf(h12);
	f[58] = cimagf(x13);
	f[59] = cimagf(h13);
	f[60] = cimagf(x14);
	f[61] = cimagf(h14);
	f[62] = cimagf(x15);
	f[63] = cimagf(h15);
}
