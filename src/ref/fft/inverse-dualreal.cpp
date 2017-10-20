#include <ccomplex>
#include <complex.h>
#include <complex-base.h>
#include <complex-ref.h>

void nnp_ifft8_dualreal__ref(const float* f, float* t) 
{
	const float x0 = f[0];
	const float h0 = f[1];
	const float x4 = f[8];
	const float h4 = f[9];

	const std::complex<float> x1 = CMPLXF(f[2], f[10]);
	const std::complex<float> h1 = CMPLXF(f[3], f[11]);
	const std::complex<float> x2 = CMPLXF(f[4], f[12]);
	const std::complex<float> h2 = CMPLXF(f[5], f[13]);
	const std::complex<float> x3 = CMPLXF(f[6], f[14]);
	const std::complex<float> h3 = CMPLXF(f[7], f[15]);

	std::complex<float> w0 = CMPLXF(x0, h0);
	std::complex<float> w1 =       x1 + std::complex<float>(0.0f, 1.0f) * h1;
	std::complex<float> w2 =       x2 + std::complex<float>(0.0f, 1.0f) * h2;
	std::complex<float> w3 =       x3 + std::complex<float>(0.0f, 1.0f) * h3;
	std::complex<float> w4 = CMPLXF(x4, h4);
	std::complex<float> w5 = conjf(x3 - std::complex<float>(0.0f, 1.0f) * h3);
	std::complex<float> w6 = conjf(x2 - std::complex<float>(0.0f, 1.0f) * h2);
	std::complex<float> w7 = conjf(x1 - std::complex<float>(0.0f, 1.0f) * h1);
	
	ifft8fc(&w0, &w1, &w2, &w3, &w4, &w5, &w6, &w7);

	t[ 0] = crealf(w0);
	t[ 1] = crealf(w1);
	t[ 2] = crealf(w2);
	t[ 3] = crealf(w3);
	t[ 4] = crealf(w4);
	t[ 5] = crealf(w5);
	t[ 6] = crealf(w6);
	t[ 7] = crealf(w7);
	t[ 8] = cimagf(w0);
	t[ 9] = cimagf(w1);
	t[10] = cimagf(w2);
	t[11] = cimagf(w3);
	t[12] = cimagf(w4);
	t[13] = cimagf(w5);
	t[14] = cimagf(w6);
	t[15] = cimagf(w7);
}

void nnp_ifft16_dualreal__ref(const float* f, float* t) 
{
	const float x0 = f[0];
	const float h0 = f[1];
	const float x8 = f[16];
	const float h8 = f[17];

	const std::complex<float> x1 = CMPLXF(f[ 2], f[18]);
	const std::complex<float> h1 = CMPLXF(f[ 3], f[19]);
	const std::complex<float> x2 = CMPLXF(f[ 4], f[20]);
	const std::complex<float> h2 = CMPLXF(f[ 5], f[21]);
	const std::complex<float> x3 = CMPLXF(f[ 6], f[22]);
	const std::complex<float> h3 = CMPLXF(f[ 7], f[23]);
	const std::complex<float> x4 = CMPLXF(f[ 8], f[24]);
	const std::complex<float> h4 = CMPLXF(f[ 9], f[25]);
	const std::complex<float> x5 = CMPLXF(f[10], f[26]);
	const std::complex<float> h5 = CMPLXF(f[11], f[27]);
	const std::complex<float> x6 = CMPLXF(f[12], f[28]);
	const std::complex<float> h6 = CMPLXF(f[13], f[29]);
	const std::complex<float> x7 = CMPLXF(f[14], f[30]);
	const std::complex<float> h7 = CMPLXF(f[15], f[31]);

	std::complex<float> w0  = CMPLXF(x0, h0);
	std::complex<float> w1  =       x1 + std::complex<float>(0.0f, 1.0f) * h1;
	std::complex<float> w2  =       x2 + std::complex<float>(0.0f, 1.0f) * h2;
	std::complex<float> w3  =       x3 + std::complex<float>(0.0f, 1.0f) * h3;
	std::complex<float> w4  =       x4 + std::complex<float>(0.0f, 1.0f) * h4;
	std::complex<float> w5  =       x5 + std::complex<float>(0.0f, 1.0f) * h5;
	std::complex<float> w6  =       x6 + std::complex<float>(0.0f, 1.0f) * h6;
	std::complex<float> w7  =       x7 + std::complex<float>(0.0f, 1.0f) * h7;
	std::complex<float> w8  = CMPLXF(x8, h8);
	std::complex<float> w9  = conjf(x7 - std::complex<float>(0.0f, 1.0f) * h7);
	std::complex<float> w10 = conjf(x6 - std::complex<float>(0.0f, 1.0f) * h6);
	std::complex<float> w11 = conjf(x5 - std::complex<float>(0.0f, 1.0f) * h5);
	std::complex<float> w12 = conjf(x4 - std::complex<float>(0.0f, 1.0f) * h4);
	std::complex<float> w13 = conjf(x3 - std::complex<float>(0.0f, 1.0f) * h3);
	std::complex<float> w14 = conjf(x2 - std::complex<float>(0.0f, 1.0f) * h2);
	std::complex<float> w15 = conjf(x1 - std::complex<float>(0.0f, 1.0f) * h1);
	
	ifft16fc(&w0, &w1, &w2, &w3, &w4, &w5, &w6, &w7, &w8, &w9, &w10, &w11, &w12, &w13, &w14, &w15);

	t[ 0] = crealf(w0);
	t[ 1] = crealf(w1);
	t[ 2] = crealf(w2);
	t[ 3] = crealf(w3);
	t[ 4] = crealf(w4);
	t[ 5] = crealf(w5);
	t[ 6] = crealf(w6);
	t[ 7] = crealf(w7);
	t[ 8] = crealf(w8);
	t[ 9] = crealf(w9);
	t[10] = crealf(w10);
	t[11] = crealf(w11);
	t[12] = crealf(w12);
	t[13] = crealf(w13);
	t[14] = crealf(w14);
	t[15] = crealf(w15);
	t[16] = cimagf(w0);
	t[17] = cimagf(w1);
	t[18] = cimagf(w2);
	t[19] = cimagf(w3);
	t[20] = cimagf(w4);
	t[21] = cimagf(w5);
	t[22] = cimagf(w6);
	t[23] = cimagf(w7);
	t[24] = cimagf(w8);
	t[25] = cimagf(w9);
	t[26] = cimagf(w10);
	t[27] = cimagf(w11);
	t[28] = cimagf(w12);
	t[29] = cimagf(w13);
	t[30] = cimagf(w14);
	t[31] = cimagf(w15);
}

void nnp_ifft32_dualreal__ref(const float* f, float* t) 
{
	const float x0 = f[0];
	const float h0 = f[1];
	const float x16 = f[32];
	const float h16 = f[33];

	const std::complex<float> x1  = CMPLXF(f[ 2], f[34]);
	const std::complex<float> h1  = CMPLXF(f[ 3], f[35]);
	const std::complex<float> x2  = CMPLXF(f[ 4], f[36]);
	const std::complex<float> h2  = CMPLXF(f[ 5], f[37]);
	const std::complex<float> x3  = CMPLXF(f[ 6], f[38]);
	const std::complex<float> h3  = CMPLXF(f[ 7], f[39]);
	const std::complex<float> x4  = CMPLXF(f[ 8], f[40]);
	const std::complex<float> h4  = CMPLXF(f[ 9], f[41]);
	const std::complex<float> x5  = CMPLXF(f[10], f[42]);
	const std::complex<float> h5  = CMPLXF(f[11], f[43]);
	const std::complex<float> x6  = CMPLXF(f[12], f[44]);
	const std::complex<float> h6  = CMPLXF(f[13], f[45]);
	const std::complex<float> x7  = CMPLXF(f[14], f[46]);
	const std::complex<float> h7  = CMPLXF(f[15], f[47]);
	const std::complex<float> x8  = CMPLXF(f[16], f[48]);
	const std::complex<float> h8  = CMPLXF(f[17], f[49]);
	const std::complex<float> x9  = CMPLXF(f[18], f[50]);
	const std::complex<float> h9  = CMPLXF(f[19], f[51]);
	const std::complex<float> x10 = CMPLXF(f[20], f[52]);
	const std::complex<float> h10 = CMPLXF(f[21], f[53]);
	const std::complex<float> x11 = CMPLXF(f[22], f[54]);
	const std::complex<float> h11 = CMPLXF(f[23], f[55]);
	const std::complex<float> x12 = CMPLXF(f[24], f[56]);
	const std::complex<float> h12 = CMPLXF(f[25], f[57]);
	const std::complex<float> x13 = CMPLXF(f[26], f[58]);
	const std::complex<float> h13 = CMPLXF(f[27], f[59]);
	const std::complex<float> x14 = CMPLXF(f[28], f[60]);
	const std::complex<float> h14 = CMPLXF(f[29], f[61]);
	const std::complex<float> x15 = CMPLXF(f[30], f[62]);
	const std::complex<float> h15 = CMPLXF(f[31], f[63]);

	std::complex<float> w0  = CMPLXF(x0, h0);
	std::complex<float> w1  =       x1  + std::complex<float>(0.0f, 1.0f) * h1;
	std::complex<float> w2  =       x2  + std::complex<float>(0.0f, 1.0f) * h2;
	std::complex<float> w3  =       x3  + std::complex<float>(0.0f, 1.0f) * h3;
	std::complex<float> w4  =       x4  + std::complex<float>(0.0f, 1.0f) * h4;
	std::complex<float> w5  =       x5  + std::complex<float>(0.0f, 1.0f) * h5;
	std::complex<float> w6  =       x6  + std::complex<float>(0.0f, 1.0f) * h6;
	std::complex<float> w7  =       x7  + std::complex<float>(0.0f, 1.0f) * h7;
	std::complex<float> w8  =       x8  + std::complex<float>(0.0f, 1.0f) * h8;
	std::complex<float> w9  =       x9  + std::complex<float>(0.0f, 1.0f) * h9;
	std::complex<float> w10 =       x10 + std::complex<float>(0.0f, 1.0f) * h10;
	std::complex<float> w11 =       x11 + std::complex<float>(0.0f, 1.0f) * h11;
	std::complex<float> w12 =       x12 + std::complex<float>(0.0f, 1.0f) * h12;
	std::complex<float> w13 =       x13 + std::complex<float>(0.0f, 1.0f) * h13;
	std::complex<float> w14 =       x14 + std::complex<float>(0.0f, 1.0f) * h14;
	std::complex<float> w15 =       x15 + std::complex<float>(0.0f, 1.0f) * h15;
	std::complex<float> w16 = CMPLXF(x16, h16);
	std::complex<float> w17 = conjf(x15 - std::complex<float>(0.0f, 1.0f) * h15);
	std::complex<float> w18 = conjf(x14 - std::complex<float>(0.0f, 1.0f) * h14);
	std::complex<float> w19 = conjf(x13 - std::complex<float>(0.0f, 1.0f) * h13);
	std::complex<float> w20 = conjf(x12 - std::complex<float>(0.0f, 1.0f) * h12);
	std::complex<float> w21 = conjf(x11 - std::complex<float>(0.0f, 1.0f) * h11);
	std::complex<float> w22 = conjf(x10 - std::complex<float>(0.0f, 1.0f) * h10);
	std::complex<float> w23 = conjf(x9  - std::complex<float>(0.0f, 1.0f) * h9);
	std::complex<float> w24 = conjf(x8  - std::complex<float>(0.0f, 1.0f) * h8);
	std::complex<float> w25 = conjf(x7  - std::complex<float>(0.0f, 1.0f) * h7);
	std::complex<float> w26 = conjf(x6  - std::complex<float>(0.0f, 1.0f) * h6);
	std::complex<float> w27 = conjf(x5  - std::complex<float>(0.0f, 1.0f) * h5);
	std::complex<float> w28 = conjf(x4  - std::complex<float>(0.0f, 1.0f) * h4);
	std::complex<float> w29 = conjf(x3  - std::complex<float>(0.0f, 1.0f) * h3);
	std::complex<float> w30 = conjf(x2  - std::complex<float>(0.0f, 1.0f) * h2);
	std::complex<float> w31 = conjf(x1  - std::complex<float>(0.0f, 1.0f) * h1);

	ifft32fc(&w0, &w1, &w2, &w3, &w4, &w5, &w6, &w7, &w8, &w9, &w10, &w11, &w12, &w13, &w14, &w15, &w16, &w17, &w18, &w19, &w20, &w21, &w22, &w23, &w24, &w25, &w26, &w27, &w28, &w29, &w30, &w31);

	t[ 0] = crealf(w0);
	t[ 1] = crealf(w1);
	t[ 2] = crealf(w2);
	t[ 3] = crealf(w3);
	t[ 4] = crealf(w4);
	t[ 5] = crealf(w5);
	t[ 6] = crealf(w6);
	t[ 7] = crealf(w7);
	t[ 8] = crealf(w8);
	t[ 9] = crealf(w9);
	t[10] = crealf(w10);
	t[11] = crealf(w11);
	t[12] = crealf(w12);
	t[13] = crealf(w13);
	t[14] = crealf(w14);
	t[15] = crealf(w15);
	t[16] = crealf(w16);
	t[17] = crealf(w17);
	t[18] = crealf(w18);
	t[19] = crealf(w19);
	t[20] = crealf(w20);
	t[21] = crealf(w21);
	t[22] = crealf(w22);
	t[23] = crealf(w23);
	t[24] = crealf(w24);
	t[25] = crealf(w25);
	t[26] = crealf(w26);
	t[27] = crealf(w27);
	t[28] = crealf(w28);
	t[29] = crealf(w29);
	t[30] = crealf(w30);
	t[31] = crealf(w31);
	t[32] = cimagf(w0);
	t[33] = cimagf(w1);
	t[34] = cimagf(w2);
	t[35] = cimagf(w3);
	t[36] = cimagf(w4);
	t[37] = cimagf(w5);
	t[38] = cimagf(w6);
	t[39] = cimagf(w7);
	t[40] = cimagf(w8);
	t[41] = cimagf(w9);
	t[42] = cimagf(w10);
	t[43] = cimagf(w11);
	t[44] = cimagf(w12);
	t[45] = cimagf(w13);
	t[46] = cimagf(w14);
	t[47] = cimagf(w15);
	t[48] = cimagf(w16);
	t[49] = cimagf(w17);
	t[50] = cimagf(w18);
	t[51] = cimagf(w19);
	t[52] = cimagf(w20);
	t[53] = cimagf(w21);
	t[54] = cimagf(w22);
	t[55] = cimagf(w23);
	t[56] = cimagf(w24);
	t[57] = cimagf(w25);
	t[58] = cimagf(w26);
	t[59] = cimagf(w27);
	t[60] = cimagf(w28);
	t[61] = cimagf(w29);
	t[62] = cimagf(w30);
	t[63] = cimagf(w31);
}
