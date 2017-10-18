#pragma once
#include <complex.h>

#ifndef CMPLXF
	#define CMPLXF(real, imag) std::complex<float>(real, imag) // ((real) + _Complex_F _Complex_I * (imag)) 
#endif

