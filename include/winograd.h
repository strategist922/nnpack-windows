#pragma once

#if defined(__cplusplus) && (__cplusplus >= 201103L)
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

	typedef void(*nnp_wt_function)(const float*, float*);

	void nnp_iwt_f6k3__fma3(const float d[], float w[]);
	void nnp_kwt_f6k3__fma3(const float g[], float w[]);
	void nnp_owt_f6k3__fma3(const float m[], float s[]);

	void nnp_iwt_f6k3__scalar(const float d[], float w[]);
	void nnp_kwt_f6k3__scalar(const float g[], float w[]);
	void nnp_owt_f6k3__scalar(const float m[], float s[]);

#ifdef __cplusplus
}
#endif
