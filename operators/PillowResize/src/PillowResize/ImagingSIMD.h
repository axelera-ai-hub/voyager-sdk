#ifndef __IMAGING_SIMD_H__
#define __IMAGING_SIMD_H__

#include <stdint.h>
#define UINT32 uint32_t
#define INT32 int32_t
#define INT16 int16_t

/* Microsoft compiler doesn't limit intrinsics for an architecture.
   This macro is set only on x86 and means SSE2 and above including AVX2. */
#if defined(_M_X64) || _M_IX86_FP == 2
    #define __SSE4_2__
#endif

#if defined(__SSE4_2__)
    #include <emmintrin.h>
    #include <mmintrin.h>
    #include <smmintrin.h>
#endif

#if defined(__AVX2__) || __aarch64__
#define USE_AVX2
#define SIMDE_ENABLE_NATIVE_ALIASES
#include <simde/x86/avx2.h>
#endif

#if defined(__SSE4_2__) || __aarch64__
static __m128i inline
mm_cvtepu8_epi32(void *ptr) {
    return _mm_cvtepu8_epi32(_mm_cvtsi32_si128(*(INT32 *) ptr));
}
#endif

#if defined(__AVX2__) || __aarch64__
static __m256i inline
mm256_cvtepu8_epi32(void *ptr) {
    return _mm256_cvtepu8_epi32(_mm_loadl_epi64((__m128i *) ptr));
}
#endif



#include <vector>

#include <opencv2/opencv.hpp>

int normalize_coeffs_8bpc_original(int outSize, int ksize, const double *prekk, std::vector<int16_t>& kk);

#endif
