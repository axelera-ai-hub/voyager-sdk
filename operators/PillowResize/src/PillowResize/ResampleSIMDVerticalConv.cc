#include "ImagingSIMD.h"

void ImagingResampleVerticalConvolution8u(UINT32 *lineOut, UINT32 **imIn,
                                          int xsize, int xmin, int xmax, INT16 *k, int coefs_precision)
{
    int x;
    int xx = 0;

    __m128i initial = _mm_set1_epi32(1 << (coefs_precision-1));

#if defined(USE_AVX2)

    __m256i initial_256 = _mm256_set1_epi32(1 << (coefs_precision-1));

    for (; xx < xsize - 7; xx += 8) {
        __m256i sss0 = initial_256;
        __m256i sss1 = initial_256;
        __m256i sss2 = initial_256;
        __m256i sss3 = initial_256;
        x = 0;
        for (; x < xmax - 1; x += 2) {
            __m256i source, source1, source2;
            __m256i pix, mmk;

            // Load two coefficients at once
            mmk = _mm256_set1_epi32(*(INT32 *) &k[x]);
            
            source1 = _mm256_loadu_si256(  // top line
                (__m256i *) &imIn[x + xmin][xx]);
            source2 = _mm256_loadu_si256(  // bottom line
                (__m256i *) &imIn[x + 1 + xmin][xx]);

            source = _mm256_unpacklo_epi8(source1, source2);
            pix = _mm256_unpacklo_epi8(source, _mm256_setzero_si256());
            sss0 = _mm256_add_epi32(sss0, _mm256_madd_epi16(pix, mmk));
            pix = _mm256_unpackhi_epi8(source, _mm256_setzero_si256());
            sss1 = _mm256_add_epi32(sss1, _mm256_madd_epi16(pix, mmk));

            source = _mm256_unpackhi_epi8(source1, source2);
            pix = _mm256_unpacklo_epi8(source, _mm256_setzero_si256());
            sss2 = _mm256_add_epi32(sss2, _mm256_madd_epi16(pix, mmk));
            pix = _mm256_unpackhi_epi8(source, _mm256_setzero_si256());
            sss3 = _mm256_add_epi32(sss3, _mm256_madd_epi16(pix, mmk));
        }
        for (; x < xmax; x += 1) {
            __m256i source, source1, pix, mmk;
            mmk = _mm256_set1_epi32(k[x]);
            
            source1 = _mm256_loadu_si256(  // top line
                (__m256i *) &imIn[x + xmin][xx]);
            
            source = _mm256_unpacklo_epi8(source1, _mm256_setzero_si256());
            pix = _mm256_unpacklo_epi8(source, _mm256_setzero_si256());
            sss0 = _mm256_add_epi32(sss0, _mm256_madd_epi16(pix, mmk));
            pix = _mm256_unpackhi_epi8(source, _mm256_setzero_si256());
            sss1 = _mm256_add_epi32(sss1, _mm256_madd_epi16(pix, mmk));

            source = _mm256_unpackhi_epi8(source1, _mm256_setzero_si256());
            pix = _mm256_unpacklo_epi8(source, _mm256_setzero_si256());
            sss2 = _mm256_add_epi32(sss2, _mm256_madd_epi16(pix, mmk));
            pix = _mm256_unpackhi_epi8(source, _mm256_setzero_si256());
            sss3 = _mm256_add_epi32(sss3, _mm256_madd_epi16(pix, mmk));
        }
        sss0 = _mm256_srai_epi32(sss0, coefs_precision);
        sss1 = _mm256_srai_epi32(sss1, coefs_precision);
        sss2 = _mm256_srai_epi32(sss2, coefs_precision);
        sss3 = _mm256_srai_epi32(sss3, coefs_precision);

        sss0 = _mm256_packs_epi32(sss0, sss1);
        sss2 = _mm256_packs_epi32(sss2, sss3);
        sss0 = _mm256_packus_epi16(sss0, sss2);
        _mm256_storeu_si256((__m256i *) &lineOut[xx], sss0);
    }

#else

    for (; xx < xsize - 7; xx += 8) {
        __m128i sss0 = initial;
        __m128i sss1 = initial;
        __m128i sss2 = initial;
        __m128i sss3 = initial;
        __m128i sss4 = initial;
        __m128i sss5 = initial;
        __m128i sss6 = initial;
        __m128i sss7 = initial;
        x = 0;
        for (; x < xmax - 1; x += 2) {
            __m128i source, source1, source2;
            __m128i pix, mmk;

            // Load two coefficients at once
            mmk = _mm_set1_epi32(*(INT32 *) &k[x]);
            
            source1 = _mm_loadu_si128(  // top line
                (__m128i *) &imIn[x + xmin][xx]);
            source2 = _mm_loadu_si128(  // bottom line
                (__m128i *) &imIn[x + 1 + xmin][xx]);

            source = _mm_unpacklo_epi8(source1, source2);
            pix = _mm_unpacklo_epi8(source, _mm_setzero_si128());
            sss0 = _mm_add_epi32(sss0, _mm_madd_epi16(pix, mmk));
            pix = _mm_unpackhi_epi8(source, _mm_setzero_si128());
            sss1 = _mm_add_epi32(sss1, _mm_madd_epi16(pix, mmk));

            source = _mm_unpackhi_epi8(source1, source2);
            pix = _mm_unpacklo_epi8(source, _mm_setzero_si128());
            sss2 = _mm_add_epi32(sss2, _mm_madd_epi16(pix, mmk));
            pix = _mm_unpackhi_epi8(source, _mm_setzero_si128());
            sss3 = _mm_add_epi32(sss3, _mm_madd_epi16(pix, mmk));
            
            source1 = _mm_loadu_si128(  // top line
                (__m128i *) &imIn[x + xmin][xx + 4]);
            source2 = _mm_loadu_si128(  // bottom line
                (__m128i *) &imIn[x + 1 + xmin][xx + 4]);

            source = _mm_unpacklo_epi8(source1, source2);
            pix = _mm_unpacklo_epi8(source, _mm_setzero_si128());
            sss4 = _mm_add_epi32(sss4, _mm_madd_epi16(pix, mmk));
            pix = _mm_unpackhi_epi8(source, _mm_setzero_si128());
            sss5 = _mm_add_epi32(sss5, _mm_madd_epi16(pix, mmk));

            source = _mm_unpackhi_epi8(source1, source2);
            pix = _mm_unpacklo_epi8(source, _mm_setzero_si128());
            sss6 = _mm_add_epi32(sss6, _mm_madd_epi16(pix, mmk));
            pix = _mm_unpackhi_epi8(source, _mm_setzero_si128());
            sss7 = _mm_add_epi32(sss7, _mm_madd_epi16(pix, mmk));
        }
        for (; x < xmax; x += 1) {
            __m128i source, source1, pix, mmk;
            mmk = _mm_set1_epi32(k[x]);
            
            source1 = _mm_loadu_si128(  // top line
                (__m128i *) &imIn[x + xmin][xx]);
            
            source = _mm_unpacklo_epi8(source1, _mm_setzero_si128());
            pix = _mm_unpacklo_epi8(source, _mm_setzero_si128());
            sss0 = _mm_add_epi32(sss0, _mm_madd_epi16(pix, mmk));
            pix = _mm_unpackhi_epi8(source, _mm_setzero_si128());
            sss1 = _mm_add_epi32(sss1, _mm_madd_epi16(pix, mmk));

            source = _mm_unpackhi_epi8(source1, _mm_setzero_si128());
            pix = _mm_unpacklo_epi8(source, _mm_setzero_si128());
            sss2 = _mm_add_epi32(sss2, _mm_madd_epi16(pix, mmk));
            pix = _mm_unpackhi_epi8(source, _mm_setzero_si128());
            sss3 = _mm_add_epi32(sss3, _mm_madd_epi16(pix, mmk));

            source1 = _mm_loadu_si128(  // top line
                (__m128i *) &imIn[x + xmin][xx + 4]);

            source = _mm_unpacklo_epi8(source1, _mm_setzero_si128());
            pix = _mm_unpacklo_epi8(source, _mm_setzero_si128());
            sss4 = _mm_add_epi32(sss4, _mm_madd_epi16(pix, mmk));
            pix = _mm_unpackhi_epi8(source, _mm_setzero_si128());
            sss5 = _mm_add_epi32(sss5, _mm_madd_epi16(pix, mmk));

            source = _mm_unpackhi_epi8(source1, _mm_setzero_si128());
            pix = _mm_unpacklo_epi8(source, _mm_setzero_si128());
            sss6 = _mm_add_epi32(sss6, _mm_madd_epi16(pix, mmk));
            pix = _mm_unpackhi_epi8(source, _mm_setzero_si128());
            sss7 = _mm_add_epi32(sss7, _mm_madd_epi16(pix, mmk));
        }
        sss0 = _mm_srai_epi32(sss0, coefs_precision);
        sss1 = _mm_srai_epi32(sss1, coefs_precision);
        sss2 = _mm_srai_epi32(sss2, coefs_precision);
        sss3 = _mm_srai_epi32(sss3, coefs_precision);
        sss4 = _mm_srai_epi32(sss4, coefs_precision);
        sss5 = _mm_srai_epi32(sss5, coefs_precision);
        sss6 = _mm_srai_epi32(sss6, coefs_precision);
        sss7 = _mm_srai_epi32(sss7, coefs_precision);

        sss0 = _mm_packs_epi32(sss0, sss1);
        sss2 = _mm_packs_epi32(sss2, sss3);
        sss0 = _mm_packus_epi16(sss0, sss2);
        _mm_storeu_si128((__m128i *) &lineOut[xx], sss0);
        sss4 = _mm_packs_epi32(sss4, sss5);
        sss6 = _mm_packs_epi32(sss6, sss7);
        sss4 = _mm_packus_epi16(sss4, sss6);
        _mm_storeu_si128((__m128i *) &lineOut[xx + 4], sss4);
    }

#endif

    for (; xx < xsize - 1; xx += 2) {
        __m128i sss0 = initial;  // left row
        __m128i sss1 = initial;  // right row
        x = 0;
        for (; x < xmax - 1; x += 2) {
            __m128i source, source1, source2;
            __m128i pix, mmk;

            // Load two coefficients at once
            mmk = _mm_set1_epi32(*(INT32 *) &k[x]);

            source1 = _mm_loadl_epi64(  // top line
                (__m128i *) &imIn[x + xmin][xx]);
            source2 = _mm_loadl_epi64(  // bottom line
                (__m128i *) &imIn[x + 1 + xmin][xx]);
            
            source = _mm_unpacklo_epi8(source1, source2);
            pix = _mm_unpacklo_epi8(source, _mm_setzero_si128());
            sss0 = _mm_add_epi32(sss0, _mm_madd_epi16(pix, mmk));
            pix = _mm_unpackhi_epi8(source, _mm_setzero_si128());
            sss1 = _mm_add_epi32(sss1, _mm_madd_epi16(pix, mmk));
        }
        for (; x < xmax; x += 1) {
            __m128i source, source1, pix, mmk;
            mmk = _mm_set1_epi32(k[x]);
            
            source1 = _mm_loadl_epi64(  // top line
                (__m128i *) &imIn[x + xmin][xx]);
            
            source = _mm_unpacklo_epi8(source1, _mm_setzero_si128());
            pix = _mm_unpacklo_epi8(source, _mm_setzero_si128());
            sss0 = _mm_add_epi32(sss0, _mm_madd_epi16(pix, mmk));
            pix = _mm_unpackhi_epi8(source, _mm_setzero_si128());
            sss1 = _mm_add_epi32(sss1, _mm_madd_epi16(pix, mmk));
        }
        sss0 = _mm_srai_epi32(sss0, coefs_precision);
        sss1 = _mm_srai_epi32(sss1, coefs_precision);

        sss0 = _mm_packs_epi32(sss0, sss1);
        sss0 = _mm_packus_epi16(sss0, sss0);
        _mm_storel_epi64((__m128i *) &lineOut[xx], sss0);
    }

    for (; xx < xsize; xx++) {
        __m128i sss = initial;
        x = 0;
        for (; x < xmax - 1; x += 2) {
            __m128i source, source1, source2;
            __m128i pix, mmk;
            
            // Load two coefficients at once
            mmk = _mm_set1_epi32(*(INT32 *) &k[x]);

            source1 = _mm_cvtsi32_si128(  // top line
                *(int *) &imIn[x + xmin][xx]);
            source2 = _mm_cvtsi32_si128(  // bottom line
                *(int *) &imIn[x + 1 + xmin][xx]);
            
            source = _mm_unpacklo_epi8(source1, source2);
            pix = _mm_unpacklo_epi8(source, _mm_setzero_si128());
            sss = _mm_add_epi32(sss, _mm_madd_epi16(pix, mmk));
        }
        for (; x < xmax; x++) {
            __m128i pix = mm_cvtepu8_epi32(&imIn[x + xmin][xx]);
            __m128i mmk = _mm_set1_epi32(k[x]);
            sss = _mm_add_epi32(sss, _mm_madd_epi16(pix, mmk));
        }
        sss = _mm_srai_epi32(sss, coefs_precision);
        sss = _mm_packs_epi32(sss, sss);
        lineOut[xx] = _mm_cvtsi128_si32(_mm_packus_epi16(sss, sss));
    }
}

void resampleVerticalSIMD(cv::Mat& im_out,
                          const cv::Mat& im_in,
                          int ksize,
                          const std::vector<int>& bounds,
                          const std::vector<double>& prekk)
{
    std::vector<int16_t> kk_16;
    int ysize = static_cast<int>(im_out.size().height);
    int xsize = static_cast<int>(im_out.size().width);
    int coefs_precision =
        normalize_coeffs_8bpc_original(ysize, ksize, prekk.data(), kk_16);
    std::vector<UINT32*> imPtr;
    for (int i = 0; i < im_in.size().height; ++i)
        imPtr.push_back((UINT32*)im_in.ptr<cv::Vec4b>(i));
    for (int yy = 0; yy < ysize; yy++) {
        int16_t* k = &kk_16[yy * ksize];
        int ymin = bounds[yy * 2 + 0];
        int ymax = bounds[yy * 2 + 1];
        ImagingResampleVerticalConvolution8u((UINT32*)im_out.ptr<cv::Vec4b>(yy),
                                             imPtr.data(), im_in.size().width,
                                             ymin, ymax, k, coefs_precision);
    }
}