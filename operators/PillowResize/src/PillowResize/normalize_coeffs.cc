#include "ImagingSIMD.h"

int normalize_coeffs_8bpc_original(int outSize, int ksize, const double *prekk, std::vector<int16_t>& kk)
{
    static constexpr unsigned int precision_bits = 32 - 8 - 2;

    int x;
    int coefs_precision;
    double maxkk;

    maxkk = prekk[0];
    for (x = 0; x < outSize * ksize; x++)
    {
        if (maxkk < prekk[x])
        {
            maxkk = prekk[x];
        }
    }

    for (coefs_precision = 0; coefs_precision < precision_bits; coefs_precision += 1)
    {
        int next_value = (int)(0.5 + maxkk * (1 << (coefs_precision + 1)));
        // The next value will be outside of the range, so just stop
        if (next_value >= (1 << 15))
            break;
    }

    for (x = 0; x < outSize * ksize; x++)
    {
        if (prekk[x] < 0)
        {
            kk.push_back( (int)(-0.5 + prekk[x] * (1 << coefs_precision)) );
        }
        else
        {
            kk.push_back( (int)(0.5 + prekk[x] * (1 << coefs_precision)) );
        }
    }
    return coefs_precision;
}