#ifndef VC3_CUMATH_CUBASICMATH
#define VC3_CUMATH_CUBASICMATH

#include "../types.cu"

#include <math.h>
#include "cumath_consts.cu"

namespace vc3_cumath {

__host__ __device__ inline flt msqrt(flt x) throw()
{
    return x > 0.00 ? sqrt(x) : 0.00;
}

__host__ __device__ inline flt2 msqrt_flt2(flt2 x) throw()
{
    return x > 0.00 ? sqrt(x) : 0.00;
}

__host__ __device__ inline flt normdist(flt x, flt mu, flt sigma) throw()
{
    return OneDivSqrt2Pi / sigma * exp(-(x - mu) * (x - mu) / 2.00 / sigma / sigma);
}

__host__ __device__ inline flt2 normdist(flt2 x, flt2 mu, flt2 sigma) throw()
{
    return OneDivSqrt2Pi / sigma * exp(-(x - mu) * (x - mu) / 2.00 / sigma / sigma);
}

__host__ __device__ inline flt2 normdistcopt(flt2 x2, flt2 sigmainv, flt2 sigma2inv) throw()
{
    return OneDivSqrt2Pi * sigmainv * exp(- x2 * sigma2inv * 0.50);
}

__host__ __device__ inline flt normdistcoptf(flt x2, flt sigmainv, flt sigma2inv) throw()
{
    return OneDivSqrt2Pi * sigmainv * expf(-x2 * sigma2inv * 0.50);
}

} //namespace vc3_cumath


#endif // VC3_CUMATH_CUBASICMATH
