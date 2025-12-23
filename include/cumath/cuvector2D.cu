#ifndef VC3_CUMATH_CUVECTOR2D
#define VC3_CUMATH_CUVECTOR2D

#include "../types.cu"

#include <math.h>
#include "cumath_consts.cu"

namespace vc3_cumath {
namespace planar {

struct cuvector
{
	flt2 x, y;


    /** Constructor
	**/
	__host__ __device__ cuvector(flt2 c=0.00) throw() {x=y=c;}


	/** Constructor
	**/
	__host__ __device__ cuvector(flt2 vx, flt2 vy) throw() {x=vx;y=vy;}

	__host__ __device__ flt2 length() const throw()
	{
	    flt2 sq=x*x+y*y;
        return sq==0.00?(0.00):(sqrt(sq));
	}

	__host__ __device__ inline flt2 length2() const throw()
	{
	    return x*x+y*y;
	}
	
	__host__ __device__ flt2 phi() const throw()
	{
		flt2 r = length();
		if (r == 0.00) return 0.00;
		flt2 phi = 0.00;
		flt2 xr = x / r;
		if (xr < 1.00)
		{
			if (xr > -1.00)
			{
				phi = acos(xr);
				if (y < 0.00) phi = TwoPi - phi;
			}
			else phi = Pi;
		}
		return phi;
	}

	__host__ __device__ flt2 phi_unitVectorsOnly() const throw()
	{
		flt2 phi = 0.00;
		if (x < 1.00)
		{
			if (x > -1.00)
			{
				phi = acos(x);
				if (y < 0.00) phi = TwoPi - phi;
			}
			else phi = Pi;
		}
		return phi;
	}

	__host__ __device__ int normalize(flt2 n=1.00) throw()
	{
		flt2 r=length();
		if(r==0.00) return 1;
		r/=n;
		x/=r;
		y/=r;
		return 0;
	}

	__host__ __device__ cuvector & operator+=(const cuvector &b) throw()
	{
		x+=b.x;
		y+=b.y;
		return *this;
	}

	__host__ __device__ cuvector & operator*=(flt2 c) throw()
	{
		x*=c;
		y*=c;
		return *this;
	}

	__host__ __device__ cuvector operator+(const cuvector &b) const throw() {return cuvector(x+b.x, y+b.y);}

	__host__ __device__ cuvector operator-(const cuvector &b) const throw() {return cuvector(x-b.x, y-b.y);}

	__host__ __device__ cuvector operator*(flt2 c) const throw() {return cuvector(x*c, y*c);}

	__host__ __device__ cuvector operator/(flt2 c) const throw()
	{
		/* if(c==0.00) throw v3_exc::bad_value(); */
		return cuvector(x/c, y/c);
	}
}; // struct cuvector

__host__ __device__ cuvector operator*(flt2 c, const cuvector a) throw()
{
    return cuvector(a.x*c, a.y*c);
}

__host__ __device__ inline flt2 SP(const cuvector &a, const cuvector &b) throw()
{
    return a.x*b.x+a.y*b.y;
}

} //namespace planar
} //namespace vc3_cumath


#endif // VC3_CUMATH_CUVECTOR2D
