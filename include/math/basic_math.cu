#ifndef VC3_MATH
#define VC3_MATH

#include <math.h>

#include "../types.cu"

namespace vc3_math
{

/** for size_t
**/

inline size_t max(size_t a, size_t b) throw()
{
	return a>=b ? a : b;
}

inline size_t min(size_t a, size_t b) throw()
{
	return a<=b ? a : b;
}


/** for unsigned type
**/

inline unsigned max(unsigned a, unsigned b) throw()
{
	return a>=b ? a : b;
}

inline unsigned min(unsigned a, unsigned b) throw()
{
	return a<=b ? a : b;
}


/** for int type
**/

inline int sign(int a) throw()
{
    return a>=0 ? 1 : -1;
}

inline int max(int a, int b) throw()
{
	return a>=b ? a : b;
}

inline int min(int a, int b) throw()
{
	return a<=b ? a : b;
}


/** for flt type
**/

inline flt sign(flt a) throw()
{
    return a>=0.00 ? 1.00 : -1.00;
}

inline flt max(flt a, flt b) throw()
{
	return a>=b ? a : b;
}

inline flt min(flt a, flt b) throw()
{
	return a<=b ? a : b;
}

inline flt fabs(flt a) throw()
{
	return a>=0.00 ? a : -a;
}

inline flt msqrt(flt d) throw()
{
	return d==0.00 ? 0.00 : sqrt(d);
}

inline flt smsqrt(flt d) throw()
{
	return d<=0.00 ? 0.00 : sqrt(d);
}

} // namespace vc3_math;


#endif
