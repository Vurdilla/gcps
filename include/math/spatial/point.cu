#ifndef VC3_MATH_SPATIAL_POINT
#define VC3_MATH_SPATIAL_POINT

#include "../../types.cu"

#include "../basic_math.cu"
#include <ostream>
#include <istream>

namespace vc3_math {
namespace spatial {

bool _usepointproximity=true;

// Spatial point proximity parameter
// - distance at which spatial points will be treaten as equivalent
const flt spatial_proximity=1.00e-8;

// Spatial point
struct point
{
	flt x, y, z;

	// Compares this and point a, returns true if this and a are the same point,
	// false if not
	// Note it operates on basis of spatial_proximity !!!
	bool operator==(const point &a) const throw();

	inline flt distance(const point &a) const;

	// Returns square of distance
	inline flt distance2(const point &a) const ;

}; // struct point

std::ostream & operator<<(std::ostream &ostr, const point &p)
{
	ostr<<p.x<<"\t"<<p.y<<"\t"<<p.z;
	return ostr;
}

std::istream & operator>>(std::istream &istr, point &p)
{
	istr>>p.x>>p.y>>p.z;
	return istr;
}





} //namespace spatial
} //namespace vc3_math


//=================================================
//Realization is here as joint template compilation is not supported

bool vc3_math::spatial::point::operator==(const point &a) const
	throw()
{
	if(_usepointproximity)
	{
		if(fabs(x-a.x)>spatial_proximity || fabs(y-a.y)>spatial_proximity
			|| fabs(z-a.z)>spatial_proximity) return false;
		return true;
	}
	else
	{
		if(x-a.x!=0.00 || y-a.y!=0.00 || z-a.z!=0.00) return false;
		return true;
	}
}

inline flt vc3_math::spatial::point::distance(const point &a) const
{
	return msqrt( (a.x-x)*(a.x-x)+(a.y-y)*(a.y-y)+(a.z-z)*(a.z-z) );
}

inline flt vc3_math::spatial::point::distance2(const point &a) const
{
	return (a.x-x)*(a.x-x)+(a.y-y)*(a.y-y)+(a.z-z)*(a.z-z);
}

#endif // #ifndef VC3_MATH_SPATIAL_POINT

