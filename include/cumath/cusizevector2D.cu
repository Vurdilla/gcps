#ifndef VC3_CUMATH_CUSIZEVECTOR2D
#define VC3_CUMATH_CUSIZEVECTOR2D

namespace vc3_cumath {
namespace planar {

struct cusizevector
{
	int x, y;

    /** Default constructor
    **/
	__host__ __device__ cusizevector() throw() {x=y=0;}


	/** Constructor
	*/
	__host__ __device__ cusizevector(int nx, int ny) throw()
    {
        x=nx;y=ny;
    }

	__host__ __device__ cusizevector operator+(const cusizevector &b) const throw()
	{
        return cusizevector(x+b.x, y+b.y);
    }

    __host__ __device__ cusizevector operator+(const int &c) const throw()
	{
        return cusizevector(x+c, y+c);
    }

    __host__ __device__ cusizevector operator*(const int &c) const throw()
	{
        return cusizevector(x*c, y*c);
    }
}; // struct cusizevector


} //namespace planar
} //namespace vc3_cumath


#endif // VC3_CUMATH_CUSIZEVECTOR2D
