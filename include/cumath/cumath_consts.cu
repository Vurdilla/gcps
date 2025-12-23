#ifndef VC3_CUMATH_CONSTS
#define VC3_CUMATH_CONSTS

#include "../types.cu"

namespace vc3_cumath {

	/** Pi */
	const flt2 HalfPi = 1.57079632679489662;
	const flt2 Pi = 3.14159265358979324;
	const flt2 TwoPi = 6.28318530717958648;
	/** Student Coefficient for 95% probability **/
	const flt2 Students_Coeff_95 = 1.96;
	/** 1/sqrt(2*Pi) **/
	const flt2 OneDivSqrt2Pi = 0.39894228040143267;
	/** Finite metrics **/
	const flt2 ZEROLENGT = 0.00000001;


} //namespace vc3_cumath


#endif // VC3_CUMATH_CONSTS
