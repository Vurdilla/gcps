#ifndef VC3_MATH_RANDGEN_UNIFORM_RAN2
#define VC3_MATH_RANDGEN_UNIFORM_RAN2

#include <stdlib.h>
#include <cstdlib>
#include <ctime>

namespace vc3_math {

/** Random number generator
    Based on ran2 routine (see Numerical Recipes in C, p.281)

Long period (>2x10^18) random number generator of L'Ecuyer with Bays-Durhem shuffle and added safeguards. Returns a uniform random deviate between 0.00 and 1.00 (exclusive of the endpoint value). Call with idum a negative number to initialize; thereafter, do not alter idum between successive derivates in sequence. RNMX should approximate the largest floating value that is less than 1.
**/

/** ORIGINAL CODE FROM "NUMERICAL RECIPES IN C", p.281

#define IM1 2147483563
#define IM2 2147483399
#define AM (1.0/IM1)
#define IMM1 (IM1-1)
#define IA1 40014
#define IA2 40692
#define IQ1 53668
#define IQ2 52774
#define IR1 12211
#define IR2 3791
#define NTAB 32
#define NTAB 32
#define NDIV (1+IMM1/NTAB)
#define EPS 1.2e-7
#define RNMX (1.0-EPS)

float ran2(long *idum)
{
    int j;
    long k;
    static long idum2=123456789;
    static long iy=0;
    static long iv[NTAB];
    float temp;

    // Initialize
    if(*idum<=0)
    {
        if(-(*idum)<1) *idum=1;
        else *idum=-(*idum);
        idum2=(*idum);
        for(j=NTAB+7;j>=0;j--)
        {
            k=(*idum)/IQ1;
            *isum=IA1*(*isum-k*IQ1))-k*IR1;
            if(*idum<0) *idum+=IM1;
            if(j<NTAB) iv[j]=*idum;
        }
        iy=iv[0];
    } // End of initialization
    k=(*idum)/IQ1;
    *idum=IA1*(*idum-k*IQ1)-k*IR1;
    if(*idum<0) *idum+=IM1;
    k=idum2/IQ2;
    idum2=IA2*(idum2-k*IQ2)-k*IR2;
    if(idum2<0) idum2+=IM2;
    j=iy/NDIV;
    iy=iv[j]-idum2;
    iv[j]=*idum;
    if(iy<1) iy_-IMM1;
    if((temp=AM*iy)>RNMX) return RNMX;
    else return temp;
}


**/


class randgen_uniform_ran2
{
	public:
	// Constructor
	randgen_uniform_ran2(unsigned int plusseed=0) throw()
    {
        //std::randomize();
        std::srand(std::time(0) + plusseed);
        _idum = -std::rand() % 321458765;
        _iy = 0;
        _iv = new long[32];

        if (-(_idum) < 1) _idum = 1;
        else _idum = -(_idum);
        _idum2 = (_idum);
        for (int j = 32 + 7; j >= 0; j--)
        {
            long k = _idum / 53668;
            _idum = 40014 * (_idum - k * 53668) - k * 12211;
            if (_idum < 0) _idum += 2147483563;
            if (j < 32) _iv[j] = _idum;
        }
        _iy = _iv[0];
    }

	// Generate uniform random number on segment [0.00; 1.00)
	virtual float segment_generate() throw()
    {
        long k = (_idum) / 53668;
        _idum = 40014 * (_idum - k * 53668) - k * 12211;
        if (_idum < 0) _idum += 2147483563;
        k = _idum2 / 52774;
        _idum2 = 40692 * (_idum2 - k * 52774) - k * 3791;
        if (_idum2 < 0) _idum2 += 2147483399;
        int j = _iy / (1 + 2147483562 / 32);
        _iy = _iv[j] - _idum2;
        _iv[j] = _idum;
        if (_iy < 1) _iy += 2147483562;
        if ((temp = (1.00 / 2147483563) * _iy) > (1.00 - 1.20e-7)) return (1.00 - 1.20e-7);
        else return temp;
    }

	// Destructor
	~randgen_uniform_ran2() throw()
    {
        delete[] _iv;
    }

	//=====================================================================
	private:
	long _iy;
	long *_iv;
	long _idum, _idum2;
	float temp;

};	//class randgen_uniform_ran2

} //namespace vc3_math

#endif // VC3_MATH_RANDGEN_UNIFORM_RAN2

