#ifndef V3_MATH_STAT_AVGERRMINMAX
#define V3_MATH_STAT_AVGERRMINMAX

#include "../../types.cu"
#include "../basic_math.cu"

namespace vc3_math {
namespace stat{


class AvgErrMinMax
{
	public:

	// Default constructor
		AvgErrMinMax() throw();

	// Nulls distribution statistics
	void reset() throw();

	// Adds point to the statistics
	void add(flt2 x) throw();

	// Returns average value
	int getavg(flt2 *avg) const throw();

	// Returns average and error value
	int getavg(flt2* avg, flt2* err) const throw();

	// Returns error value
	int geterr(flt2* err) const throw();

	// Returns variance value
	int getvar(flt2* var) const throw();

	// Returns min value
	flt2 getmin() const throw();

	// Returns max value
	flt2 getmax() const throw();

	// Returns number of data used in distribution
	int getamount() const throw();

	// Merge two datasets
	void merge(const AvgErrMinMax& a) throw();


	//**********************************************************************
	protected:

    int _npoints; // Number of data points used
	flt2 _sum, _sum2; // Sum of all points and sum squared
	flt2 _min, _max; // Min and max values;
}; //class AvgErrMinMax

AvgErrMinMax::AvgErrMinMax()
    throw():
    _npoints(0), _sum(0.00), _sum2(0.00), _min(0.00), _max(0.00)
{

}

void AvgErrMinMax::reset()
	throw()
{
	_npoints=0;
	_sum=0.00;
	_sum2=0.00;
	_min=0.00;
	_max=0.00;
}

void AvgErrMinMax::add(flt2 x)
		throw()
{
    if(_npoints==0)
    {
        _min=x;
        _max=x;
    }
	_npoints++;
	_sum+=x;
	_sum2+=x*x;
	_min = _min<=x?_min:x;
	_max = _min>=x?_max:x;
}

int AvgErrMinMax::getavg(flt2 *avg) const throw()
{
	if(_npoints==0) return 0;
	*avg=_sum/flt2(_npoints);
	return _npoints;
}

int AvgErrMinMax::getavg(flt2* avg, flt2* err) const throw()
{
	if (_npoints==0) return 0;
	*avg=_sum/flt2(_npoints);
	if (_npoints==1) *err=0.00;
	else *err=msqrt( (_sum2-_sum*_sum/flt2(_npoints)) / (flt2(_npoints-1)) );
	return _npoints;
}

int AvgErrMinMax::geterr(flt2* err) const throw()
{
	if (_npoints == 0) return 0;
	if (_npoints == 1) *err = 0.00;
	else *err = msqrt((_sum2 - _sum * _sum / flt2(_npoints)) / (flt2(_npoints - 1)));
	return _npoints;
}

int AvgErrMinMax::getvar(flt2* var) const throw()
{
	if (_npoints == 0) return 0;
	if (_npoints == 1) *var = 0.00;
	else *var = (_sum2 - _sum * _sum / flt2(_npoints)) / flt2(_npoints);
	return _npoints;
}

flt2 AvgErrMinMax::getmin() const
		throw()
{
	return _min;
}

flt2 AvgErrMinMax::getmax() const
		throw()
{
	return _max;
}

int AvgErrMinMax::getamount() const
		throw()
{
	return _npoints;
}

void AvgErrMinMax::merge(const AvgErrMinMax& a) throw()
{
	_sum += a._sum;
	_sum2 += a._sum2;
	if (_npoints == 0)
	{
		_min = a._min;
		_max = a._max;
	}
	else if(a._npoints > 0)
	{
		_min = _min <= a._min ? _min : a._min;
		_max = _max >= a._max ? _max : a._max;
	}
	_npoints += a._npoints;
}

} //namespace stat
} //namespace vc3_math

#endif //#ifndef V3_MATH_STAT_AVGERRMINMAX
