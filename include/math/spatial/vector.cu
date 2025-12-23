#ifndef VC3_MATH_SPATIAL_VECTOR
#define VC3_MATH_SPATIAL_VECTOR

#include "../../types.cu"

#include "../basic_math.cu"
#include "../math_consts.cu"
#include "point.cu"
#include <ostream>
#include <istream>

namespace vc3_math {
namespace spatial {

struct vector
{
	flt x, y, z;

	vector() throw() {}
	vector(flt vx, flt vy, flt vz) throw() {x=vx;y=vy;z=vz;}

	vector(const point &begin, const point &end) throw()
	{
		x=end.x-begin.x;
		y=end.y-begin.y;
		z=end.z-begin.z;
	}

	void set(const point &begin, const point &end) throw()
	{
		x=end.x-begin.x;
		y=end.y-begin.y;
		z=end.z-begin.z;
	}

	void normalize() throw(/*v3_exc::bad_data*/)
	{
		flt r=length();
		//if(r==0.00) throw v3_exc::bad_data(); // Couldn't normalize zero length vector
		x/=r;
		y/=r;
		z/=r;
	}

	vector & operator*=(flt c) throw()
	{
		x*=c;
		y*=c;
		z*=c;
		return *this;
	}

	vector & operator+=(flt c) throw()
	{
		x+=c;
		y+=c;
		z+=c;
		return *this;
	}

	vector & operator+=(vector v) throw()
	{
		x+=v.x;
		y+=v.y;
		z+=v.z;
		return *this;
	}

	vector operator+(const vector &b) const
	throw()
	{return vector(x+b.x, y+b.y, z+b.z);}

	vector operator-(const vector &b) const
	throw()
	{return vector(x-b.x, y-b.y, z-b.z);}

	vector operator*(flt c) const
	throw()
	{return vector(x*c, y*c, z*c);}

	vector operator/(flt c) const
	throw(/*v3_exc::bad_value*/)
	{
		/*if(c==0.00) throw v3_exc::bad_value();*/
		return vector(x/c, y/c, z/c);
	}

	void CP(const vector &a, const vector &b) throw()
	{
		x=a.y*b.z-a.z*b.y;
		y=a.z*b.x-a.x*b.z;
		z=a.x*b.y-a.y*b.x;
	}

	flt length() const throw()
	{
        return msqrt(x*x+y*y+z*z);
	}

}; // struct vector

/*
vector operator*(flt c, const vector a)
	throw()
	{return vector(a.x*c, a.y*c, a.z*c);}*/

inline flt SP(const vector &a, const vector &b) throw()
	{return a.x*b.x+a.y*b.y+a.z*b.z;}

inline vector CP(const vector &a, const vector &b) throw()
{
	vector v;
	v.x=a.y*b.z-a.z*b.y;
	v.y=a.z*b.x-a.x*b.z;
	v.z=a.x*b.y-a.y*b.x;
	return v;
}

inline flt arcavg(const vector &v1, const vector &v2, vector *r)
	throw()
{
	flt v12=SP(v1,v2), k12;

	// Replace v1 with effective vector that is -v1 if v12<0.00
	// (as in director field n an -n is the same physical system)
	vector v2eff=v2;
	if(v12<0.00)
	{
	    v2eff*=-1.00;
	    v12*=-1.00;
	}

	/* The following part is reqiured only if there is no replacement of n -> -n (v1eff)

	if(v12<-0.999999)
	{
		if(v1.z>0.999999)
		{
			r->x=1.00;
			r->y=r->z=0.00;
		}
		else
		{
			flt l=sqrt(v1.x*v1.x+v1.y*v1.y);
			r->x=v1.y/l;
			r->y=-v1.x/l;
			r->z=0.00;
		}
	}
	else
	{*/
		if(v12>0.999999) *r=v1;
		else
		{
			k12=sqrt(0.50/(1.00+v12));
			*r=(v1+v2eff)*k12;
		}
	/*}*/
	return v12;
}

inline flt arcderiv(const vector &v1, const vector &v2, flt dt, vector *dvdt)
	throw()
{
	flt v12=SP(v1, v2);

	// Replace v1 with effective vector that is -v1 if v12<0.00
	// (as in director field n an -n is the same physical system)
	vector v2eff=v2;
	if(v12<0.00)
	{
	    v2eff*=-1.00;
	    v12*=-1.00;
	}

	if(v12>=1.00)
	{	// No rotation, simplest case
		dvdt->x=dvdt->y=dvdt->z=0.00;
	}
	else
	{
	    /* The following part is reqiured only if there is no replacement of n -> -n (v1eff)

		if(v12<=-1.00)
		{	// Pi rotation
			if(v1.z>0.99999999999999)
			{
				dvdt->x=Pi/dt;
				dvdt->y=dvdt->z=0.00;
			}
			else
			{
				flt l=msqrt(v1.x*v1.x+v1.y*v1.y);
				dvdt->x=v1.y/l*Pi/dt;
				dvdt->y=-v1.x/l*Pi/dt;
				dvdt->z=0.00;
            }
		}
		else
		{	// Normal calculations*/
			flt dphi=acos(v12);
			flt k2=sqrt(1.00/(1.00-v12*v12))*dphi/dt;
			flt k1=-k2*v12;
			*dvdt=v1*k1+v2eff*k2;
		/*}*/
	}
	return v12;
}

inline flt arcderiv(const vector &v1, const vector &v2, flt dt, vector *dv1dt, vector *dv2dt)
	throw()
{
	flt v12=SP(v1, v2);

	// Replace v1 with effective vector that is -v1 if v12<0.00
	// (as in director field n an -n is the same physical system)
	vector v2eff=v2;
	bool inv=false;
	if(v12<0.00)
	{
	    v2eff*=-1.00;
	    v12*=-1.00;
	    inv=true;
	}

	if(v12>=1.00)
	{	// No rotation, simplest case
		dv1dt->x=dv1dt->y=dv1dt->z=0.00;
		dv2dt->x=dv2dt->y=dv2dt->z=0.00;
	}
	else
	{
	    /* The following part is reqiured only if there is no replacement of n -> -n (v1eff)

		if(v12<=-1.00)
		{	// Pi rotation
			if(v1.z>0.99999999999999)
			{
				dv1dt->x=Pi/dt;
				dv1dt->y=dv1dt->z=0.00;
			}
			else
			{
				flt l1=msqrt(v1.x*v1.x+v1.y*v1.y);
				dv1dt->x=v1.y/l1*Pi/dt;
				dv1dt->y=-v1.x/l1*Pi/dt;
				dv1dt->z=0.00;
			}
			if(v2.z>0.99999999999999)
			{
				dv2dt->x=-Pi/dt;
				dv2dt->y=dv2dt->z=0.00;
			}
			else
			{
				flt l2=msqrt(v2.x*v2.x+v2.y*v2.y);
				dv2dt->x=-v2.y/l2*Pi/dt;
				dv2dt->y=v2.x/l2*Pi/dt;
				dv2dt->z=0.00;
			}
		}
		else
		{	// Normal calculations*/
			flt dphi=acos(v12);
			flt k2=sqrt(1.00/(1.00-v12*v12))*dphi/dt;
			flt k1=-k2*v12;
			*dv1dt=v1*k1+v2eff*k2;
			*dv2dt=(v2eff*(-k1)+v1*(-k2))*(inv?-1.00:1.00);
		/*}*/
	}
	return v12;
}

inline flt arcavg3(const vector &v1, const vector &v2, const vector &v3, vector *r) throw()
{
    vector avg12, avg13, avg23;
    flt cos12=arcavg( v1, v2, &(avg12) );
    flt cos13=arcavg( v1, v3, &(avg13) );
    flt cos23=arcavg( v2, v3, &(avg23) );
    flt maxcos=cos12*cos12>cos13*cos13?cos12:cos13;
    maxcos=maxcos*maxcos>cos23*cos23?maxcos:cos23;

    flt v1213=SP(avg12, avg13), v1223=SP(avg12, avg23);
    vector avg13eff=avg13, avg23eff=avg23;
    if(v1213<0.00) avg13eff*=-1.00;
    if(v1223<0.00) avg23eff*=-1.00;

    *r=avg12+avg13eff+avg23eff;
    r->normalize();

	return maxcos;
}


std::ostream & operator<<(std::ostream &ostr, const vector &v)
{
	ostr<<v.x<<"\t"<<v.y<<"\t"<<v.z;
	return ostr;
}

std::istream & operator>>(std::istream &istr, vector &v)
{
	istr>>v.x>>v.y>>v.z;
	return istr;
}

} //namespace spatial
} //namespace vc3_math


//=================================================
//Realization is here as joint template compilation is not supported

#endif // #ifndef VC3_MATH_SPATIAL_VECTOR


