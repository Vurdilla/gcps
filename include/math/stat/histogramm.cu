#ifndef VC3_MATH_STAT_HISTOGRAMM
#define VC3_MATH_STAT_HISTOGRAMM

#include "../containers/oarray.cu"
#include <fstream>
#include <string>
#include <vector>

namespace vc3_math {

enum HISTMODE
{
	HISTMODE_CUTOFF = 0,		// In this mode point out of histogramm interval
	// will been accounted in distribution
	HISTMODE_EXT = 1,		// In this mode point out of historgamm interval
	// will been added to the left or to the right column
	HISTMODE_MINEXT = 2,		// In this mode point left to the historgamm interval
	// will been added to the left column
	HISTMODE_MAXEXT = 3		// In this mode point right to the historgamm interval
	// will been added to the right column
};

//
template<typename FLT> class Histogramm
{
	public:

	// Default constructor
	Histogramm(FLT xmin=0.00, FLT xmax=1.00, int nbins=10,
				HISTMODE mode=HISTMODE_CUTOFF)
		throw(/*v3_exc::bad_index, v3_exc::bad_value*/);

	// Nulls distribution statistics
	void reset() throw();

	// Changes histogramm parameters (automatically nulls statistics)
	void change(FLT xmin, FLT xmax, int nbins,
				HISTMODE mode=HISTMODE_CUTOFF)
		throw(/*v3_exc::bad_index, v3_exc::bad_value*/);

	// Adds point to the statistics
	void add(FLT x) throw();

	// Merge current data with one in new histogramm with current weight of 1 and new weight of w
	void merge(const Histogramm& h, FLT w = 1.00)
		throw(/*vc3_exc::bad_data*/);

	// Writes distribution
	void gethist(OArray<FLT> *hist) const 
		throw();

	// Writes histogram parameters
	void getparams(FLT* xmin, FLT* dx, HISTMODE* mode, int* amountofdata) const 
		throw();

	// Returns number of data used in distribution
	int getamount() const 
		throw();

	FLT getpercentile(FLT p, bool interpolate = true) const 
		throw(/*v3_exc::bad_value, v3_exc::bad_data*/);

	// Write
	// xmode: 0 - print bin center, 1 - print bin boundaries, 2 - print bin boundaries and center
	// norm: 0 - print counts * k, 1 - normalization & *k (i.e. print counts / npoints * k), 2 - normalization to PDF & *k, 3 - normalization to PDF with log10 X scale & *k
	void write(std::ostream& f, bool humanreadable = false, int xmode = 0, bool product = false, int norm = 0, FLT k = 1.00) const 
		throw();

	// Read
	void read(std::istream& f) throw();


	//**********************************************************************
	protected:

	long int npoints_;      			// Number of data points used
	FLT xmin_;						// Left edge of interval
	FLT dx_; 						// Thickness of columns
	HISTMODE mode_;						// Mode
	OArray<FLT> distr_;				// Distribution function
}; //class Histogramm

} //namespace vc3_math


//*************************************************
//Realization is here as joint template compilation is not supported

template<typename FLT>
vc3_math::Histogramm<FLT>::Histogramm(FLT xmin, FLT xmax,
	int nhist, HISTMODE mode)
		throw(/*v3_exc::bad_index, v3_exc::bad_value*/):
		npoints_(0), xmin_(xmin), dx_((xmax-xmin)/FLT(nhist)), mode_(mode),
		distr_(nhist, 0.00)
{
	//if(xmin>=xmax) throw v3_exc::bad_value();
}

template<typename FLT>
void vc3_math::Histogramm<FLT>::reset()
	throw()
{
	npoints_=0;
	distr_.set(0.00);
}

template<typename FLT>
void vc3_math::Histogramm<FLT>::change(FLT xmin, FLT xmax,
	int nhist, HISTMODE mode)
		throw(/*v3_exc::bad_index, v3_exc::bad_value*/)
{
	//if(xmin>=xmax) throw v3_exc::bad_value();
	npoints_=0;
	xmin_=xmin;
	dx_=(xmax-xmin)/FLT(nhist);
	mode_=mode;
	distr_.resize(nhist);
	distr_.set(0.00);
}

template<typename FLT>
void vc3_math::Histogramm<FLT>::add(FLT x)
		throw()
{
	FLT fx = (x - xmin_) / dx_;
	int nx = fx;
	if (fx - FLT(nx) < 0.00) nx--;
	if (nx < 0)
	{
		switch (mode_)
		{
		case HISTMODE_CUTOFF:
		case HISTMODE_MAXEXT:
			return;
			break;
		case HISTMODE_EXT:
		case HISTMODE_MINEXT:
			npoints_++;
			distr_(0)++;
			return;
			break;
		}
	}
	if (nx >= distr_.nElem())
	{
		switch (mode_)
		{
		case HISTMODE_CUTOFF:
		case HISTMODE_MINEXT:
			return;
			break;
		case HISTMODE_EXT:
		case HISTMODE_MAXEXT:
			npoints_++;
			distr_(distr_.nElem() - 1)++;
			return;
			break;
		}
	}
	npoints_++;
	distr_(nx)++;
}

template<typename FLT>
void vc3_math::Histogramm<FLT>::merge(const vc3_math::Histogramm<FLT>& h, FLT w)
	throw(/*v3_exc::bad_data*/)
{
	if (xmin_ != h.xmin_ || dx_ != h.dx_ || mode_ != h.mode_ || distr_.nElem() != h.distr_.nElem())
		return; // throw v3_exc::bad_data();

	npoints_ += h.npoints_;
	for (int q = 0; q < distr_.nElem(); q++)
	{
		distr_(q) += h.distr_(q) * w;
	}
}

template<typename FLT>
void vc3_math::Histogramm<FLT>::gethist(OArray<FLT> *hist) const
	throw()
{
	*hist=distr_;
}

template<typename FLT>
void vc3_math::Histogramm<FLT>::getparams(FLT* xmin, FLT* dx, HISTMODE* mode, int* amountofdata) const
	throw()
{
	*xmin = xmin_;
	*dx = dx_;
	*mode = mode_;
	*amountofdata = npoints_;
}

template<typename FLT>
int vc3_math::Histogramm<FLT>::getamount() const
		throw()
{
	return npoints_;
}

template<typename FLT>
FLT vc3_math::Histogramm<FLT>::getpercentile(FLT p, bool interpolate) const
	throw(/*v3_exc::bad_value, v3_exc::bad_data*/)
{
	FLT pcount = p * FLT(npoints_);
	int ppos = -1;
	FLT hcount = 0.00;
	for (int q = 0; q < distr_.nElem() && hcount < pcount; q++)
	{
		hcount += distr_(q);
		ppos++;
	}
	if (ppos == -1) return xmin_;
	if (ppos < distr_.nElem())
	{
		if (interpolate)
		{
			FLT hcountprev = hcount - distr_(ppos);
			FLT k = (pcount - hcountprev) / (hcount - hcountprev);
			FLT px = xmin_ + dx_ * (FLT(ppos - 1) + k);
			return px;
		}
		else return xmin_ + dx_ * FLT(ppos);
	}
	else
	{
		return xmin_ + dx_ * distr_.nElem();
	}
}

template<typename FLT>
void vc3_math::Histogramm<FLT>::write(std::ostream& f, bool humanreadable, int xmode, bool product, int norm, FLT k) const
	throw()
{
	/*
	long int npoints_;      			// Number of data points used
	FLT xmin_;						// Left edge of interval
	FLT dx_; 						// Thickness of columns
	HISTMODE mode_;						// Mode
	Array<FLT> distr_;				// Distribution function
	*/
	if (!humanreadable)
	{
		f << "\nnpoints\t" << npoints_ << "\nxmin\t" << xmin_ << "\ndx\t" << dx_ << "\nhistmode\t";
		switch (mode_)
		{
		case HISTMODE_CUTOFF:
			f << 0;
			break;
		case HISTMODE_EXT:
			f << 1;
			break;
		case HISTMODE_MINEXT:
			f << 2;
			break;
		case HISTMODE_MAXEXT:
		default:
			f << 3;
		}
		f << "\ndistr\t" << distr_.nElem() << "\n";
		for (int q = 0; q < distr_.nElem(); q++) f << "\t" << distr_(q);
	}
	else // humanreadable==true
	{
		f << "Data\t" << npoints_;
		double prod = 0;
		for (int q = 0; q < distr_.nElem(); q++)
		{
			if (xmode == 0) f << "\n" << xmin_ + dx_ * (0.50 + FLT(q));
			if (xmode == 1) f << "\n" << xmin_ + dx_ * FLT(q) << "\t" << xmin_ + dx_ * FLT(q + 1);
			if (xmode == 2) f << "\n" << xmin_ + dx_ * FLT(q) << "\t" << xmin_ + dx_ * FLT(q + 1) << "\t" << xmin_ + dx_ * (0.50 + FLT(q));

			if (norm == 0) // 0 - print counts * k
			{
				f << "\t" << distr_(q) * k;
				prod += distr_(q);
				if (product) f << "\t" << prod * k;
			}
			if (norm == 1) // 1 - normalization & *k (i.e. print counts / npoints * k)
			{
				f << "\t" << distr_(q) * k;
				f << "\t" << double(distr_(q)) / double(npoints_) * k;
				prod += distr_(q);
				if (product) f << "\t" << prod / double(npoints_) * k;

			}
			if (norm == 2) // 2 - normalization to PDF & *k
			{
				f << "\t" << distr_(q) * k;
				f << "\t" << double(distr_(q)) / double(npoints_) / dx_ * k;
				prod += distr_(q);
				if (product) f << "\t" << prod / double(npoints_) * k;
			}
			if (norm == 3) // 3 - normalization to PDF with log10 X scale & *k
			{
				f << "\t" << distr_(q) * k;
				double dx = pow(10, xmin_ + dx_ * FLT(q + 1)) - pow(10, xmin_ + dx_ * FLT(q));
				f << "\t" << double(distr_(q)) / double(npoints_) / dx * k;
				prod += distr_(q);
				if (product) f << "\t" << prod / double(npoints_) * k;
			}
		}
		f << "\n";
	}
}

template<typename FLT>
void vc3_math::Histogramm<FLT>::read(std::istream& f)
	throw()
{
	/*
	long int npoints_;      			// Number of data points used
	FLT xmin_;						// Left edge of interval
	FLT dx_; 						// Thickness of columns
	HISTMODE mode_;						// Mode
	Array<FLT> distr_;				// Distribution function
	*/
	std::string s;
	int n;
	f >> s >> npoints_;
	f >> s >> xmin_;
	f >> s >> dx_;
	f >> s >> n;
	switch (n)
	{
	case 0:
		mode_ = HISTMODE_CUTOFF;
		break;
	case 1:
		mode_ = HISTMODE_EXT;
		break;
	case 2:
		mode_ = HISTMODE_MINEXT;
		break;
	case 3:
	default:
		mode_ = HISTMODE_MAXEXT;
	}
	f >> s >> n;
	distr_.resize(n);
	for (int q = 0; q < n; q++) f >> distr_(q);
}

#endif //#ifndef VC3_MATH_STAT_HISTOGRAMM
