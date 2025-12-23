#ifndef VC3_MATH_ARRAY
#define VC3_MATH_ARRAY

#include "oarray.cu"

namespace vc3_exc
{
	class incompatible_sizes: public bad_size
	{
		public:
		incompatible_sizes(std::string s="Error: operation requires arrays of the same size")
			{name=s;}
	};

	class self_handling: public bad_data
	{
		public:
		self_handling(std::string s="Error: Object tries to change copy of itself")
			{name=s;}
    };
} //namespace vc3_exc


namespace vc3_math {

// Array class: Array of objects with basic math operations defined
// (=, ==, *=, +=, -=)
template<typename FLT> class Array: public OArray<FLT>
{
	public:

	// Default Constructor. Creates an nElem array with unknown values.
	// If number of elements is not specified, it is set to 1.
	Array(int nElem=1) throw(vc3_exc::bad_alloc, vc3_exc::bad_index):
		OArray<FLT>(nElem) {};

	// Constructor. Creates nElem array
	// with all the same values equal to value; no default value.
	Array(int nElem, FLT value) throw(vc3_exc::bad_alloc, vc3_exc::bad_index):
		OArray<FLT>(nElem,value) {};

	// Copy Constructor.
	// Used when a copy of an object is produced
	Array(const Array<FLT> &a) throw(vc3_exc::bad_alloc):
		OArray<FLT>(a) {};

	// Multiplication operator. Returns reference to the left operand.
	Array<FLT> & operator*=(FLT k) noexcept;

	// Addition operator. Returns reference to the left operand.
	Array<FLT> & operator+=(const Array<FLT> &a) throw(vc3_exc::self_handling, vc3_exc::incompatible_sizes);

	// Subtraction operator. Returns reference to the left operand.
	Array<FLT> & operator-=(const Array<FLT> &a) throw(vc3_exc::self_handling, vc3_exc::incompatible_sizes);

}; //class Array

template<typename FLT>
FLT mult(const Array<FLT> &left, const Array<FLT> &right)
		throw(vc3_exc::incompatible_sizes);

} //namespace vc3_math


//*************************************************
//Realization is here as joint template compilation is not supported

template<class FLT>
vc3_math::Array<FLT> & vc3_math::Array<FLT>::operator*=(FLT k)
		noexcept
{
	for(int i=0;i<OArray<FLT>::nElem_;i++) OArray<FLT>::data_[i]*=k;
	return *this;
}

template<class FLT>
vc3_math::Array<FLT> & vc3_math::Array<FLT>::operator+=(const Array<FLT> &a)
		throw(vc3_exc::self_handling, vc3_exc::incompatible_sizes)
{
	if(this==&a) throw vc3_exc::self_handling("Array::operator+=(..) self handling");
	if(OArray<FLT>::nElem_!=a.nElem_) throw vc3_exc::incompatible_sizes();
	for(int i=0;i< OArray<FLT>::nElem_;i++) OArray<FLT>::data_[i]+=a.data_[i];
	return *this;
}

template<class FLT>
vc3_math::Array<FLT> & vc3_math::Array<FLT>::operator-=(const Array<FLT> &a)
		throw(vc3_exc::self_handling, vc3_exc::incompatible_sizes)
{
	if(this==&a) throw vc3_exc::self_handling("Array::operator-=(..) self handling");
	if(OArray<FLT>::Elem_!=a.nElem_) throw vc3_exc::incompatible_sizes();
	for(int i=0;i< OArray<FLT>::nElem_;i++) OArray<FLT>::data_[i]-=a.data_[i];
	return *this;
}

template<typename FLT>
FLT vc3_math::mult(const Array<FLT> &left, const Array<FLT> &right)
		throw(vc3_exc::incompatible_sizes)
{
	if(left.nElem() != right.nElem()) throw vc3_exc::incompatible_sizes();
	FLT m(0.0);
	for(int nR=0;nR<left.nElem();nR++)
		m+=left(nR)*right(nR);
	return m;
}

#endif //#ifndef VC3_MATH_ARRAY
