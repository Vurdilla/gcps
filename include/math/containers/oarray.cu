#ifndef VC3_MATH_OARRAY
#define VC3_MATH_OARRAY

#include "../../general/vc3_exceptions.cu"


namespace vc3_math {

// OArray class: array of user-defined objects
// Required operations for these objects are: =, ==;
template<typename elemtype> class OArray
{
	public:

	// Default Constructor. Creates an nElem array with unknown values.
	// If number of elements is not specified, it is set to 1.
	OArray(size_t nElem=1) throw(/*v3_exc::bad_alloc, v3_exc::bad_index*/);

	// Constructor. Creates nElem array
	// with all the same values equal to value; no default value.
	OArray(size_t nElem, elemtype value) throw(/*v3_exc::bad_alloc, v3_exc::bad_index*/);

	// Copy Constructor.
	// Used when a copy of an object is produced
	OArray(const OArray<elemtype> &a) throw(/*v3_exc::bad_alloc*/);

	// Destructor. Called when a OArray object goes out of scope.
	~OArray();

	// Assignment operator function.
	// Overloads the equal sign operator to work with OArray objects.
	OArray<elemtype> & operator=(const OArray<elemtype> &a) throw(/*v3_exc::bad_alloc*/);

	// Compare operator function
	bool operator==(const OArray<elemtype> &a) const noexcept;

	// Compare operator function.
	bool operator!=(const OArray<elemtype> &a) const noexcept;

	// Compare operator function
	bool operator<(const OArray<elemtype> &a) const noexcept;

	// Resize function.
	// * Do not use it if you can avoid it, very time-consuming function.
	virtual void resize(size_t nElem) throw(/*v3_exc::bad_alloc, v3_exc::bad_index*/);

	// Simple "get" functions. Return number of elements.
	size_t nElem() const noexcept;

	// Parenthesis operator function.
	// Allows access to values of OArray via index 'i';
	// Example: a(1) = 2*b(3);
	elemtype & operator() (size_t i) throw(/*v3_exc::bad_index*/);

	// Parenthesis operator function (const version).
	const elemtype & operator() (size_t i) const throw(/*v3_exc::bad_index*/);

	// Set function. Sets all elements of a array to a given value.
	void set(elemtype value) noexcept;

	void merge(const OArray<elemtype> &a) const throw(/*v3_exc::bad_index*/);


	//*********************************************************************
	protected:
	// OArray data
	size_t nElem_; 		// Number of elements
	elemtype *data_;	// Pointer used to allocate memory for data.

}; //class OArray

} //namespace v3_math


//*************************************************
//Realization is here as joint template compilation is not supported

template<typename elemtype>
vc3_math::OArray<elemtype>::OArray(size_t nElem)
		throw(/*v3_exc::bad_alloc, v3_exc::bad_index*/)
{
	/*if(nElem<=0) throw v3_exc::bad_index();//("OArray element - out of range");*/
	nElem_=nElem;
	data_ = new elemtype[nElem];  // Allocate memory
	/*if(data_==0) throw v3_exc::bad_alloc(); // Check that memory was allocated*/
}

template<typename elemtype>
vc3_math::OArray<elemtype>::OArray(size_t nElem, elemtype value)
		throw(/*v3_exc::bad_alloc, v3_exc::bad_index*/)
{
	/*if(nElem<=0) throw v3_exc::bad_index();//("OArray element - out of range");*/
	nElem_=nElem;
	data_=new elemtype[nElem];  // Allocate memory
	/*if(data_==0) throw v3_exc::bad_alloc(); // Check that memory was allocated*/
	for(size_t i=0;i<nElem;i++) data_[i]=value;
}

template<typename elemtype>
vc3_math::OArray<elemtype>::OArray(const OArray<elemtype> &a)
		throw(/*v3_exc::bad_alloc*/)
{
	nElem_=a.nElem_;
	data_=new elemtype[nElem_];
	/*if(data_==0) throw v3_exc::bad_alloc(); // Check that memory was allocated*/
	for(size_t i=0;i<nElem_;i++) data_[i]=a.data_[i];
}

template<typename elemtype>
vc3_math::OArray<elemtype>::~OArray()
{
	if(data_!=0) delete [] data_;   // Release allocated memory
	// Else the memory was not allocated
}

template<typename elemtype>
vc3_math::OArray<elemtype> & vc3_math::OArray<elemtype>::operator=(const OArray<elemtype> &a)
		throw(/*v3_exc::bad_alloc*/)
{
	if(this==&a) return *this;	// If two sides equal, do nothing.
	if(nElem_!=a.nElem_)
	{
		delete [] data_;					// Delete data on left hand side
		nElem_=a.nElem_;
		data_=new elemtype[nElem_];
		/*if(data_==0) throw v3_exc::bad_alloc();	// Check that memory was allocated*/
	}
	for(size_t i=0;i<nElem_;i++) data_[i]=a.data_[i]; // Copy right hand side to l.h.s.
	return *this;
}

template<typename elemtype>
bool vc3_math::OArray<elemtype>::operator==(const OArray<elemtype> &a) const
		noexcept
{
	// Compare subsequently all elements
	if(nElem_!=a.nElem_) return false;
	for(size_t q=0;q<nElem_;q++)
		if(data_[q]!=a.data_[q]) return false;
	return true;
}

template<typename elemtype>
bool vc3_math::OArray<elemtype>::operator!=(const OArray<elemtype> &a) const
		noexcept
{
	// Compare subsequently all elements
	if(nElem_!=a.nElem_) return true;
	for(size_t q=0;q<nElem_;q++)
		if(data_[q]!=a.data_[q]) return true;
	return false;
}

template<typename elemtype>
bool vc3_math::OArray<elemtype>::operator<(const OArray<elemtype> &a) const
		noexcept
{
	// Compare subsequently all elements
	if(nElem_>a.nElem_) return false;
	if(nElem_<a.nElem_) return true;
	for(size_t q=0;q<nElem_;q++)
    {
        if(data_[q]<a.data_[q]) return true;
        if(data_[q]>a.data_[q]) return false;
    }
	return false;
}

template<typename elemtype>
void vc3_math::OArray<elemtype>::resize(size_t nElem)
		throw(/*v3_exc::bad_alloc, v3_exc::bad_index*/)
{
	// Check that nElem > 0
	/*if(nElem<=0) throw v3_exc::bad_index();//("OArray element - out of range");*/
	if(nElem_!=nElem)
	{
		nElem_=nElem;
		delete[] data_;					// Free old memory
		data_=new elemtype[nElem];		// Allocate memory
	}
	/*if(data_==0) throw v3_exc::bad_alloc(); // Check that memory was allocated*/
}

template<typename elemtype>
size_t vc3_math::OArray<elemtype>::nElem() const noexcept
{
	return nElem_;
}

template<typename elemtype>
elemtype & vc3_math::OArray<elemtype>::operator() (size_t i)
		throw(/*v3_exc::bad_index*/)
{
	// Check nElem
	/*if(i<0 || i>=nElem_) throw v3_exc::bad_index();//("OArray element - out of range");*/
	return data_[i];  // Access appropriate value
}

template<typename elemtype>
const elemtype & vc3_math::OArray<elemtype>::operator() (size_t i) const
		throw(/*v3_exc::bad_index*/)
{
	// Check nElem
	/*if(i<0 || i>=nElem_) throw v3_exc::bad_index();//("OArray element - out of range");*/
	return data_[i];  // Access appropriate value
}

template<typename elemtype>
void vc3_math::OArray<elemtype>::set(elemtype value) noexcept
{
	for(size_t i=0;i<nElem_;i++) data_[i]=value;
}

template<typename elemtype>
void vc3_math::OArray<elemtype>::merge(const OArray<elemtype> &a) const
        throw(/*v3_exc::bad_index*/)
{
    for(size_t i=0;i<nElem_;i++) data_[i]+=a.data_[i];
}


#endif //#ifndef VC3_MATH_OARRAY



