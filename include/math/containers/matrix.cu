#ifndef VC3_MATH_MATRIX
#define VC3_MATH_MATRIX

#include "array.cu"

namespace vc3_exc
{
	class zero_determinant: public bad_value
	{
        public:
		zero_determinant(std::string s="Matrix determinant = 0")
			{name=s;}
	}; // class zero_determinant

}; //namespace vc3_exc

namespace vc3_math {

// Matrix class: array-based class of user-defined objects of objects for wich
// mathematical operators are defined (requested for mathematical operations on
// Matrix like inversion, determinant calculations, multiplication)
template<typename FLT> class Matrix: public Array<FLT>
{
	public:

	// Default Constructor. Creates Matrix of nRow x nCol with unknown values.
	// If number of elements is not specified, it is set to 1.
	Matrix(int nRow=1, int nCol=1) throw(vc3_exc::bad_alloc, vc3_exc::bad_index);

	// Constructor. Creates  Matrix of nRow x nCol
	// with all the same values equal to 'value'; no default value.
	Matrix(int nRow, int nCol, FLT value) throw(vc3_exc::bad_alloc, vc3_exc::bad_index);

	// Copy Constructor.
	// Used when a copy of an object is produced
	Matrix(const Matrix<FLT> &m) throw(vc3_exc::bad_alloc);

	// Assignment operator function.
	// Overloads the equal sign operator to work with Matrix objects.
	// Changes size of left operand if needed.
	virtual Matrix<FLT> & operator=(const Matrix<FLT> &m) throw(vc3_exc::bad_alloc);

	// Resize functions
	// * Do not use it if you can avoid it, very time-consuming function.
	// Redefenition of OArray resize function (nCol=1);
	virtual void resize(int nRow) throw(vc3_exc::bad_alloc, vc3_exc::bad_index);
	// Normal resize
	virtual void resize(int nRow, int nCol) throw(vc3_exc::bad_alloc, vc3_exc::bad_index);

	// Simple "get" functions. Return number of elements.
	int nCol() const noexcept;
	int nRow() const noexcept;

	// Parenthesis operator function.
	// Allows access to values of Matrix via indexes i,j;
	// Example: a(1,3) = 2*b(3,4);
	FLT & operator() (int nRow, int nCol) throw(vc3_exc::bad_index);

	// Parenthesis operator function (const version).
	const FLT & operator() (int nRow, int nCol) const throw(vc3_exc::bad_index);

	// Matrix * Matrix multiplication.
	// Resizes resulting matrix (this) if needed.
	void mult(const Matrix<FLT> &left, const Matrix<FLT> &right) throw(vc3_exc::self_handling, vc3_exc::incompatible_sizes);

	// Array * Array multiplication.
	// Resizes resulting matrix (this) if needed.
	void mult(const Array<FLT> &left, const Array<FLT> &right) throw(vc3_exc::self_handling, vc3_exc::incompatible_sizes);

	// Matrix * Array multiplication.
	// Resizes resulting matrix (this) if needed.
	void mult(const Matrix<FLT> &left, const Array<FLT> &right) throw(vc3_exc::self_handling, vc3_exc::incompatible_sizes);

	// Array * Matrix multiplication.
	// Resizes resulting matrix (this) if needed.
	void mult(const Array<FLT> &left, const Matrix<FLT> &right) throw(vc3_exc::self_handling, vc3_exc::incompatible_sizes);

	// Transposition function.
	// Resizes resulting matrix (this) if needed.
	void tr(const Matrix<FLT> &m) throw(vc3_exc::self_handling);

	// Compute inverse of matrix and return determinant
	FLT inv(Matrix<FLT> m) throw(vc3_exc::self_handling, vc3_exc::incompatible_sizes, vc3_exc::zero_determinant);

	//*********************************************************************
	protected:
	// Matrix data
	int nRow_, nCol_; 		// Number of elements

}; //class Matrix

template<typename FLT>
void mult(Array<FLT> *to, const Matrix<FLT> &left, const Array<FLT> &right)
		throw(vc3_exc::self_handling, vc3_exc::incompatible_sizes);

template<typename FLT>
void mult(Array<FLT> *to, const Array<FLT> &left, const Matrix<FLT> &right)
		throw(vc3_exc::self_handling, vc3_exc::incompatible_sizes);

} //namespace vc3_math


//*************************************************
//Realization is here as joint template compilation is not supported

template<typename FLT>
vc3_math::Matrix<FLT>::Matrix(int nRow, int nCol)
		throw(vc3_exc::bad_alloc, vc3_exc::bad_index):
		Array<FLT>(nRow*nCol)
{
	if(nRow<=0 || nCol<=0) throw vc3_exc::bad_index();//("Matrix constructor: Row or Col index <= 0");
	nRow_=nRow;
	nCol_=nCol;
}

template<typename FLT>
vc3_math::Matrix<FLT>::Matrix(int nRow, int nCol, FLT value)
		throw(vc3_exc::bad_alloc, vc3_exc::bad_index):
		Array<FLT>(nRow*nCol,value)
{
	if(nRow<=0 || nCol<=0) throw vc3_exc::bad_index();//("Matrix constructor: Row or Col index <= 0");
	nRow_=nRow;
	nCol_=nCol;
}

template<typename FLT>
vc3_math::Matrix<FLT>::Matrix(const Matrix<FLT> &m)
		throw(vc3_exc::bad_alloc):
		Array<FLT>(m)
{
	nRow_=m.nRow_;
	nCol_=m.nCol_;
}


template<typename FLT>
vc3_math::Matrix<FLT> & vc3_math::Matrix<FLT>::operator=(const Matrix<FLT> &m)
		throw(vc3_exc::bad_alloc)
{
	if(this==&m) return *this;	// If two sides equal, do nothing.
	if(nRow_!=m.nRow_ || nCol_!=m.nCol_)
	{
		delete [] OArray<FLT>::data_;					// Delete data on left hand side
		OArray<FLT>::nElem_=m.nElem_;
		nRow_=m.nRow_;
		nCol_=m.nCol_;
		OArray<FLT>::data_=new FLT[OArray<FLT>::nElem_];
		if(OArray<FLT>::data_==0) throw vc3_exc::bad_alloc();	// Check that memory was allocated
	}
	for(int i=0;i<OArray<FLT>::nElem_;i++) OArray<FLT>::data_[i]=m.data_[i]; // Copy right hand side to l.h.s.
	return *this;
}

template<typename FLT>
void vc3_math::Matrix<FLT>::resize(int nElem)
		throw(vc3_exc::bad_alloc, vc3_exc::bad_index)
{
	if(nElem<0) throw vc3_exc::bad_index();//("Matrix resize: Row or Col index <= 0");	// Check that both nRow and nCol > 0.
	nRow_=1;
	nCol_=nElem;
	OArray<FLT>::nElem_=nRow_*nCol_;
	delete[] OArray<FLT>::data_;					// Free old memory
	OArray<FLT>::data_=new FLT[OArray<FLT>::nElem_];		// Allocate memory
	if(OArray<FLT>::data_==0) throw vc3_exc::bad_alloc(); // Check that memory was allocated
}

template<typename FLT>
void vc3_math::Matrix<FLT>::resize(int nRow, int nCol)
		throw(vc3_exc::bad_alloc, vc3_exc::bad_index)
{
	if(nRow<=0 || nCol<0) throw vc3_exc::bad_index();//("Matrix resize: Row or Col index <= 0");	// Check that both nRow and nCol > 0.
	nRow_=nRow;
	nCol_=nCol;
	OArray<FLT>::nElem_=nRow_*nCol_;
	delete[] OArray<FLT>::data_;					// Free old memory
	OArray<FLT>::data_=new FLT[OArray<FLT>::nElem_];		// Allocate memory
	if(OArray<FLT>::data_==0) throw vc3_exc::bad_alloc(); // Check that memory was allocated
}

template<typename FLT>
int vc3_math::Matrix<FLT>::nCol() const noexcept
{
	return nCol_;
}

template<typename FLT>
int vc3_math::Matrix<FLT>::nRow() const noexcept
{
    return nRow_;
}

template<typename FLT>
FLT & vc3_math::Matrix<FLT>::operator() (int nRow, int nCol)
		throw(vc3_exc::bad_index)
{
	if(nRow<0 || nRow>=nRow_ || nCol<0 || nCol>=nCol_) 
		throw vc3_exc::bad_index();//("Matrix parenthis: invalid nRow or nCol index");	// Check that both nRow and nCol have acceptable values.
	return OArray<FLT>::data_[nRow*nCol_+nCol];  // Access appropriate value
}

	// Parenthesis operator function (const version).
template<typename FLT>
const FLT & vc3_math::Matrix<FLT>::operator() (int nRow, int nCol) const
		throw(vc3_exc::bad_index)
{
	if(nRow<0 || nRow>=nRow_ || nCol<0 || nCol>=nCol_) throw vc3_exc::bad_index();//("Matrix parenthis: invalid nRow or nCol index");	// Check that both nRow and nCol have acceptable values.
	return OArray<FLT>::data_[nRow*nCol_+nCol];  // Access appropriate value
}

template<typename FLT>
void vc3_math::Matrix<FLT>::mult(const Array<FLT> &left, const Array<FLT> &right)
		throw(vc3_exc::self_handling, vc3_exc::incompatible_sizes)
{
	if(this == &left || this == &right ) throw vc3_exc::self_handling();//("Matrix::mult self handling");
	if(left.nElem() != right.nElem()) throw vc3_exc::incompatible_sizes();//(/*"Matrix::mult(..) left.nElem != right.nElem"*/);
	if(nRow_ != left.nElem() || nCol_ != right.nElem())
		resize(left.nElem(), right.nElem());
	for(int nR=0;nR<nRow_;nR++)
		for(int nC=0;nC<nCol_;nC++)
			OArray<FLT>::data_[nR*nCol_+nC]=left(nR)*right(nC);
}

template<typename FLT>
void vc3_math::Matrix<FLT>::mult(const Matrix<FLT> &left, const Matrix<FLT> &right)
		throw(vc3_exc::self_handling, vc3_exc::incompatible_sizes)
{
	if(this == &left || this == &right ) throw vc3_exc::self_handling();//("Matrix::mult self handling");
	if(left.nCol_ != right.nRow_) throw vc3_exc::incompatible_sizes();//(/*"Matrix::mult(..) left.nCol != right.nRow"*/);
	if(nRow_ != left.nRow_ || nCol_ != right.nCol_)
		resize(left.nRow_, right.nCol_);
	for(int nR=0;nR<nRow_;nR++)
	{
		for(int nC=0;nC<nCol_;nC++)
		{
			OArray<FLT>::data_[nR*nCol_+nC]=0;
			for(int i=0;i<left.nCol_;i++)
				OArray<FLT>::data_[nR*nCol_+nC]+=left.data_[nR*left.nCol_+i]*right.data_[i*right.nCol_+nC];
		}
	}
}

template<typename FLT>
void vc3_math::Matrix<FLT>::mult(const Matrix<FLT> &left, const Array<FLT> &right)
		throw(vc3_exc::self_handling, vc3_exc::incompatible_sizes)
{
	if(this == &left || this == &right ) throw vc3_exc::self_handling();//("Matrix::mult self handling");
	if(left.nCol_ != right.nElem()) throw vc3_exc::incompatible_sizes();//("Matrix::mult(..) left.nCol != right.nElem");
	if(nRow_ != left.nRow_ || nCol_ != 1)
		resize(left.nRow_, 1);
	for(int nR=0;nR<nRow_;nR++)
	{
		for(int nC=0;nC<nCol_;nC++) // actually no cycle
		{
			OArray<FLT>::data_[nR*nCol_+nC]=0;
			for(int i=0;i<left.nCol_;i++)
				OArray<FLT>::data_[nR*nCol_+nC]+=left.data_[nR*left.nCol_+i]*right(i+nC);
		}
	}
}

template<typename FLT>
void vc3_math::Matrix<FLT>::mult(const Array<FLT> &left, const Matrix<FLT> &right)
		throw(vc3_exc::self_handling, vc3_exc::incompatible_sizes)
{
	if(this == &left || this == &right ) throw vc3_exc::self_handling();//("Matrix::mult self handling");
	if(left.nElem() != right.nRow_) throw vc3_exc::incompatible_sizes();//("Matrix::mult(..) left.nElem != right.nRow");
	if(nRow_ != 1 || nCol_ != right.nCol_)
		resize(1, right.nCol_);
	for(int nR=0;nR<nRow_;nR++) // actually no cycle
	{
		for(int nC=0;nC<nCol_;nC++)
		{
			OArray<FLT>::data_[nR*nCol_+nC]=0;
			for(int i=0;i<left.nElem_;i++)
				OArray<FLT>::data_[nR*nCol_+nC]+=left(nR*left.nElem()+i)*right.data_[i*right.nCol_+nC];
		}
	}
}

template<typename FLT>
void vc3_math::Matrix<FLT>::tr(const Matrix<FLT> &m)
		throw(vc3_exc::self_handling)
{
    /**! ERROR
    RESULTS IN SEGFAULT WHEN RESIZE IS CALLED !**/

	// If two sides are equal, do nothing.

	if(this==&m) throw vc3_exc::self_handling();//("Matrix::tr self handling");
	if(nRow_ != m.nCol_ || nCol_ != m.nRow_) resize(m.nCol_, m.nRow_);
	for(int nR=0;nR<nRow_;nR++)
		for(int nC=1;nC<=nCol_;nC++)
			OArray<FLT>::data_[ nCol_*(nR-1) + (nC-1) ] = m.data_[ m.nCol_*(nC-1) + (nR-1) ];
}

template<typename FLT>
FLT vc3_math::Matrix<FLT>::inv(Matrix<FLT> m)
		throw(vc3_exc::self_handling, vc3_exc::incompatible_sizes, vc3_exc::zero_determinant)
{

	// If two sides are equal, do nothing.

	if(this==&m) throw vc3_exc::self_handling();//("Matrix::inv self handling");

	if(m.nRow()!=m.nCol()) throw vc3_exc::incompatible_sizes();//(/*"Matrix::inv(..) nCol != nRow"*/);

	int N=m.nRow();

	*this=m;  // Copy matrix to ensure Ainv is same size
	Matrix scale(N), b(N,N);	 // Scale factor and work array
	int *index=new int [N];

	//* Matrix b is initialized to the identity matrix
	b.set(FLT(0.0));
	for(int i=0;i<N;i++) b(i,i) = FLT(1.0);

	//* Set scale factor, scale(i) = max( |a(i,j)| ), for each row
	for(int i=0;i<N;i++)
	{
		index[i]=i;			  // Initialize row index list
		FLT scalemax = FLT(0.0);
		for(int j=0;j<N;j++)
			scalemax=( scalemax>FLT( fabs(m(i,j)) ) ) ? scalemax : FLT( fabs(m(i,j)) );
		scale(i,0)=scalemax;
	}

	//* Loop over rows k = 0, ..., (N-1)
	int signDet=1;
	for(int k=0;k<N-1;k++ )
	{
		//* Select pivot row from max( |a(j,k)/s(j)| )
		FLT ratiomax=0.0;
		int jPivot = k;
		for(int i=k;i<N;i++)
		{
			FLT ratio=fabs(m(index[i],k))/scale(index[i],0);
			if(ratio>ratiomax)
			{
				jPivot=i;
				ratiomax=ratio;
			}
		}

		//* Perform pivoting using row index list
		int indexJ=index[k];
		if(jPivot!=k )
		{	          // Pivot
			indexJ=index[jPivot];
			index[jPivot]=index[k];   // Swap index jPivot and k
			index[k]=indexJ;
			signDet*=-1;			  // Flip sign of determinant
		}

		//* Perform forward elimination
		for(int i=k+1;i<N;i++)
		{
			FLT coeff = m(index[i],k)/m(indexJ,k);
			for(int j=k+1;j<N;j++) m(index[i],j)-=coeff*m(indexJ,j);
			m(index[i],k)=coeff;
			for(int j=0;j<N;j++) b(index[i],j)-=m(index[i],k)*b(indexJ,j);
		}
	}

	//* Compute determinant as product of diagonal elements
	FLT determ=signDet;	   // Sign of determinant
	for(int i=0;i<N;i++) determ*=m(index[i],i);

	if(determ==FLT(0.0))
	{
		delete [] index;	// Release allocated memory
/* should be replaced with Array<FLT> to avoid this casualties!!! */
		throw vc3_exc::zero_determinant();//("Matrix::inv(..) determinant is equal to zero!");
	}

	//* Perform backsubstitution
	for(int k=0;k<N;k++)
	{
		(*this)(N-1,k) = b(index[N-1],k)/m(index[N-1],N-1);
		for(int i=N-2;i>=0;i--)
		{
			FLT sum=b(index[i],k);
			for(int j=i+1;j<N;j++) sum-=m(index[i],j) * ((*this)(j,k));
			(*this)(i,k) = sum/m(index[i],i);
		}
	}

	delete [] index;	// Release allocated memory
	return( determ );
}

template<typename FLT>
void vc3_math::mult(Array<FLT> *to, const Matrix<FLT> &left, const Array<FLT> &right)
		throw(vc3_exc::self_handling, vc3_exc::incompatible_sizes)
{
	if(/*to == &left ||*/ to == &right ) throw vc3_exc::self_handling();//("mult(..) self handling");
	if(left.nCol() != right.nElem()) throw vc3_exc::incompatible_sizes();//(/*"Matrix::mult(..) left.nCol != right.nElem"*/);
	if(to->nElem() != right.nElem())
		to->resize(right.nElem());
	for(int nR=0;nR<to->nElem();nR++)
	{
		(*to)(nR)=0;
		for(int i=0;i<left.nCol();i++)
			(*to)(nR)+=left(nR,i)*right(i);
	}
}

template<typename FLT>
void vc3_math::mult(Array<FLT> *to, const Array<FLT> &left, const Matrix<FLT> &right)
		throw(vc3_exc::self_handling, vc3_exc::incompatible_sizes)
{
	if(to == &left || to == &right ) throw vc3_exc::self_handling();//("mult(..) self handling");
	if(left.nElem() != right.nRow()) throw vc3_exc::incompatible_sizes();//("Matrix::mult(..) left.nElem != right.nRow");
	if(to->nElem() != left.nElem())
		to->resize(left.nElem());
	for(int nC=0;nC<to->nElem();nC++)
	{
		(*to)(nC)=0;
		for(int i=0;i<right.nRow();i++)
			(*to)(nC)+=left(i)*right(i,nC);
	}
}

#endif //#ifndef VC3_MATH_MATRIX

