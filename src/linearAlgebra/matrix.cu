#include "matrix.h"


namespace cm { namespace linalg {

template <typename T>
Matrix<T>::Matrix(const shape_t& vshape, const shape_t& hshape)
	: boost::numeric::ublas::matrix<T>(vshape.pitch(), hshape.pitch())
{
	resize(vshape.pad(32), hshape.pad(32));
}

template <typename T>
Matrix<T>::~Matrix()
{
	cudaFree(m_buffer);
}

template <typename T>
void Matrix<T>::resize(const shape_t& vshape, const shape_t& hshape)
{
	m_vshape = vshape;
	m_hshape = hshape;
	auto bytes = size() * sizeof(T);
	cudaMalloc((void**)m_buffer, bytes);
	cudaMemset(m_buffer, 0, bytes);

	boost::numeric::ublas::matrix<T>::resize(m_vshape.width(), m_hshape.width());
	cudaMemset(begin(), 0, bytes);

	checkLastCudaError("Matrix<T>::resize");
}

} // cm namespace
} // linalg namespace
