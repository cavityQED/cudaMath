#include "vector.h"

namespace cm { namespace linalg {


template <typename dataType>
Vector<dataType>::Vector(const shape_t& shape)
{
	resize(shape);
}

template <typename dataType>
cudaError_t Vector<dataType>::resize(const shape_t& shape)
{
	m_shape = shape;
	m_shape.pad(32);

	m_host_data.resize(m_shape.size());

	cudaMalloc(	(void**)&m_buffer, m_shape.span() * sizeof(dataType));
	return cudaMemset(m_buffer, 0, m_shape.span() * sizeof(dataType));
}


template class Vector<double>;

} //namespace linalg
} //namespace cm