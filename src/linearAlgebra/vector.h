#ifndef VECTOR_H
#define VECTOR_H

#include "cudaMath.h"

namespace cm { namespace linalg {

template <typename dataType>
class Vector
{

	typedef boost::numeric::ublas::vector<dataType> boostVector;

public:

	struct shape_t
	{

		size_t x = 1;
		size_t y = 1;
		size_t z = 1;

		size_t p = 0; //padding

		shape_t(size_t x = 1, size_t y = 1, size_t z = 1) 	: x(x), y(y), z(z)						{}
		shape_t(const shape_t& shape)						: x(shape.x), y(shape.y), z(shape.z)	{}

		size_t	size()				{return x*y*z;	}
		size_t	span()				{return x*y*z + p;}
		void 	pad(size_t f = 32)	{p = cm::next_multiple(size(), f) - size();}

	};

	Vector(const shape_t& shape);

	cudaError_t resize(const shape_t& shape);


protected:

	boostVector m_hst_data;
	dataType*	m_dev_data;
	shape_t		m_shape;
};

} //namespace linalg
} //namespace cm

#endif
