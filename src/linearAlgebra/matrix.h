#ifndef MATRIX_H
#define MATRIX_H

#include "cudaMath.h"
#include "utilities/timer.hpp"


namespace cm { namespace linalg {



template <typename T>
class Matrix : public boost::numeric::ublas::matrix<T>
{

public:

	struct shape_t
	{
		size_t x = 1;
		size_t y = 1;
		size_t z = 1;
		size_t p = 0; // padding

		shape_t(size_t x = 1, size_t y = 1, size_t z = 1) : x(x), y(y), z(z) {}

		shape_t(const shape_t& shape) : x(shape.x), y(shape.y), z(shape.z) {}

		void	pad(size_t f = 32)	{p = cm::next_multiple(n_elements(), f) - n_elements();}
		
		size_t	width() {return x*y*z;}
		size_t	pitch() {return x*y*z + p;}
	};


public:

	Matrix(const shape_t& vshape, const shape_t& hshape);

	~Matrix();

	void resize(const shape_t& vshape, const shape_t& hshape);

public:

			size_t	cols()			const	{return m_hshape.width();}
			size_t	rows()			const	{return m_vshape.width();}
			size_t	pitch()			const	{return m_hshape.pitch();}	
			size_t	height()		const	{return m_vshape.pitch();}
			size_t	size()			const	{return pitch() * height();}
			size_t	n_elements()	const	{return cols() * rows();}
			T*		begin()					{return &(*this)(0,0);}
	const	T*		begin()			const	{return &(*this)(0,0);}
	const	T*		end()			const	{return begin() + n_elements();}
	const	T*		buffer()		const	{return m_buffer;}
			T*		buffer()				{return m_buffer;}

protected:

	shape_t		m_vshape;
	shape_t		m_hshape;
	T*			m_buffer;

};



} // cm namespace
} // linear_algebra

#endif