#ifndef ARRAY_H
#define ARRAY_H

#include "cudaMath.h"
#include "utilities/timer.hpp"

namespace cm {	namespace linear_algebra
{

	template <typename dataType>
	class Array
	{
		typedef boost::numeric::ublas::matrix<dataType>	boostMatrix;
		typedef std::shared_ptr<Array>					arrayPtr;

	public:

		struct shape_t
		{
			size_t x		= 1;
			size_t y		= 1;
			size_t z		= 1;
			size_t b		= 1;
			size_t cpad		= 0;	// padding for columns
			size_t rpad 	= 0;	// padding for rows
			bool   row_data = false;

			size_t	cols()			const {return (row_data)? x*y*z : b;}
			size_t	rows()			const {return (row_data)? b : x*y*z;}
			size_t	pitch()			const {return cols() + cpad;}
			size_t	height()		const {return rows() + rpad;}
			size_t	size()			const {return pitch() * height();}
			size_t	chunk_size()	const {return pitch() * rows();}
			size_t	n_elements()	const {return x*y*z*b;}

			// pad height and width to make them multiples of f
			void pad(size_t f = 32)
			{
				cpad = cm::next_multiple(cols(), f) - cols();
				rpad = cm::next_multiple(rows(), f) - rows();
			}

			void xyzb(std::string& str) const
			{
				std::ostringstream out;
				out << "(" << x << ", " << y << ", " << z << ", " << b << ")";
				str = out.str().c_str();
			}

			friend std::ostream& operator<<(std::ostream& out, const shape_t shape)
			{
				if(shape.row_data)
					out << "\tData in Rows\n";
				else
					out << "\tData in Columns\n";

				out << "\tShape [x, y , z, b]:\t[" << shape.x << ", " << shape.y << ", " << shape.z << ", " << shape.b << "]\n";
				out << "\tPads [col, row]:\t["	<< shape.cpad << ", " << shape.rpad << "]\n";
				out << "\tPitch:\t\t\t"			<< shape.pitch() << '\n';
				out << "\tHeight:\t\t\t"		<< shape.height() << '\n';
				out << "\tElements:\t\t"		<< shape.n_elements() << '\n';
				out << "\tBuffer Size:\t\t"		<< shape.size() << '\n';

				return out;
			}
		};


	public:

		Array() {}
		Array(const shape_t& shape);

		~Array() {cudaFree(m_buffer);}

		cudaError_t resize(const shape_t& shape);
		cudaError_t gpuSend();
		cudaError_t gpuRecv();

		cudaError_t setData		(dataType val = 1);
		cudaError_t setData		(const Array& arr);
		cudaError_t setRand		(dataType min = 0, dataType max = 1);
		cudaError_t setBuffer	(dataType val = 0);
		cudaError_t clearBuffer	();

		cudaError_t makeOnes();

	public:

		const 	dataType*		buffer()	const	{return m_buffer;}
			 	dataType*		buffer()			{return m_buffer;}
		const 	boostMatrix&	data()		const	{return m_data;}
				boostMatrix&	data()				{return	m_data;}
		const 	dataType*		begin()		const	{return &m_data(0,0);}
				dataType*		begin()				{return &m_data(0,0);}
		const 	dataType*		end()		const	{return &m_data(0,0) + m_shape.n_elements();}
		const 	shape_t&		shape()		const	{return m_shape;}
				size_t			rows()		const	{return m_shape.rows();}
				size_t			cols()		const	{return m_shape.cols();}
				size_t			pitch()		const	{return m_shape.pitch();}
				size_t			height()	const	{return m_shape.height();}

	public:

		void print() const;

	public:

		cudaError_t	element_add		(const Array& rhs, dataType c = 1);
		cudaError_t product			(const Array& lhs, const Array& rhs, dataType c = 1);

		static	arrayPtr	element_add	(const Array& lhs, const Array& rhs, dataType c = 1);
				arrayPtr	transpose	();

	public:

		bool operator==(const boostMatrix& mat)	const;
		bool operator==(const Array& arr)		const;

		arrayPtr operator+(const Array& rhs)	const;
		arrayPtr operator-(const Array& rhs)	const;

		arrayPtr operator+(dataType c)			const;
		arrayPtr operator-(dataType c)			const;

	protected:

		shape_t			m_shape;
		boostMatrix		m_data;
		dataType*		m_buffer;

	}; // Array class


	template <typename T>
	class Container : public boost::numeric::ublas::matrix<T>
	{

	public:

		Container(size_t size);

		virtual ~Container();

		virtual void resize(size_t size);
		virtual void sendData();
		virtual void recvData();
		virtual void clear();

				size_t	size()		const	{return m_size;}
		const	T*		begin()		const	{return &(*this)(0,0);}
				T*		begin()				{return &(*this)(0,0);}
		const	T*		end()		const	{return begin() + m_size;}
				T*		end()				{return begin() + m_size;}
		const	T*		buffer()	const	{return m_buffer;}
				T*		buffer()			{return m_buffer;}

	protected:

		size_t		m_size;
		T*			m_buffer;
		
	}; // Container class

} // linear_algebra namespace
} // cm namespace

#endif