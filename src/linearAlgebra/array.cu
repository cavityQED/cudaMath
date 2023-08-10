#include "array.h"
#include "arrayOps.cu"

namespace cm { namespace linear_algebra {


template <typename dataType>
Array<dataType>::Array(const shape_t& shape)
{	
	m_shape = shape;
	m_shape.pad(32);

	if(resize(m_shape) != cudaSuccess)
		std::cout << "\nResize Failed\n" << m_shape << '\n';

}

template <typename dataType>
cudaError_t Array<dataType>::resize(const shape_t& shape)
{
	size_t bytes = 0;

	m_shape = shape;

	// Resize the cpu data
	bytes = m_shape.n_elements() * sizeof(dataType);
	m_data.resize(m_shape.rows(), m_shape.cols());
	std::memset(begin(), 0, bytes);

	// Resize the buffer (gpu data)
	bytes = m_shape.size() * sizeof(dataType);
	cudaMalloc((void**)&m_buffer, bytes);
	cudaMemset(m_buffer, 0, bytes);

	checkLastCudaError("Resize");
	return cudaGetLastError();
}

template <typename dataType>
cudaError_t Array<dataType>::gpuSend()
{
	auto dst_ptr	= m_buffer;
	auto dst_pitch	= m_shape.pitch() * sizeof(dataType);
	auto src_ptr	= begin();
	auto src_pitch	= m_shape.cols() * sizeof(dataType);
	auto width		= src_pitch;
	auto height		= m_shape.rows();
/*
	std::cout << "\n\tSending Data to GPU:\n";
	std::cout << "\tDestination Pitch:\t" 		<< dst_pitch 	<< '\n';
	std::cout << "\tSource Pitch:\t\t"			<< src_pitch 	<< '\n';
	std::cout << "\tWidth\t\t\t"				<< width		<< '\n';
	std::cout << "\tHeight\t\t\t"				<< height		<< '\n';
*/
	cudaMemcpy2D(	dst_ptr,
					dst_pitch,
					src_ptr,
					src_pitch,
					width,
					height,
					cudaMemcpyHostToDevice);

	checkLastCudaError("GPU Send");
	return cudaGetLastError();
}

template <typename dataType>
cudaError_t Array<dataType>::gpuRecv()
{
	auto dst_ptr	= begin();
	auto dst_pitch	= m_shape.cols() * sizeof(dataType);
	auto src_ptr	= m_buffer;
	auto src_pitch	= m_shape.pitch() * sizeof(dataType);
	auto width		= dst_pitch;
	auto height		= m_shape.rows();
/*
	std::cout << "\n\tRecieving Data from GPU:\n";
	std::cout << "\tDestination Pitch:\t" 		<< dst_pitch 	<< '\n';
	std::cout << "\tSource Pitch:\t\t"			<< src_pitch 	<< '\n';
	std::cout << "\tWidth\t\t\t"				<< width		<< '\n';
	std::cout << "\tHeight\t\t\t"				<< height		<< '\n';
*/
	cudaMemcpy2D(	dst_ptr,
					dst_pitch,
					src_ptr,
					src_pitch,
					width,
					height,
					cudaMemcpyDeviceToHost);

	checkLastCudaError("GPU Recv");
	return cudaGetLastError();
}

template <typename dataType>
cudaError_t Array<dataType>::setData(dataType val)
{
	for(auto ptr = begin(); ptr != end(); ptr++)
		*ptr = val;

	gpuSend();

	checkLastCudaError("Set Data");
	return cudaGetLastError();
}

template <typename dataType>
cudaError_t Array<dataType>::setData(const Array<dataType>& arr)
{
	auto dst_ptr = begin();
	auto src_ptr = arr.begin();

	while(dst_ptr != end() && src_ptr != arr.end())
	{
		*dst_ptr = *src_ptr;
		++dst_ptr;
		++src_ptr;
	}

	gpuSend();

	checkLastCudaError("Set Data");
	return cudaGetLastError();
}

template <typename dataType>
cudaError_t Array<dataType>::setBuffer(dataType val)
{
	dim3	blocks;
	dim3	threads;
	size_t	nx = m_shape.cols();
	size_t	ny = m_shape.rows();

	threads.x = cm::factor_of(nx, 32);
	threads.y = cm::factor_of(ny, 32);

	blocks.x = nx / threads.x;
	blocks.y = ny / threads.y;

	cm::kernels::set_buffer<<<blocks, threads>>>(m_buffer, val, m_shape.pitch());
	cudaDeviceSynchronize();

	checkLastCudaError();
	return cudaGetLastError();
}

template <typename dataType>
cudaError_t Array<dataType>::setRand(dataType min, dataType max)
{
	for(auto ptr = begin(); ptr != end(); ptr++)
		*ptr = cm::rng<dataType>(min, max);

	gpuSend();

	checkLastCudaError("Set Rand");
	return cudaGetLastError();
}

template <typename dataType>
cudaError_t Array<dataType>::makeOnes()
{
	for(auto ptr = begin(); ptr != end(); ptr++)
		*ptr = (dataType)!(*ptr == 0);
	gpuSend();

	checkLastCudaError("Make Ones");
	return cudaGetLastError();
}

template <typename dataType>
void Array<dataType>::print() const
{
	std::cout << "Array:\n" << m_shape << '\n';

	for(auto row = 0; row < m_data.size1(); row++)
	{
		auto it1 = m_data.begin1() + row;		
		std::cout << "\tRow " << row << ":\t";
		for(auto it2 = it1.begin(); it2 != it1.end(); it2++)
			std::cout << *it2 << " ";
		std::cout << '\n';
	}
}

/****************
*	Functions	*
* **************/

template <typename dataType>
cudaError_t Array<dataType>::product(const Array<dataType>& lhs, const Array<dataType>& rhs, dataType c)
{
	if(m_shape.b != rhs.shape().b)
	{
		m_shape.b = rhs.shape().b;
		m_shape.pad(32);
		resize(m_shape);
	}

	size_t ne = lhs.pitch();
	size_t nx = rhs.pitch();
	size_t ny = lhs.height();

	ASSERT(ne == rhs.height());
	ASSERT(nx == pitch());
	ASSERT(ny == height());

	dim3 blocks;
	dim3 threads;

	threads.x = cm::factor_of(nx*ne, 32);
	threads.y = threads.x;

	blocks.x = ne / threads.x;
	blocks.y = ny / threads.y;
	blocks.z = nx / threads.x;

	cm::kernels::product<<<blocks, threads>>>(	lhs.buffer(),
												rhs.buffer(),
												m_buffer,
												c);
	cudaDeviceSynchronize();
	checkLastCudaError();
	return cudaGetLastError();
}

template <typename dataType>
std::shared_ptr<Array<dataType>> Array<dataType>::transpose()
{
	Array<dataType>::shape_t out_shape	= m_shape;
	out_shape.row_data 					= !m_shape.row_data;
	auto out							= std::make_shared<Array<dataType>>(out_shape);

	dim3 blocks;
	dim3 threads;

	threads.x = cm::factor_of(m_shape.cols(), 32);
	blocks.x = m_shape.cols() / threads.x;

	threads.y = threads.x;
	blocks.y = m_shape.rows() / threads.y;

	cm::kernels::transpose<<<blocks, threads>>>(	m_buffer,
													out->buffer(),
													m_shape.pitch(),
													out_shape.pitch());
	cudaDeviceSynchronize();
	checkLastCudaError("Transpose");

	return out;
}

template <typename dataType>
std::shared_ptr<Array<dataType>> Array<dataType>::element_add(const Array<dataType>& lhs, const Array<dataType>& rhs, dataType c)
{
	size_t nl = lhs.shape().chunk_size();
	size_t nr = rhs.shape().chunk_size();
	
	ASSERT(nl == nr);

	dim3 threads;
	dim3 blocks;

	threads.x	= cm::factor_of(nl, 32);
	blocks.x	= nl / threads.x;

	auto out = std::make_shared<Array<dataType>>(lhs.shape());

	cm::kernels::element_add<<<blocks, threads>>>(	lhs.buffer(),
													rhs.buffer(),
													out->buffer(),
													c);
	cudaDeviceSynchronize();
	checkLastCudaError("Element Add");
	return out;
}


/************************
 * 	Operator Overloads	*
 ***********************/

template <typename dataType>
bool Array<dataType>::operator==(const boostMatrix& mat) const
{
	auto arr_ptr = begin();
	auto mat_ptr = &mat(0,0);

	size_t count = 0;

	while(arr_ptr != end())
	{
		if(std::fabs(*arr_ptr - *mat_ptr) > 0.001)
		{
			std::cout << "Array Value:\t" << *arr_ptr << '\n';
			std::cout << "Boost Value:\t" << *mat_ptr << '\n';
			return false;
		}

		++arr_ptr;
		++mat_ptr;
		++count;
	}

	return true;
}

template <typename dataType>
bool Array<dataType>::operator==(const Array<dataType>& arr) const
{
	return *this == arr.data();
}

template <typename dataType>
std::shared_ptr<Array<dataType>> Array<dataType>::operator+(const Array<dataType>& rhs) const
{
	ASSERT(m_shape.chunk_size() == rhs.shape().chunk_size());

	return element_add(*this, rhs, 1);
}

template <typename dataType>
std::shared_ptr<Array<dataType>> Array<dataType>::operator-(const Array<dataType>& rhs) const
{
	ASSERT(m_shape.chunk_size() == rhs.shape().chunk_size());

	return element_add(*this, rhs, -1);
}

template <typename dataType>
std::shared_ptr<Array<dataType>> Array<dataType>::operator+(dataType c) const
{
	auto rhs = std::make_shared<Array<dataType>>(m_shape);
	rhs->setBuffer(c);

	return element_add(*this, *rhs, 1);
}

template class Array<double>;
template class Array<int>;


template <typename T>
Container<T>::Container(size_t size) : boost::numeric::ublas::matrix<T>(size, 1)
{
	m_size = size;
	auto bytes = m_size * sizeof(T);
	cudaMalloc((void**)&m_buffer, bytes);
	cudaMemset(m_buffer, 0, bytes);

	std::memset(&(*this)(0,0), 0, bytes);

	checkLastCudaError("Container<T>::Container(size_t)");
}

template <typename T>
Container<T>::~Container()
{
	cudaFree(m_buffer);
}

template <typename T>
void Container<T>::resize(size_t size)
{
	m_size = size;
	auto bytes = m_size * sizeof(T);
	cudaMalloc((void**)&m_buffer, bytes);
	cudaMemset(m_buffer, 0, bytes);

	boost::numeric::ublas::matrix<T>::resize(size, 1);
	std::memset(&(*this)(0,0), 0, bytes);

	checkLastCudaError("Container<T>::resize(size_t)");
}

template <typename T>
void Container<T>::sendData()
{
	cudaMemcpy(	m_buffer,
				begin(),
				m_size * sizeof(T),
				cudaMemcpyHostToDevice);

	checkLastCudaError("Container<T>::sendData()");
}

template <typename T>
void Container<T>::recvData()
{
	cudaMemcpy(	begin(),
				m_buffer,
				m_size * sizeof(T),
				cudaMemcpyDeviceToHost);

	checkLastCudaError("Container<T>::recvData()");
}

template <typename T>
void Container<T>::clear()
{
	auto bytes = m_size * sizeof(T);
	cudaMemset(m_buffer, 0, bytes);
	std::memset(begin(), 0, bytes);

	checkLastCudaError("Container<T>::clear()");
}

template class Container<double>;
template class Container<float>;
template class Container<int>;

} // linear_algebra namespace
} // cm namespace