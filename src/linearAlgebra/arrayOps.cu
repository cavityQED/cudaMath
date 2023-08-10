

namespace cm {	namespace kernels {

template <typename dataType>
__global__ void set_buffer(	dataType*	buf,
							dataType	val,
							size_t		stride)
{
	size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	size_t idy = blockIdx.y * blockDim.y + threadIdx.y;
	size_t id = idy * stride + idx;

	buf[id] = val;
}


template <typename dataType>
__global__ void element_add(	const dataType* lhs,
								const dataType* rhs,
								dataType*		out,
								dataType		c)
{
	size_t id = blockIdx.x * blockDim.x + threadIdx.x;

	out[id] = lhs[id] + c*rhs[id];
}

template <typename dataType>
__global__ void element_add(	const dataType* in,
								dataType*		out,
								dataType		c)
{
	size_t id = blockIdx.x * blockDim.x + threadIdx.x;

	out[id] = in[id] + c;
}

template <typename dataType>
__global__ void product(	const dataType* lhs,
							const dataType* rhs,
							dataType*		out,
							dataType		c)
{
	size_t ne = gridDim.x * blockDim.x;
	size_t nx = gridDim.z * blockDim.x;

	size_t ide = blockIdx.x * blockDim.x;
	size_t idx = blockIdx.z * blockDim.x + threadIdx.x;
	size_t idy = blockIdx.y * blockDim.y + threadIdx.y;

	__shared__ dataType lTile[32][32];
	__shared__ dataType rTile[32][32];

	lTile[threadIdx.y][threadIdx.x] = lhs[idy * ne + ide + threadIdx.x];
	rTile[threadIdx.y][threadIdx.x] = rhs[(ide + threadIdx.y) * nx + idx];

	__syncthreads();

	dataType sum = 0;
	for(int i = 0; i < blockDim.x; i++)
		sum += lTile[threadIdx.y][i] * rTile[i][threadIdx.x];

	atomicAdd(&out[idy * nx + idx], c*sum);
}

template <typename dataType>
__global__ void transpose(	const dataType* src,
							dataType*		dst,
							size_t			src_stride,
							size_t			dst_stride)
{
	size_t inx = blockIdx.x * blockDim.x + threadIdx.x;
	size_t iny = blockIdx.y * blockDim.y + threadIdx.y;

	size_t outx = blockIdx.y * blockDim.y + threadIdx.x;
	size_t outy = blockIdx.x * blockDim.x + threadIdx.y;

	__shared__ dataType srcTile[32][32];
	srcTile[threadIdx.y][threadIdx.x] = src[iny*src_stride + inx];

	__syncthreads();

	dst[outy*dst_stride + outx] = srcTile[threadIdx.x][threadIdx.y];
}


} // kernels namespace
} // cm namespace