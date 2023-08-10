#ifndef CUDAMATH_H
#define CUDAMATH_H

#include <iostream>
#include <cstring>
#include <memory>
#include <vector>
#include <cmath>

#include <cuda_runtime_api.h>
#include <cuda_profiler_api.h>

#include <boost/numeric/ublas/io.hpp>
#include <boost/numeric/ublas/matrix.hpp>

#define ASSERT(condition) { if(!(condition)) {std::cout << "\nAssert Failed: " << #condition << " @ " << __FILE__ << ", line " << __LINE__ << '\n';}}
#define PRINT_LINE	{std::cout << "\nLine: " << __LINE__ << "\nFile: " << __FILE__ << "\n\n";}

namespace cm
{
	size_t	get_pow_2(size_t n);				// Get the nearest power of 2 bigger than n
	size_t	next_multiple(size_t n, size_t f);	// Get the next multiple of f bigger than n
	size_t	factor_of(size_t n, size_t max);	// Get the largest factor of n <= max
	bool	is_pow_2(size_t n);					// Check if n is a power of 2

	//	Random number generator
	template <typename dataType>
	dataType rng(dataType min, dataType max)
	{		
		static constexpr double fraction { 1.0 / ( 1.0 + RAND_MAX ) };
		return static_cast<dataType>( min + ( (max - min) * ( std::rand() * fraction ) ) );
	}

	//	Check Cuda Error
	inline void checkLastCudaError(const std::string& str = "")
	{
		auto err = cudaGetLastError();
		if(err != cudaSuccess)
			std::cout << "\n" << str << " Error:\t" << cudaGetErrorString(err) << '\n';
	}


	inline void init()
	{
		std::srand(static_cast<unsigned int>(time(0)));		
	}

} // cm namespace

#endif