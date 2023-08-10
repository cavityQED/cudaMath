cmake_minimum_required(VERSION 3.2)
set(CMAKE_CXX_COMPILER /usr/bin/g++)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_ARCHITECTURES "61")
project(cudaMath LANGUAGES CUDA CXX)

include_directories(${CMAKE_SOURCE_DIR})
include_directories(/usr/include/mpi)
include_directories(/usr/local/cuda-12/include)
include_directories(/home/mike/externalLibraries/boost_1_81_0)
include_directories(/home/mike/externalLibraries/boost_1_81_0/boost/numeric/ublas)

find_package(CUDA REQUIRED)
if(CUDA_FOUND)
	message("Found CUDA")
endif()

add_subdirectory(utilities)
add_subdirectory(linearAlgebra)

set(SOURCE_FILES	
					cudaMath.cu
					linearAlgebra/array.cpp)

add_library				(cudaMath SHARED cudaMath.cpp)
set_target_properties	(cudaMath PROPERTIES CUDA_SEPARABLE_COMPILIATION ON)
target_link_libraries	(cudaMath PUBLIC ${CUDA_LIBRARIES} linearAlgebra)

add_subdirectory(tests)