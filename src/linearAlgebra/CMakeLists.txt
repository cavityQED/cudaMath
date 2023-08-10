set(SOURCE_FILES	array.cu
					arrayOps.cu
					matrix.cu
					${CMAKE_SOURCE_DIR}/cudaMath.cpp)

add_library(linearAlgebra	SHARED	${SOURCE_FILES})
target_include_directories(linearAlgebra	PRIVATE ${CMAKE_SOURCE_DIR})