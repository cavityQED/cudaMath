add_executable				(arrayTest	arrayTest.cpp)
target_include_directories	(arrayTest	PRIVATE ${CMAKE_SOURCE_DIR}/linearAlgebra)
target_include_directories	(arrayTest	PRIVATE ${CMAKE_SOURCE_DIR}/utilities)
target_link_libraries		(arrayTest	PRIVATE cudaMath utilities)