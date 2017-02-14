#The name of the project
project(ICL)

#The CMake Minimum version that is required. The FindCUDA script
#is distributed since version 2.8
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

find_package(CUDA)
include(FindCUDA)
find_package(Boost ${QUIET_} REQUIRED COMPONENTS filesystem system)

find_package( OpenCV REQUIRED )

include_directories ( $ENV{BOOST_INC} )
include_directories ( $ENV{SOPHUS_INC} )
include_directories ( $ENV{EIGEN_INC} )

if(UNIX)
	find_package( Eigen3 REQUIRED)	
	find_package( Sophus REQUIRED)
endif()

link_directories ( $ENV{BOOST_LIB} )

#The rule to build the executable program matrixMul from matrixmul.cu
CUDA_ADD_EXECUTABLE(ICL 
	 	   src/add_kinect_noise.cu 
		   src/add_kinect_noise.cuh 
 		   src/VaFRIC.h 
		   src/vector_math.hpp 
		   src/VaFRIC.cpp 
		   src/test_vaFRIC.cpp)
target_link_libraries( ICL ${OpenCV_LIBS} ${Boost_FILESYSTEM_LIBRARY} ${Boost_SYSTEM_LIBRARY})


