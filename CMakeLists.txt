#The name of the project
project(ICL)

#The CMake Minimum version that is required. The FindCUDA script
#is distributed since version 2.8
CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

find_package(CUDA)
include(FindCUDA)
find_package(Boost ${QUIET_} REQUIRED COMPONENTS filesystem system)

find_package( OpenCV REQUIRED PATHS $ENV{OPENCV_NEW} NO_DEFAULT_PATH )
include_directories ( $ENV{OPENCV_NEW}/include/ )

include_directories ( $ENV{BOOST_INC} )
include_directories ( $ENV{EIGEN_INC} )

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


