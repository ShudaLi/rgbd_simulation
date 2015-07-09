#The name of the project
project(ICL)

#The CMake Minimum version that is required. The FindCUDA script
#is distributed since version 2.8
CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

find_package(CUDA)
include(FindCUDA)

find_package( OpenCV REQUIRED PATHS $ENV{OPENCV_NEW} NO_DEFAULT_PATH )
include_directories ( $ENV{OPENCV_NEW}/include/ )

include_directories ( $ENV{BOOST_INC} )
include_directories ( $ENV{EIGEN_INC} )

link_directories ( $ENV{BOOST_LIB} )

#The rule to build the executable program matrixMul from matrixmul.cu
CUDA_ADD_EXECUTABLE(ICL add_kinect_noise.cu add_kinect_noise.cuh VaFRIC.h vector_math.hpp VaFRIC.cpp test_vaFRIC.cpp)
target_link_libraries( ICL ${OpenCV_LIBS} )


