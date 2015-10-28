# This CMakeLists.txt file can be used to generate makefiles for several projects at once.
# It should be put in the directory containing the projects.

cmake_minimum_required(VERSION 2.8)
set(CUDNN_PATH ~/cuDNN/cudnn-7.0-linux-x64-v3 CACHE PATH "Location of cuDNN libraries")
set(GTEST_PATH ~/gtest/gtest-1.7.0 CACHE PATH "Location of gtest library")

#Set output bin directory
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "bin/")

find_package(CUDA QUIET REQUIRED)

#Set various cuda flags
#NOTE the arch used here is for a gtx 960, which is in the machien sisters.
#Change arch type as need be
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -arch=sm_52; )

if (CMAKE_BUILD_TYPE STREQUAL "Debug")
   set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -Xptxas; -v; -keep; -lineinfo; -g; -G;)
else (CMAKE_BUILD_TYPE STREQUAL "Debug")
   set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -O;)
endif (CMAKE_BUILD_TYPE STREQUAL "Debug")

include_directories(
   src
   ${CUDNN_PATH}/include
   ${GTEST_PATH}/include
)

link_directories(
   ${CUDNN_PATH}/lib64
   ${GTEST_PATH}
)

#Get all files with extensions in the src directory
file(GLOB libSrcCPP src/*.cpp)
file(GLOB libSrcHPP src/*.hpp)
file(GLOB libSrcCU src/*.cu)

file(GLOB testSrcCPP tests/*.cpp)

##Add compilation files to NVCC
#cuda_add_executable(
#   depthMLP
#   ${libSrcCPP}
#   ${libSrcHPP}
#   ${libSrcCU}
#)

#target_link_libraries(
#   depthMLP
#   -lcudnn
#)

#Add compilation files to testing suite
cuda_add_executable(
   test
   ${testSrcCPP}
   ${libSrcCPP}
   ${libSrcHPP}
   ${libSrcCU}
)

target_link_libraries(
   test
   -lcudnn
   -lgtest
   -lpthread
)
