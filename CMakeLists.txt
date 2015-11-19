# This CMakeLists.txt file can be used to generate makefiles for several projects at once.
# It should be put in the directory containing the projects.

cmake_minimum_required(VERSION 2.8)

set(CUDNN_PATH ~/cuDNN/cudnn-7.0-linux-x64-v3 CACHE PATH "Location of cuDNN libraries")
set(GTEST_PATH ~/gtest/gtest-1.7.0 CACHE PATH "Location of gtest library")
set(MATIO_PATH ~/matio/build CACHE PATH "Location of matio library")
set(CIMG_PATH ~/cimg/CImg-1.6.8_pre102815/ CACHE PATH "Location of cimg library")

#Set output bin directory
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "bin/")

find_package(CUDA QUIET REQUIRED)

#Set various cuda flags
#NOTE the arch used here is for a gtx 960, which is in the machien sisters.
#Change arch type as need be
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -arch=sm_52;)
set(CMAKE_CXX_FLAGS "-W -Wall -Wno-unused-parameter -Wno-unused-variable")

if (CMAKE_BUILD_TYPE STREQUAL "Debug")
   set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -Xptxas; -v; -keep; -lineinfo; -g; -G;)
else (CMAKE_BUILD_TYPE STREQUAL "Debug")
   set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -O;)
endif (CMAKE_BUILD_TYPE STREQUAL "Debug")

include_directories(
   src
   ${CUDNN_PATH}/include
   ${GTEST_PATH}/include
   ${MATIO_PATH}/include
   ${CIMG_PATH}
   ${CMAKE_CURRENT_SOURCE_DIR}
)

link_directories(
   ${CUDNN_PATH}/lib64
   ${GTEST_PATH}
   ${MATIO_PATH}/lib
)

#Get all files with extensions in the src directory
file(GLOB tmp src/*.[ch]pp)
set(LibSrc ${LibSrc} ${tmp})
file(GLOB tmp src/*.cu)
set(LibSrc ${LibSrc} ${tmp})
file(GLOB tmp src/layers/*.[ch]pp)
set(LibSrc ${LibSrc} ${tmp})
file(GLOB tmp src/layers/*.cu)
set(LibSrc ${LibSrc} ${tmp})
file(GLOB tmp src/connections/*.[ch]pp)
set(LibSrc ${LibSrc} ${tmp})

file(GLOB testLibSrc tests/*.[ch]pp)

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
   ${testLibSrc}
   ${LibSrc}
)

#Link libraries
target_link_libraries(
   test
   -lcudnn
   -lgtest
   -lpthread
   -lmatio
)

