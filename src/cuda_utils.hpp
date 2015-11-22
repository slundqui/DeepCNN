#ifndef CUDA_UTILS_HPP_ 
#define CUDA_UTILS_HPP_ 

extern "C" void setArray(float* array, int count, float initVal);
extern "C" void calcRunSize(void* kernel, int* gridSize, int* blockSize, int batchcount);

#endif
