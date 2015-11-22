#include "cuda_utils.hpp"
#include "utils.hpp"
#include <curand.h>
//CUDA kernel
__global__ void k_setArray(float* array, int count, float initVal){
   int idx = blockIdx.x * blockDim.x + threadIdx.x;
   if(idx<count) array[idx] = initVal;
}

void setArray(float* array, int count, float initVal){
   int block_size;
   int minGridSize;

   CudaError(cudaOccupancyMaxPotentialBlockSize(&minGridSize, &block_size, k_setArray, 0, count));
   int n_blocks = count/block_size + (count%block_size == 0 ? 0 : 1);

   k_setArray <<< n_blocks, block_size >>> (array, count, initVal);
   CudaCallError();
}

void calcRunSize(void* kernel, int* gridSize, int* blockSize, int batchcount){
   int minGridSize;
   CudaError(cudaOccupancyMaxPotentialBlockSize(&minGridSize, blockSize, kernel, 0, batchcount));
   (*gridSize) = batchcount/(*blockSize) + (batchcount%(*blockSize) == 0 ? 0 : 1);
}
