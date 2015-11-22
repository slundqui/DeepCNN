#include "../utils.hpp"
#include "../cuda_utils.hpp"
#include "../kernels.hpp"

//extern "C" void softmaxTotalCost(float* estimate, float* truth, int count, int nbatch, float* out, int n_blocks, int block_size);
//extern "C" void softmaxTotalCostRunSize(int* h_block_size, int* h_n_blocks, int fSize);

//CUDA kernel
__global__ void k_SoftmaxTotalCost(float* truth, float* estimate, int batchcount, float* out){
   //Linear index into batch and features
   int idx = blockIdx.x * blockDim.x + threadIdx.x;
   //If within range
   if(idx < batchcount){
      //Calculate sum of position
      float sumMe = -1 * truth[idx] * log(estimate[idx]);
      //Atomic add into output
      atomicAdd(out, sumMe);
   }
}

void softmaxTotalCostRunSize(int* gridSize, int* blockSize, int batchcount){
   calcRunSize((void*)&k_SoftmaxTotalCost, gridSize, blockSize, batchcount);
}

void softmaxTotalCost(float* estimate, float* truth, int batchcount, float* out, int n_blocks, int block_size){
   //Reset final cost device variable
   CudaError(cudaMemset(out, 0, sizeof(float)));
   CudaError(cudaDeviceSynchronize());
   k_SoftmaxTotalCost<<< n_blocks, block_size >>> (truth, estimate, batchcount, out);
   CudaError(cudaDeviceSynchronize());
   CudaCallError();
}
