/**
 * Convolution.cu
 * Author: Sheng Lundquist
 **/

#include "../utils.hpp"
#include "../cuda_utils.hpp"
#include "../kernels.hpp"

extern "C" void convLearningRule(float* d_Weight, float* d_dWeight, float* d_GWeight, int count, float eps, float mom, float decay, int n_blocks, int block_size); 
extern "C" void calcSizeLearn(int* h_block_size, int* h_n_blocks, int count);


//CUDA kernel
__global__ void k_convLearningRule(float* d_Weight, float* d_dWeight, float* d_GWeight, int count, float eps, float mom, float decay){ 
   //Linear index into batch and features
   int idx = blockIdx.x * blockDim.x + threadIdx.x;
   //If within range
   if(idx < count){
      //d_dWeight[idx] = mom*d_dWeight[idx] - decay*eps*d_Weight[idx] - eps * d_GWeight[idx];
      //d_Weight[idx] += d_dWeight[idx];
      d_Weight[idx] = d_Weight[idx] + (eps * d_GWeight[idx]);
   }
}

void convLearningRule(float* d_Weight, float* d_dWeight, float* d_GWeight, int count, float eps, float mom, float decay, int n_blocks, int block_size){ 
   //Reset final cost device variable
   CudaError(cudaDeviceSynchronize());
   k_convLearningRule<<< n_blocks, block_size >>> (d_Weight, d_dWeight, d_GWeight, count, eps, mom, decay);
   CudaCallError();
   CudaError(cudaDeviceSynchronize());
}

void convLearningRuleRunSize(int* gridSize, int* blockSize, int count){
   calcRunSize((void*)&k_convLearningRule, gridSize, blockSize, count);
}

