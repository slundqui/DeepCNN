/**
 * LeastSquaresCost.cu
 * Author: Sheng Lundquist
 **/

#include "../utils.hpp"
#include "../cuda_utils.hpp"
#include "../kernels.hpp"


//CUDA kernel
__global__ void k_leastSqTotalCost(float* truth, float* estimate, int batchcount, float* out){
   //Linear index into batch and features
   int idx = blockIdx.x * blockDim.x + threadIdx.x;
   //If within range
   if(idx < batchcount){
      //Calculate sum of position
      float sumMe = .5 * pow(truth[idx] - estimate[idx], 2);
      //Atomic add into output
      atomicAdd(out, sumMe);
   }
}

__global__ void k_leastSqCalcGrad(float* truth, float* estimate, int batchcount, float* out){
   //Linear index into batch and features
   int idx = blockIdx.x * blockDim.x + threadIdx.x;
   //If within range
   if(idx < batchcount){
      //Is this supposed to be estimate - truth?
      out[idx] = (truth[idx] - estimate[idx]);
   }
}

void leastSqTotalCostRunSize(int* gridSize, int* blockSize, int batchcount){
   calcRunSize((void*)&k_leastSqTotalCost, gridSize, blockSize, batchcount);
}

void leastSqCalcGradRunSize(int* gridSize, int* blockSize, int batchcount){
   calcRunSize((void*)&k_leastSqCalcGrad, gridSize, blockSize, batchcount);
}

void leastSqTotalCost(float* truth, float* estimate, int batchcount, float* out, int gridSize, int blockSize){
   //Reset final cost device variable
   CudaError(cudaMemset(out, 0, sizeof(float)));
   CudaError(cudaDeviceSynchronize());
   k_leastSqTotalCost<<< gridSize, blockSize>>> (truth, estimate, batchcount, out);
   CudaError(cudaDeviceSynchronize());
   CudaCallError();
}

void leastSqCalcGrad(float* truth, float* estimate, int batchcount, float* out, int gridSize, int blockSize){
   //Reset final cost device variable
   CudaError(cudaDeviceSynchronize());
   k_leastSqCalcGrad<<< gridSize, blockSize>>> (truth, estimate, batchcount, out);
   CudaError(cudaDeviceSynchronize());
   CudaCallError();
}


