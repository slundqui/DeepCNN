/**
 * LeastSquaresCost.cu
 * Author: Sheng Lundquist
 **/

#include "../utils.hpp"
#include "../cuda_utils.hpp"

extern "C" void leastSqTotalCost(float* estimate, float* truth, int batchcount, float* out, int n_blocks, int block_size);
extern "C" void calcSizeTotalCost(int* h_block_size, int* h_n_blocks, int batchcount);

extern "C" void leastSqCalcGrad(float* estimate, float* truth, int batchcount, float* out, int n_blocks, int block_size);
extern "C" void calcSizeCalcGrad(int* h_block_size, int* h_n_blocks, int batchcount);

//CUDA kernel
__global__ void k_leastSqTotalCost(float* estimate, float* truth, int batchcount, float* out){
   //Linear index into batch and features
   int idx = blockIdx.x * blockDim.x + threadIdx.x;
   //If within range
   if(idx < batchcount){
      //Calculate sum of position
      float sumMe = .5 * pow(estimate[idx] - truth[idx], 2);
      //Atomic add into output
      atomicAdd(out, sumMe);
   }
}

__global__ void k_leastSqCalcGrad(float* estimate, float* truth, int batchcount, float* out){
   //Linear index into batch and features
   int idx = blockIdx.x * blockDim.x + threadIdx.x;
   //If within range
   if(idx < batchcount){
      //Is this supposed to be estimate - truth?
      //out[idx] = estimate[idx] - truth[idx];
      out[idx] = (truth[idx] - estimate[idx]);
   }
}

void calcSizeTotalCost(int* h_block_size, int* h_n_blocks, int batchcount){
   int minGridSize;
   //Calculate efficient block and grid size
   CudaError(cudaOccupancyMaxPotentialBlockSize(&minGridSize, h_block_size, &k_leastSqTotalCost, 0, batchcount));
   (*h_n_blocks) = batchcount/(*h_block_size) + (batchcount%(*h_block_size) == 0 ? 0 : 1);
}

void calcSizeCalcGrad(int* h_block_size, int* h_n_blocks, int batchcount){
   int minGridSize;
   //Calculate efficient block and grid size
   CudaError(cudaOccupancyMaxPotentialBlockSize(&minGridSize, h_block_size, &k_leastSqTotalCost, 0, batchcount));
   (*h_n_blocks) = batchcount/(*h_block_size) + (batchcount%(*h_block_size) == 0 ? 0 : 1);
}

void leastSqTotalCost(float* estimate, float* truth, int batchcount, float* out, int n_blocks, int block_size){
   //Reset final cost device variable
   CudaError(cudaMemset(out, 0, sizeof(float)));
   CudaError(cudaDeviceSynchronize());
   k_leastSqTotalCost<<< n_blocks, block_size >>> (estimate, truth, batchcount, out);
   CudaError(cudaDeviceSynchronize());
   CudaCallError();
}

void leastSqCalcGrad(float* estimate, float* truth, int batchcount, float* out, int n_blocks, int block_size){
   //Reset final cost device variable
   CudaError(cudaDeviceSynchronize());
   k_leastSqCalcGrad<<< n_blocks, block_size >>> (estimate, truth, batchcount, out);
   CudaError(cudaDeviceSynchronize());
   CudaCallError();
}


