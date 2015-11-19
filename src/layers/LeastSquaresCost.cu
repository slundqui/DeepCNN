/**
 * LeastSquaresCost.cu
 * Author: Sheng Lundquist
 **/

#include "../utils.hpp"
#include "../cuda_utils.hpp"

extern "C" void leastSqTotalCost(float* estimate, float* truth, int count, int nbatch, float* out, int n_blocks, int block_size);
extern "C" void calcSizeTotalCost(int* h_block_size, int* h_n_blocks, int fSize);
extern "C" void leastSqCalcGrad(float* estimate, float* truth, int batchcount, float* out, int n_blocks, int block_size);
extern "C" void calcSizeCalcGrad(int* h_block_size, int* h_n_blocks, int batchcount);

//CUDA kernel
__global__ void k_leastSqTotalCost(float* estimate, float* truth, int count, int nbatch, float* out){
   //Linear index into batch and features
   int idx = blockIdx.x * blockDim.x + threadIdx.x;
   //If within range
   if(idx < nbatch*count){
      //Grab batch index
      int batchIdx = idx/count;
      //Calculate sum of position
      float sumMe = .5 * pow(truth[idx] - estimate[idx], 2);
      //Atomic add into output
      atomicAdd(&(out[batchIdx]), sumMe);
   }
}

__global__ void k_leastSqCalcGrad(float* estimate, float* truth, int batchcount, float* out){
   //Linear index into batch and features
   int idx = blockIdx.x * blockDim.x + threadIdx.x;
   //If within range
   if(idx < batchcount){
      //Is this supposed to be estimate - truth?
      //out[idx] = estimate[idx] - truth[idx];
      out[idx] = truth[idx] - estimate[idx];
   }
}

void calcSizeTotalCost(int* h_block_size, int* h_n_blocks, int fSize){
   int minGridSize;
   //Calculate efficient block and grid size
   CudaError(cudaOccupancyMaxPotentialBlockSize(&minGridSize, h_block_size, &k_leastSqTotalCost, 0, fSize));
   (*h_n_blocks) = fSize/(*h_block_size) + (fSize%(*h_block_size) == 0 ? 0 : 1);
}

void calcSizeCalcGrad(int* h_block_size, int* h_n_blocks, int batchcount){
   int minGridSize;
   //Calculate efficient block and grid size
   CudaError(cudaOccupancyMaxPotentialBlockSize(&minGridSize, h_block_size, &k_leastSqTotalCost, 0, batchcount));
   (*h_n_blocks) = batchcount/(*h_block_size) + (batchcount%(*h_block_size) == 0 ? 0 : 1);
}

void leastSqTotalCost(float* estimate, float* truth, int count, int bSize, float* out, int n_blocks, int block_size){
   //Reset final cost device variable
   CudaError(cudaMemset(out, 0, count*bSize*sizeof(float)));
   CudaError(cudaDeviceSynchronize());
   k_leastSqTotalCost<<< n_blocks, block_size >>> (estimate, truth, count, bSize, out);
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


