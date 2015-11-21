#include "../utils.hpp"
#include "../cuda_utils.hpp"

extern "C" void softmaxTotalCost(float* estimate, float* truth, int count, int nbatch, float* out, int n_blocks, int block_size);
extern "C" void calcSizeSoftmaxCost(int* h_block_size, int* h_n_blocks, int fSize);

//CUDA kernel
__global__ void k_SoftmaxTotalCost(float* estimate, float* truth, int count, int nbatch, float* out){
   //Linear index into batch and features
   int idx = blockIdx.x * blockDim.x + threadIdx.x;
   //If within range
   if(idx < nbatch*count){
      //Grab batch index
      int batchIdx = idx/count;
      //Calculate sum of position
      float sumMe = (-1/count) * truth[idx] * log(estimate[idx]);
      //Atomic add into output
      atomicAdd(&(out[batchIdx]), sumMe);
   }
}

void calcSizeSoftmaxCost(int* h_block_size, int* h_n_blocks, int count){
   int minGridSize;
   //Calculate efficient block and grid size
   CudaError(cudaOccupancyMaxPotentialBlockSize(&minGridSize, h_block_size, &k_SoftmaxTotalCost, 0, count));
   (*h_n_blocks) = count/(*h_block_size) + (count%(*h_block_size) == 0 ? 0 : 1);
}

void softmaxTotalCost(float* estimate, float* truth, int count, int bSize, float* out, int n_blocks, int block_size){
   //Reset final cost device variable
   CudaError(cudaMemset(out, 0, bSize*sizeof(float)));
   CudaError(cudaDeviceSynchronize());
   k_SoftmaxTotalCost<<< n_blocks, block_size >>> (estimate, truth, count, bSize, out);
   CudaError(cudaDeviceSynchronize());
   CudaCallError();
}
