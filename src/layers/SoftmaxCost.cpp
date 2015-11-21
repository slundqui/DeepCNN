/**
 * SoftmaxCost.cpp
 * Author: Sheng Lundquist
 **/

#include "SoftmaxCost.hpp"
#include "../utils.hpp"
#include "../cuda_utils.hpp"
#include <cuda_runtime.h>

extern "C" void softmaxTotalCost(float* estimate, float* truth, int count, int nbatch, float* out, int n_blocks, int block_size);
extern "C" void calcSizeSoftmaxCost(int* h_block_size, int* h_n_blocks, int fSize);

extern "C" void leastSqCalcGrad(float* estimate, float* truth, int batchcount, float* out, int n_blocks, int block_size);
extern "C" void calcSizeCalcGrad(int* h_block_size, int* h_n_blocks, int batchcount);

SoftmaxCost::SoftmaxCost()
{
   totalcost_block_size = 0;
   totalcost_n_blocks = 0;
}

SoftmaxCost::~SoftmaxCost(){
}

//int LeastCostFunction::setParams(Column* c, std::string layerName, std::string outCostFile, std::string outAccuracyFile){
//   if(outCostFile != ""){
//      //TODO open cost file for writing
//   }
//   return BaseLayer::setParams(c, layerName);
//}

int SoftmaxCost::initialize(){
   BaseCostFunction::initialize();

   //Currently only allowing 1x1xf connections
   assert(xSize == 1 && ySize == 1);
   calcSizeSoftmaxCost(&totalcost_block_size, &totalcost_n_blocks, bSize*fSize);
   calcSizeCalcGrad(&calcgrad_block_size, &calcgrad_n_blocks, bSize*fSize);

   return SUCCESS;
}

int SoftmaxCost::allocate(){
   BaseCostFunction::allocate();
   return SUCCESS;
}

//Softmax activation
int SoftmaxCost::applyActivation(){
   float alpha = 1;
   float beta = 0;
   cudnnHandle_t handle = col->getCudnnHandle();
   CudaError(cudaDeviceSynchronize());
   CudnnError(cudnnSoftmaxForward(
      handle,
      CUDNN_SOFTMAX_ACCURATE,
      CUDNN_SOFTMAX_MODE_CHANNEL,
      &alpha,
      layerDescriptor,
      d_UData,
      &beta,
      layerDescriptor,
      d_AData));
   return SUCCESS;
}

int SoftmaxCost::calcTotalCost(){
   float* truth = col->getGroundTruthLayer()->getDeviceA();
   int count = fSize * xSize * ySize;
   softmaxTotalCost(d_AData, truth, count, bSize, d_TotalCost, totalcost_n_blocks, totalcost_block_size); 
   return SUCCESS;
}

int SoftmaxCost::calcGradient(){
   float* truth = col->getGroundTruthLayer()->getDeviceA();
   int batchcount = bSize * fSize * xSize * ySize;
   float alpha = 1;
   float beta = 0;
   cudnnHandle_t handle = col->getCudnnHandle();

   //Same gradient calculation as leastSq (est - truth)
   leastSqCalcGrad(d_AData, truth, batchcount, d_GData, calcgrad_n_blocks, calcgrad_block_size);
   return SUCCESS;
}

