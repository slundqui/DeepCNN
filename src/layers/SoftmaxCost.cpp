/**
 * SoftmaxCost.cpp
 * Author: Sheng Lundquist
 **/

#include "SoftmaxCost.hpp"
#include "../utils.hpp"
#include "../cuda_utils.hpp"
#include <cuda_runtime.h>
#include "../kernels.hpp"


SoftmaxCost::SoftmaxCost()
{
   totalCostBlockSize = 0;
   totalCostGridSize = 0;
   calcGradBlockSize = 0;
   calcGradGridSize = 0;
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

   int batchcount = bSize * fSize * xSize * ySize;
   softmaxTotalCostRunSize(&totalCostGridSize, &totalCostBlockSize, batchcount);

   leastSqCalcGradRunSize(&calcGradGridSize, &calcGradBlockSize, batchcount);

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
   int batchcount = bSize * fSize * xSize * ySize;
   softmaxTotalCost(truth, d_AData, batchcount, d_TotalCost, totalCostGridSize, totalCostBlockSize); 
   return SUCCESS;
}

int SoftmaxCost::calcGradient(){
   float* truth = col->getGroundTruthLayer()->getDeviceA();
   int batchcount = bSize * fSize * xSize * ySize;
   float alpha = 1;
   float beta = 0;
   cudnnHandle_t handle = col->getCudnnHandle();

   //Same gradient calculation as leastSq (est - truth)
   leastSqCalcGrad(truth, d_AData, batchcount, d_GData, calcGradGridSize, calcGradBlockSize);
   return SUCCESS;
}

