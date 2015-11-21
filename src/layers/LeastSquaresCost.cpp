/**
 * LeastSquaresCost.cpp
 * Author: Sheng Lundquist
 **/

#include "LeastSquaresCost.hpp"
#include "../utils.hpp"
#include "../cuda_utils.hpp"
#include <cuda_runtime.h>

extern "C" void leastSqTotalCost(float* estimate, float* truth, int count, int nbatch, float* out, int n_blocks, int block_size);
extern "C" void calcSizeTotalCost(int* h_block_size, int* h_n_blocks, int fSize);
extern "C" void leastSqCalcGrad(float* estimate, float* truth, int batchcount, float* out, int n_blocks, int block_size);
extern "C" void calcSizeCalcGrad(int* h_block_size, int* h_n_blocks, int batchcount);

LeastSquaresCost::LeastSquaresCost()
{
   totalcost_block_size = 0;
   totalcost_n_blocks = 0;
}

LeastSquaresCost::~LeastSquaresCost(){
}

//int LeastCostFunction::setParams(Column* c, std::string layerName, std::string outCostFile, std::string outAccuracyFile){
//   if(outCostFile != ""){
//      //TODO open cost file for writing
//   }
//   return BaseLayer::setParams(c, layerName);
//}

int LeastSquaresCost::initialize(){
   BaseCostFunction::initialize();

   //Currently only allowing 1x1xf connections
   assert(xSize == 1 && ySize == 1);
   calcSizeTotalCost(&totalcost_block_size, &totalcost_n_blocks, bSize);
   calcSizeCalcGrad(&calcgrad_block_size, &calcgrad_n_blocks, bSize*fSize);

   return SUCCESS;
}

int LeastSquaresCost::allocate(){
   BaseCostFunction::allocate();
   return SUCCESS;
}

int LeastSquaresCost::applyActivation(){
   float alpha = 1;
   float beta = 0;
   cudnnHandle_t handle = col->getCudnnHandle();
   CudaError(cudaDeviceSynchronize());
   CudnnError(cudnnActivationForward(
      handle,
      CUDNN_ACTIVATION_SIGMOID,
      &alpha,
      layerDescriptor,
      d_UData,
      &beta,
      layerDescriptor,
      d_AData));
   return SUCCESS;
}

int LeastSquaresCost::calcTotalCost(){
   float* truth = col->getGroundTruthLayer()->getDeviceA();
   int count = fSize * xSize * ySize;
   leastSqTotalCost(d_AData, truth, count, bSize, d_TotalCost, totalcost_n_blocks, totalcost_block_size); 
   return SUCCESS;
}

int LeastSquaresCost::calcGradient(){
   float* truth = col->getGroundTruthLayer()->getDeviceA();
   int batchcount = bSize * fSize * xSize * ySize;
   float alpha = 1;
   float beta = 0;
   cudnnHandle_t handle = col->getCudnnHandle();

   leastSqCalcGrad(d_AData, truth, batchcount, d_GData, calcgrad_n_blocks, calcgrad_block_size);
   CudaError(cudaDeviceSynchronize());
   CudnnError(cudnnActivationBackward(
      handle,
      CUDNN_ACTIVATION_SIGMOID,
      &alpha,
      layerDescriptor, //Layer src data, postactivation buffer
      d_AData,
      layerDescriptor, //Layer srcDiffData, gradients
      d_GData,
      layerDescriptor, //destData, preactivation buffer
      d_UData,
      &beta,
      layerDescriptor, //Layer destDiffData, gradients
      d_GData));
   return SUCCESS;
}

