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
   numCorrect = 0;
   numTests = 0;
}

SoftmaxCost::~SoftmaxCost(){
}

int SoftmaxCost::setParams(Column* c, std::string layerName, 
      //std::string activationType,
      int in_writePeriod,
      std::string in_outCostFile,
      std::string in_outAccuracyFile){
   //Overwriting applyActivation to do softmax, so activation type here does not matter
   return BaseCostFunction::setParams(c, layerName, "relu", in_writePeriod, in_outCostFile, in_outAccuracyFile);
}

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

//Calculated in calcGradient
int SoftmaxCost::applyGradient(){
   calcGradient();
   CudaError(cudaDeviceSynchronize());
   CudaError(cudaMemcpy(d_GUData, d_GAData, gpuDataSize, cudaMemcpyDeviceToDevice));
   return SUCCESS;
}

int SoftmaxCost::calcTotalCost(){
   float* truth = col->getGroundTruthLayer()->getDeviceA();
   int batchcount = bSize * fSize * xSize * ySize;
   softmaxTotalCost(truth, d_AData, batchcount, bSize, d_TotalCost, totalCostGridSize, totalCostBlockSize); 
   return SUCCESS;
}

int SoftmaxCost::calcGradient(){
   //std::cout << "Softmax calculating gradient\n";
   float* truth = col->getGroundTruthLayer()->getDeviceA();
   int batchcount = bSize * fSize * xSize * ySize;
   //Same gradient calculation as leastSq (est - truth)
   leastSqCalcGrad(truth, d_AData, batchcount, bSize, d_GAData, calcGradGridSize, calcGradBlockSize);
   return SUCCESS;
}

int SoftmaxCost::calcAccuracy(){
   //Get activity based on threshold
   CudaError(cudaDeviceSynchronize());
   float* d_GTData = col->getGroundTruthLayer()->getDeviceA();
   CudaError(cudaMemcpy(h_estBuf, d_AData, gpuDataSize, cudaMemcpyDeviceToHost));
   CudaError(cudaMemcpy(h_gtBuf, d_GTData, gpuDataSize, cudaMemcpyDeviceToHost));
   CudaError(cudaDeviceSynchronize());
   float tolerance = 1e-6;

   int count = fSize * xSize * ySize;
   bool correct = true;
   //Each feature 
   for(int bi = 0; bi < bSize; bi++){
      float maxVal = h_estBuf[bi*count];
      int maxIdx = bi*count;
      for(int ni = 0; ni < count; ni++){ 
         int idx = bi*count + ni;
         if(h_estBuf[idx] > maxVal){ 
            maxVal = h_estBuf[idx];
            maxIdx = idx;
         }
      }
      //if GT[idx] == 1
      if(fabs(h_gtBuf[maxIdx] - (float)1) < tolerance){
         numCorrect++;
      }
      numTests++;
   }
   return SUCCESS;
}

