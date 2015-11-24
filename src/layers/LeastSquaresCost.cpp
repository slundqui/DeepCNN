/**
 * LeastSquaresCost.cpp
 * Author: Sheng Lundquist
 **/

#include "LeastSquaresCost.hpp"
#include "../utils.hpp"
#include "../cuda_utils.hpp"
#include <cuda_runtime.h>
#include "../kernels.hpp"
#include <cmath>

LeastSquaresCost::LeastSquaresCost()
{
   totalCostBlockSize = 0;
   totalCostGridSize = 0;
   calcGradBlockSize = 0;
   calcGradGridSize = 0;
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
   int batchcount = bSize * fSize * xSize * ySize;
   leastSqTotalCostRunSize(&totalCostGridSize, &totalCostBlockSize, batchcount);
   leastSqCalcGradRunSize(&calcGradGridSize, &calcGradBlockSize, batchcount);

   return SUCCESS;
}

int LeastSquaresCost::allocate(){
   BaseCostFunction::allocate();
   return SUCCESS;
}

int LeastSquaresCost::calcTotalCost(){
   float* truth = col->getGroundTruthLayer()->getDeviceA();
   int batchcount = bSize * fSize * xSize * ySize;
   leastSqTotalCost(truth, d_AData, batchcount, bSize, d_TotalCost, totalCostGridSize, totalCostBlockSize); 
   return SUCCESS;
}

int LeastSquaresCost::calcGradient(){
   float* truth = col->getGroundTruthLayer()->getDeviceA();
   int batchcount = bSize * fSize * xSize * ySize;
   float alpha = 1;
   float beta = 0;

   leastSqCalcGrad(truth, d_AData, batchcount, bSize, d_GAData, calcGradGridSize, calcGradBlockSize);
   return SUCCESS;
}

int LeastSquaresCost::calcAccuracy(){
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

