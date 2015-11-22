/**
 * LeastSquaresCost.cpp
 * Author: Sheng Lundquist
 **/

#include "LeastSquaresCost.hpp"
#include "../utils.hpp"
#include "../cuda_utils.hpp"
#include <cuda_runtime.h>
#include "../kernels.hpp"

//extern "C" void leastSqTotalCost(float* estimate, float* truth, int batchcount, float* out, int n_blocks, int block_size);
//extern "C" void leastSqTotalCostRunSize(int* h_block_size, int* h_n_blocks, int fSize);
//extern "C" void leastSqCalcGrad(float* estimate, float* truth, int batchcount, float* out, int n_blocks, int block_size);
//extern "C" void leastSqCalcGradRunSize(int* h_block_size, int* h_n_blocks, int batchcount);

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
   leastSqTotalCost(truth, d_AData, batchcount, d_TotalCost, totalCostGridSize, totalCostBlockSize); 
   return SUCCESS;
}

int LeastSquaresCost::calcGradient(){
   float* truth = col->getGroundTruthLayer()->getDeviceA();
   int batchcount = bSize * fSize * xSize * ySize;
   float alpha = 1;
   float beta = 0;

   leastSqCalcGrad(truth, d_AData, batchcount, d_GData, calcGradGridSize, calcGradBlockSize);
   return SUCCESS;
}

