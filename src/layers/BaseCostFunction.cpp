/**
 * BaseCostFunction.cpp
 * Author: Sheng Lundquist
 **/

#include "BaseCostFunction.hpp"
#include "../utils.hpp"
#include "../cuda_utils.hpp"


BaseCostFunction::BaseCostFunction()
{
}

BaseCostFunction::~BaseCostFunction(){
   CudaError(cudaFree(d_TotalCost));
   free(h_TotalCost);
}

int BaseCostFunction::setParams(Column* c, std::string layerName, std::string outCostFile, std::string outAccuracyFile){
   if(outCostFile != ""){
      //TODO open cost file for writing
   }
   if(outAccuracyFile != ""){
      //TODO open accuracy file for writing
   }
   return BaseLayer::setParams(c, layerName);
}

int BaseCostFunction::initialize(){
   BaseLayer::initialize();
   //Make sure ground truth layer exists and is the same size as this layer
   BaseLayer* groundTruth = col->getGroundTruthLayer();
   assert(xSize == groundTruth->getXSize());
   assert(ySize == groundTruth->getYSize());
   assert(fSize == groundTruth->getFSize());
   assert(bSize == groundTruth->getBSize());
   //Make sure cost function is the last layer
   assert(nextConn == NULL);
   return SUCCESS;
}

int BaseCostFunction::allocate(){
   BaseLayer::allocate();
   CudaError(cudaMalloc(&d_TotalCost, bSize * sizeof(float)));
   h_TotalCost = (float*)malloc(bSize * sizeof(float));
   
   return SUCCESS;
}

const float* BaseCostFunction::getHostTotalCost(){
   //calculate total cost
   calcTotalCost();
   CudaError(cudaDeviceSynchronize());
   CudaError(cudaMemcpy(h_TotalCost, d_TotalCost, bSize*sizeof(float), cudaMemcpyDeviceToHost));
   CudaError(cudaDeviceSynchronize());
   return h_TotalCost;
}

int BaseCostFunction::forwardUpdate(int timestep){
   BaseLayer::forwardUpdate(timestep);
   //Calculate accuracy

   return SUCCESS;
}

int BaseCostFunction::backwardsUpdate(int timestep){
   BaseLayer::backwardsUpdate(timestep);
   //Sets gradient based on cost function subclass
   calcGradient();
   return SUCCESS;
};



