/**
 * BaseCostFunction.cpp
 * Author: Sheng Lundquist
 **/

#include "BaseCostFunction.hpp"
#include "../utils.hpp"
#include "../cuda_utils.hpp"


BaseCostFunction::BaseCostFunction()
{
   outCostFile = "";
   d_TotalCost = NULL;
   h_estBuf = NULL;
   h_gtBuf = NULL;
}

BaseCostFunction::~BaseCostFunction(){
   CudaError(cudaFree(d_TotalCost));
   if(outCostFile != ""){
      costFile.close();
   }
   free(h_estBuf);
   free(h_gtBuf);
}

int BaseCostFunction::setParams(Column* c, std::string layerName, std::string activationType, int in_writePeriod, std::string in_outCostFile, std::string in_outAccuracyFile){
   outCostFile = in_outCostFile;
   outAccuracyFile = in_outAccuracyFile;
   writePeriod = in_writePeriod;
   return Activation::setParams(c, layerName, activationType);
}

int BaseCostFunction::initialize(){
   Activation::initialize();
   //Make sure ground truth layer exists and is the same size as this layer
   BaseLayer* groundTruth = col->getGroundTruthLayer();
   assert(xSize == groundTruth->getXSize());
   assert(ySize == groundTruth->getYSize());
   assert(fSize == groundTruth->getFSize());
   assert(bSize == groundTruth->getBSize());
   //Make sure cost function is the last layer
   assert(nextConn == NULL);

   if(outCostFile != ""){
      costFile.open(outCostFile.c_str(), std::ofstream::out);
      assert(costFile.is_open());
   }
   if(outAccuracyFile != ""){
      
   }
   return SUCCESS;
}

int BaseCostFunction::allocate(){
   Activation::allocate();
   CudaError(cudaMalloc(&d_TotalCost, sizeof(float)));

   h_estBuf = (float*) malloc(gpuDataSize);
   h_gtBuf = (float*) malloc(gpuDataSize);
   
   return SUCCESS;
}

float BaseCostFunction::getHostTotalCost(){
   float h_TotalCost;
   //calculate total cost
   calcTotalCost();
   CudaError(cudaDeviceSynchronize());
   CudaError(cudaMemcpy(&h_TotalCost, d_TotalCost, sizeof(float), cudaMemcpyDeviceToHost));
   CudaError(cudaDeviceSynchronize());
   return h_TotalCost;
}

float BaseCostFunction::getHostAccuracy(){
   if(numTests == 0){
      return 0;
   }
   else{
      return (float)numCorrect/(float)numTests;
   }
}


int BaseCostFunction::forwardUpdate(int timestep){
   Activation::forwardUpdate(timestep);
   if(outCostFile != "" && (timestep + 1) % writePeriod == 0){
      costFile << timestep << "," << getHostTotalCost() << std::endl;
   }
   calcAccuracy();

   return SUCCESS;
}


int BaseCostFunction::applyGradient(){
   if(DEBUG) std::cout << "Cost function layer " << name << " applying gradient\n";
   //Sets gradient based on cost function subclass
   calcGradient();
   return Activation::applyGradient();
}

//int BaseCostFunction::backwardsUpdate(int timestep){
//   Activation::backwardsUpdate(timestep);
//   return SUCCESS;
//};



