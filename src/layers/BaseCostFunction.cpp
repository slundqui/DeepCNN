/**
 * BaseCostFunction.cpp
 * Author: Sheng Lundquist
 **/

#include "BaseCostFunction.hpp"
#include "../utils.hpp"
#include "../cuda_utils.hpp"


BaseCostFunction::BaseCostFunction()
{
   costFilename = "";
   accuracyFilename = "";
   d_TotalCost = NULL;
   h_estBuf = NULL;
   h_gtBuf = NULL;
   numCorrect = 0;
   numTests = 0;
   sumCost = 0;
   currCost = 0;
   currCorrect = 0;
   writePeriod = 0;
}

BaseCostFunction::~BaseCostFunction(){
   CudaError(cudaFree(d_TotalCost));
   if(costFilename != ""){
      costFile.close();
   }
   if(accuracyFilename != ""){
      accuracyFile.close();
   }
   if(estFilename != ""){
      estFile.close();
   }
   free(h_estBuf);
   free(h_gtBuf);
}

int BaseCostFunction::setParams(Column* c, std::string layerName, std::string activationType, int in_writePeriod, std::string in_costFilename, std::string in_accuracyFilename, std::string in_estFilename){
   costFilename = in_costFilename;
   accuracyFilename = in_accuracyFilename;
   estFilename = in_estFilename;
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

   if(costFilename != ""){
      costFile.open(costFilename.c_str(), std::ofstream::out);
      if(!costFile.is_open()){
         std::cerr << "Error opening file " << costFilename << " for writing.\n";
         exit(FILEIO_ERROR);
      }
   }
   if(accuracyFilename != ""){
      accuracyFile.open(accuracyFilename.c_str(), std::ofstream::out);
      if(!accuracyFile.is_open()){
         std::cerr << "Error opening file " << accuracyFilename << " for writing.\n";
         exit(FILEIO_ERROR);
      }
   }
   if(estFilename != ""){
      estFile.open(estFilename.c_str(), std::ofstream::out);
      if(!estFile.is_open()){
         std::cerr << "Error opening file " << estFilename << " for writing.\n";
         exit(FILEIO_ERROR);
      }
   }
   return SUCCESS;
}

int BaseCostFunction::allocate(){
   Activation::allocate();
   CudaError(cudaMalloc(&d_TotalCost, sizeof(float)));
   CudaError(cudaMemset(d_TotalCost, 0, sizeof(float)));

   h_estBuf = (float*) malloc(gpuDataSize);
   h_gtBuf = (float*) malloc(gpuDataSize);

   return SUCCESS;
}

float BaseCostFunction::getCurrentCost(){
   return currCost;
}

float BaseCostFunction::getAverageCost(){
   if(numTests == 0){
      return 0;
   }
   else{
      return sumCost/(float)numTests;
   }
}

float BaseCostFunction::getAccuracy(){
   if(numTests == 0){
      return 0;
   }
   else{
      return (float)numCorrect/(float)numTests;
   }
}

int BaseCostFunction::writeEst(){
   int count = xSize * ySize * fSize;
   for(int bi = 0; bi < bSize; bi++){
      float maxGt = h_gtBuf[0];
      float maxEst = h_estBuf[0];
      int maxGtIdx = 0;
      int maxEstIdx = 0;
      for(int i = 0; i < count; i++){
         int idx = bi * count + i;
         if(maxEst < h_estBuf[idx]){
            maxEst = h_estBuf[idx];
            maxEstIdx = i;
         }
         if(maxGt < h_gtBuf[idx]){
            maxGt = h_gtBuf[idx];
            maxGtIdx = i;
         }
      }
      //GT, Est, Confidence
      estFile << maxGtIdx << "," << maxEstIdx << "," << maxEst << "\n";
   }
   return SUCCESS;
}

int BaseCostFunction::updateHostData(){
   float* d_GTData = col->getGroundTruthLayer()->getDeviceA();
   CudaError(cudaDeviceSynchronize());
   CudaError(cudaMemcpy(h_estBuf, d_AData, gpuDataSize, cudaMemcpyDeviceToHost));
   CudaError(cudaMemcpy(h_gtBuf, d_GTData, gpuDataSize, cudaMemcpyDeviceToHost));
   return SUCCESS;
}

int BaseCostFunction::forwardUpdate(int timestep){
   Activation::forwardUpdate(timestep);

   //timestep + 2 to keep stats until writePeriod
   if((timestep + 2) % writePeriod == 0){
      //reset counts
      reset();
   }

   updateHostData();

   currCost = calcCost();
   currCorrect = calcCorrect();
   sumCost += currCost;
   numCorrect += currCorrect;
   numTests += bSize;


   if(estFilename != ""){
      writeEst();
   }
   //timestep + 1 to skip timestep 0
   if((timestep+1) % writePeriod == 0){
      if(costFilename != ""){
         costFile << timestep << "," << getAverageCost() << std::endl;
      }
      if(accuracyFilename != ""){
         accuracyFile << timestep << "," << getAccuracy() << std::endl;
         std::cout << "Timestep: " << timestep << " accuracy " << getAccuracy() << "\n";
      }
   }

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



