/**
 * BaseLayer.cpp
 * Author: Sheng Lundquist
 **/

#include "BaseLayer.hpp"

BaseLayer::BaseLayer(std::string layerName)
{
   d_AData = NULL;
   paramsSet = false;
   name = layerName;
   bSize = 0;
   ySize = 0;
   wSize = 0;
   fSize = 0;
}

int BaseLayer::setParams(int in_bSize, int in_ySize, int in_wSize, int in_fSize){
   bSize = in_bSize;
   ySize = in_ySize;
   wSize = in_wSize;
   fSize = in_fSize;
   paramsSet = true;
}


int BaseLayer::initialize(){
   if(!paramsSet){
      std::cerr << "Error! Layer " << name << " did not set parameters before trying to initialize\n";
      exit(UNDEFINED_PARAMS);
   }
   //Allocate d_AData based on size parameters
   size_t memSize = bSize * ySize * wSize * fSize * sizeof(float);
   //Testing error calling
   CudaError( cudaMalloc(&d_AData, 0));
   return SUCCESS;
}


BaseLayer::~BaseLayer(){
   CudaError( cudaFree(d_AData));
}

