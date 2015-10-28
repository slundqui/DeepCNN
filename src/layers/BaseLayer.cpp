/**
 * BaseLayer.cpp
 * Author: Sheng Lundquist
 **/

#include "BaseLayer.hpp"

BaseLayer::BaseLayer()
{
   d_AData = NULL;
   bSize = 0;
   ySize = 0;
   xSize = 0;
   fSize = 0;
   prevConn = NULL;
   nextConn = NULL;
}

//TODO fix the size paramters to be scales of column
int BaseLayer::setParams(std::string layerName, int in_bSize, int in_ySize, int in_xSize, int in_fSize){
   name = layerName;
   bSize = in_bSize;
   ySize = in_ySize;
   xSize = in_xSize;
   fSize = in_fSize;
   paramsSet = true;
}

int BaseLayer::initialize(){
   if(!paramsSet){
      std::cerr << "Error! Layer did not set parameters before trying to initialize\n";
      exit(UNDEFINED_PARAMS);
   }
   //Allocate d_AData based on size parameters
   size_t memSize = bSize * ySize * xSize * fSize * sizeof(float);
   //Testing error calling
   CudaError( cudaMalloc(&d_AData, memSize));
   return SUCCESS;
}


BaseLayer::~BaseLayer(){
   CudaError( cudaFree(d_AData));
}

