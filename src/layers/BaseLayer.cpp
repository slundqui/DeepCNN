/**
 * BaseLayer.cpp
 * Author: Sheng Lundquist
 **/

#include "BaseLayer.hpp"
#include "../Column.hpp"

BaseLayer::BaseLayer()
{
   d_AData = NULL;
   //h_AData = NULL;
   bSize = 0;
   ySize = 0;
   xSize = 0;
   fSize = 0;
   stride = 0;
   prevConn = NULL;
   nextConn = NULL;
}

BaseLayer::~BaseLayer(){
   CudaError( cudaFree(d_AData));
   //free(h_AData);
}

int BaseLayer::setParams(Column* c, std::string layerName, int in_stride, int num_features){
   name = layerName;
   stride = in_stride;
   fSize = num_features;
   bSize = c->getBSize();

   int colYSize = c->getYSize();
   assert(colYSize % stride == 0);
   ySize = colYSize/stride;

   int colXSize = c->getXSize();
   assert(colXSize % stride == 0);
   xSize = colXSize/stride;

   paramsSet = true;
   return SUCCESS;
}

float* BaseLayer::getDeviceAct(){
   size_t memSize = bSize * ySize * xSize * fSize * sizeof(float);
   float * h_outMem = (float*) malloc(memSize);
   CudaError(cudaMemcpy(h_outMem, d_AData, memSize, cudaMemcpyDeviceToHost));
   return h_outMem;
}

int BaseLayer::initialize(){
   if(!paramsSet){
      std::cerr << "Error! Layer did not set parameters before trying to initialize\n";
      exit(UNDEFINED_PARAMS);
   }
   //TODO get actual size based on scale and column parameters

   //Allocate d_AData based on size parameters
   size_t memSize = bSize * ySize * xSize * fSize * sizeof(float);

   CudaError( cudaMalloc(&d_AData, memSize));

   return SUCCESS;
}


