#include "BaseLayer.hpp"
#include "../connections/BaseConnection.hpp"
#include "../utils.hpp"
#include "../Column.hpp"

/**
 * BaseLayer.cpp
 * Author: Sheng Lundquist
 **/

BaseLayer::BaseLayer()
{
   d_AData = NULL;
   d_GData = NULL;
   //h_AData = NULL;
   bSize = 0;
   ySize = 0;
   xSize = 0;
   fSize = 0;
   prevConn = NULL;
   nextConn = NULL;
}

BaseLayer::~BaseLayer(){
   CudaError( cudaFree(d_AData));
   CudaError( cudaFree(d_GData));
   
   //Currently bugging out, maybe need to set the tensor first
   //CudnnError( cudnnDestroyTensorDescriptor(cudnnADescriptor));
   //CudnnError( cudnnDestroyTensorDescriptor(cudnnUDescriptor));

   //free(h_AData);
}

int BaseLayer::setParams(Column* c, std::string layerName){
   BaseData::setParams(c, layerName);
   bSize = c->getBSize();

   paramsSet = true;
   return SUCCESS;
}

float* BaseLayer::getHostA(){
   size_t memSize = bSize * ySize * xSize * fSize * sizeof(float);
   float * h_outMem = (float*) malloc(memSize);
   CudaError(cudaMemcpy(h_outMem, d_AData, memSize, cudaMemcpyDeviceToHost));
   return h_outMem;
}

//Sets the layer's size accordingly
int BaseLayer::setSize(){
   //Call previous connection to set sizes
   prevConn->setNextLayerSize(&ySize, &xSize, &fSize);
   if(DEBUG) std::cout << "Setting layer " << name << " size to ( " << ySize << ", " << xSize << ", " << fSize << ")\n";
   return SUCCESS;
}

int BaseLayer::initialize(){
   if(!paramsSet){
      std::cerr << "Error! Layer did not set parameters before trying to initialize\n";
      exit(UNDEFINED_PARAMS);
   }

   //Set this layer's size based on previous connection
   setSize();

   gpuDataSize = bSize * ySize * xSize * fSize * sizeof(float);

   //Create descriptor objects for both input and output buffers
   CudnnError(cudnnCreateTensorDescriptor(&cudnnADescriptor));
   CudnnError(cudnnCreateTensorDescriptor(&cudnnUDescriptor));

   if(DEBUG) std::cout << "Layer " << name << " setting descriptors " << bSize << ", " << fSize << ", " << ySize << ", " << xSize << "\n";
   CudnnError(cudnnSetTensor4dDescriptor(cudnnADescriptor,
      CUDNN_TENSOR_NCHW, //Ordering
      CUDNN_DATA_FLOAT, //Type
      bSize, //Number of images
      fSize, //Number of feature maps per image
      ySize, //Height of each feature map
      xSize) //Width of each feature map
   ); 

   CudnnError(cudnnSetTensor4dDescriptor(cudnnUDescriptor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
      bSize, //Number of images
      fSize, //Number of feature maps per image
      ySize, //Height of each feature map
      xSize) //Width of each feature map
   ); 

   return SUCCESS;
}

int BaseLayer::allocate(){
   CudaError( cudaMalloc(&d_AData, gpuDataSize));
   CudaError( cudaMalloc(&d_GData, gpuDataSize));
   return SUCCESS;
}

