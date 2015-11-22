#include "BaseLayer.hpp"
#include "../connections/BaseConnection.hpp"
#include "../utils.hpp"
#include "../Column.hpp"
#include "../cuda_utils.hpp"

/**
 * BaseLayer.cpp
 * Author: Sheng Lundquist
 **/

BaseLayer::BaseLayer()
{
   d_UData = NULL;
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
   CudaError( cudaFree(d_UData));
   CudaError( cudaFree(d_AData));
   CudaError( cudaFree(d_GData));
   
   //Currently bugging out, maybe need to set the tensor first
   //CudnnError( cudnnDestroyTensorDescriptor(cudnnADescriptor));
   //CudnnError( cudnnDestroyTensorDescriptor(cudnnGDescriptor));

   //free(h_AData);
}

int BaseLayer::setParams(Column* c, std::string layerName){
   BaseData::setParams(c, layerName);
   bSize = c->getBSize();

   paramsSet = true;
   return SUCCESS;
}

float* BaseLayer::getHostU(){
   size_t memSize = bSize * ySize * xSize * fSize * sizeof(float);
   assert(memSize == gpuDataSize);
   CudaError(cudaDeviceSynchronize());
   float * h_outMem = (float*) malloc(gpuDataSize);
   CudaError(cudaMemcpy(h_outMem, d_UData, gpuDataSize, cudaMemcpyDeviceToHost));
   CudaError(cudaDeviceSynchronize());
   return h_outMem;
}

float* BaseLayer::getHostA(){
   size_t memSize = bSize * ySize * xSize * fSize * sizeof(float);
   assert(memSize == gpuDataSize);
   CudaError(cudaDeviceSynchronize());
   float * h_outMem = (float*) malloc(gpuDataSize);
   CudaError(cudaMemcpy(h_outMem, d_AData, gpuDataSize, cudaMemcpyDeviceToHost));
   CudaError(cudaDeviceSynchronize());
   return h_outMem;
}

float* BaseLayer::getHostG(){
   size_t memSize = bSize * ySize * xSize * fSize * sizeof(float);
   assert(memSize == gpuDataSize);
   CudaError(cudaDeviceSynchronize());
   float * h_outMem = (float*) malloc(gpuDataSize);
   CudaError(cudaMemcpy(h_outMem, d_GData, gpuDataSize, cudaMemcpyDeviceToHost));
   CudaError(cudaDeviceSynchronize());
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

   //Create descriptor objects for layer buffers
   CudnnError(cudnnCreateTensorDescriptor(&layerDescriptor));
   //CudnnError(cudnnCreateTensorDescriptor(&cudnnGDescriptor));

   if(DEBUG) std::cout << "Layer " << name << " setting descriptors " << bSize << ", " << fSize << ", " << ySize << ", " << xSize << "\n";
   CudnnError(cudnnSetTensor4dDescriptor(layerDescriptor,
      CUDNN_TENSOR_NCHW, //Ordering
      CUDNN_DATA_FLOAT, //Type
      bSize, //Number of images
      fSize, //Number of feature maps per image
      ySize, //Height of each feature map
      xSize) //Width of each feature map
   ); 

   return SUCCESS;
}

int BaseLayer::allocate(){
   CudaError( cudaMalloc(&d_UData, gpuDataSize));
   CudaError( cudaMalloc(&d_AData, gpuDataSize));
   CudaError( cudaMalloc(&d_GData, gpuDataSize));

   //Initialize all layer data to 0
   //int count = bSize * ySize * xSize * fSize;

   CudaError(cudaMemset(d_UData, 0, gpuDataSize));
   CudaError(cudaMemset(d_AData, 0, gpuDataSize));
   CudaError(cudaMemset(d_GData, 0, gpuDataSize));
   
   //setArray(d_UData, count, 0);
   //setArray(d_AData, count, 0);
   //setArray(d_GData, count, 0);

   return SUCCESS;
}

int BaseLayer::applyActivation(){
   CudaError(cudaMemcpy(d_AData, d_UData, gpuDataSize, cudaMemcpyDeviceToDevice));
   CudaError(cudaDeviceSynchronize());
   return SUCCESS;
}

int BaseLayer::forwardUpdate(int timestep){
   prevConn->forwardDeliver();
   applyActivation();
   return SUCCESS;
}

int BaseLayer::applyGradient(){
   //TODO seperate G buffers.
   return SUCCESS;
}

int BaseLayer::backwardsUpdate(int timestep){
   //Update current g buffer
   if(nextConn){
      nextConn->backwardDeliver();
   }
   //Apply gradient activation
   applyGradient();

   return SUCCESS;
}

void BaseLayer::printU(){
   float* h_data = getHostU();
   printMat(h_data, bSize, fSize, ySize, xSize);
   free(h_data);
}

void BaseLayer::printA(){
   float* h_data = getHostA();
   printMat(h_data, bSize, fSize, ySize, xSize);
   free(h_data);
}

void BaseLayer::printG(){
   float* h_data = getHostG();
   printMat(h_data, bSize, fSize, ySize, xSize);
   free(h_data);
}

