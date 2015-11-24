/**
 * MaxPool.cpp
 * Author: Sheng Lundquist
 **/


#include "MaxPool.hpp"
#include "../layers/BaseLayer.hpp"
#include "../utils.hpp"
#include "../cuda_utils.hpp"
#include "../Column.hpp"
#include "../kernels.hpp"

MaxPool::MaxPool(){
   needGrad = 1;
}

MaxPool::~MaxPool(){
}

int MaxPool::setParams(
      Column* c, std::string connName,
      int in_nyp,
      int in_nxp,
      int in_ystride,
      int in_xstride
      ){

   return BaseConnection::setParams(c, connName, in_nyp, in_nxp, 1, in_ystride, in_xstride);
}

int MaxPool::setCudnnDescriptors(){
   int inNf = prevLayer->getFSize();
   //Needed to make layer size defined by stride only
   int pady = ystride % 2 == 0 ? (nyp-1)/2 : (nyp/2);
   int padx = xstride % 2 == 0 ? (nxp-1)/2 : (nxp/2);

   //Set up pool descriptor
   CudnnError(cudnnCreatePoolingDescriptor(&poolingDescriptor));
   if(DEBUG) std::cout << "Connection " << name << " setting cudnn pooling descrptor with " << nfp << ", " << nyp << ", " << nxp << "\n";
   CudnnError(cudnnSetPooling2dDescriptor(poolingDescriptor, CUDNN_POOLING_MAX,
      nyp, //Height of window
      nxp,
      pady, //Padding
      padx,
      ystride, //stride of pooling
      xstride
      ) //Width of window
   );

   return SUCCESS;
}


int MaxPool::initialize(){
   //Set nfp to be previous layers's nf
   nfp = prevLayer->getFSize();

   BaseConnection::initialize();
   setCudnnDescriptors();

   return SUCCESS;
}

int MaxPool::allocate(){
   return SUCCESS;
}

int MaxPool::setNextLayerSize(int* ySize, int* xSize, int* fSize){
   //int bSize = col->getBSize();
   int tempBSize;

   cudnnTensorDescriptor_t inputDesc = prevLayer->getLayerDescriptor();

   //Query output layout and check with PV layout
   CudnnError(cudnnGetPooling2dForwardOutputDim(
      poolingDescriptor, //Conv descriptor
      inputDesc, //Input descriptor
      &tempBSize, //num images
      fSize, //num output features
      ySize, //output height
      xSize) //output width
   );
   assert(tempBSize == prevLayer->getBSize());

   return SUCCESS;
}

int MaxPool::forwardDeliver(){
   if(DEBUG) std::cout << "MaxPool deliver called\n";

   cudnnHandle_t handle = col->getCudnnHandle();
   cudnnTensorDescriptor_t inputDesc = prevLayer->getLayerDescriptor();
   float* inputPtr = prevLayer->getDeviceA();

   cudnnTensorDescriptor_t outputDesc = nextLayer->getLayerDescriptor();
   float* outputPtr = nextLayer->getDeviceU();


   int outputSize = nextLayer->getBSize() * nextLayer->getFSize() * nextLayer->getYSize() * nextLayer->getXSize() * sizeof(float);
   CudaError(cudaDeviceSynchronize());
   CudaError(cudaMemset(outputPtr, 0, outputSize));
   CudaError(cudaDeviceSynchronize());

   float alpha = 1; //input scaling
   float beta = 0; //output scaling, 0 means do not scale

   CudnnError(cudnnPoolingForward(
      handle, //cudnn handle
      poolingDescriptor,
      &alpha, //Input scaling factor
      inputDesc, //Input descriptor
      inputPtr, //Input pointer
      &beta, //Output scaling factor
      outputDesc, //Output descriptor
      outputPtr //Output pointer
   ));
   return SUCCESS;
}

int MaxPool::backwardDeliver(){
   if(!needGrad) return SUCCESS;
   

   if(DEBUG) std::cout << "MaxPool gradient called\n";
   
   cudnnHandle_t handle = col->getCudnnHandle();

   cudnnTensorDescriptor_t prevLayerDesc = prevLayer->getLayerDescriptor();
   cudnnTensorDescriptor_t nextLayerDesc = nextLayer->getLayerDescriptor();

   float* prevLayerA = prevLayer->getDeviceA();
   float* nextLayerU = nextLayer->getDeviceU();
   float* nextLayerGU = nextLayer->getDeviceGU();
   float* prevLayerGA = prevLayer->getDeviceGA();

   CudaError(cudaDeviceSynchronize());

   //Clear all gradient buffers
   int prevLayerSize = prevLayer->getBSize() * prevLayer->getFSize() * prevLayer->getYSize() * prevLayer->getXSize() * sizeof(float);
   CudaError(cudaMemset(prevLayerGA, 0, prevLayerSize));

   CudaError(cudaDeviceSynchronize());

   float alpha = 1; //input scaling
   float beta = 0; //output scaling, 0 means do not scale

   //Bias gradient calculation
   CudnnError(cudnnPoolingBackward(
      handle,
      poolingDescriptor,
      &alpha,
      nextLayerDesc, //Next layer's activity
      nextLayerU,
      nextLayerDesc, //Next layer's gradient
      nextLayerGU,
      prevLayerDesc, //Prev layer's activity
      prevLayerA,
      &beta,
      prevLayerDesc, //Prev layer's gradient
      prevLayerGA
   ));

   return SUCCESS;
}
