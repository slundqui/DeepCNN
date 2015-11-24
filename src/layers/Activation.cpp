#include "Activation.hpp"
#include "../connections/BaseConnection.hpp"
#include "../utils.hpp"
#include "../Column.hpp"
#include "../cuda_utils.hpp"

/**
 * Activation.cpp
 * Author: Sheng Lundquist
 **/

Activation::Activation()
{
   isLinear = false;
}

Activation::~Activation(){
}

int Activation::setParams(Column* c, std::string layerName, std::string inMode){
   BaseLayer::setParams(c, layerName);

   if(inMode == "sigmoid"){
      activationMode = CUDNN_ACTIVATION_SIGMOID;
      isLinear = false;
   }
   else if(inMode == "relu"){
      activationMode = CUDNN_ACTIVATION_RELU;
      isLinear = false;
   }
   else if(inMode == "tanh"){
      activationMode = CUDNN_ACTIVATION_TANH;
      isLinear = false;
   }
   else if(inMode == "linear"){
      isLinear = true;
   }
   else{
      std::cerr << "Activation mode " << inMode << " not recognized, must be \"sigmoid\", \"relu\", \"tanh\", or \"linear\".\n";
      exit(BAD_PARAM);
   }

   return SUCCESS;
}

int Activation::initialize(){
   BaseLayer::initialize();

   return SUCCESS;
}

int Activation::allocate(){
   BaseLayer::allocate();
   return SUCCESS;
}

int Activation::applyActivation(){
   if(isLinear){
      CudaError(cudaDeviceSynchronize());
      CudaError(cudaMemcpy(d_AData, d_UData, gpuDataSize, cudaMemcpyDeviceToDevice));
   }
   else{
      //std::cout << "Applying activation to layer " << name << "\n";
      float alpha = 1;
      float beta = 0;
      cudnnHandle_t handle = col->getCudnnHandle();
      CudaError(cudaDeviceSynchronize());
      CudaError(cudaMemset(d_AData, 0, gpuDataSize));
      CudaError(cudaDeviceSynchronize());

      CudnnError(cudnnActivationForward(
         handle,
         activationMode,
         &alpha,
         layerDescriptor,
         d_UData,
         &beta,
         layerDescriptor,
         d_AData));
   }
   return SUCCESS;
}

int Activation::applyGradient(){
   if(isLinear){
      CudaError(cudaDeviceSynchronize());
      CudaError(cudaMemcpy(d_GUData, d_GAData, gpuDataSize, cudaMemcpyDeviceToDevice));
   }
   else{
      //if(DEBUG)std::cout << "Applying gradient to layer " << name << "\n";
      //std::cout << "Applying gradient to layer " << name << "\n";
      float alpha = 1;
      float beta = 0;
      
      //CudaError(cudaMemset(d_GUData, 0, gpuDataSize));

      cudnnHandle_t handle = col->getCudnnHandle();
      CudaError(cudaDeviceSynchronize());
      CudaError(cudaMemset(d_GUData, 0, gpuDataSize));
      CudaError(cudaDeviceSynchronize());

      CudnnError(cudnnActivationBackward(
         handle,
         activationMode,
         &alpha,
         layerDescriptor, //Layer src data, postactivation buffer
         d_AData,
         layerDescriptor, //Layer srcDiffData, gradients
         d_GAData,
         layerDescriptor, //destData, preactivation buffer
         d_UData,
         &beta,
         layerDescriptor, //Layer destDiffData, gradients
         d_GUData));
   }

   return SUCCESS;
}


