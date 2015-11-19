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
}

Activation::~Activation(){
}

int Activation::setParams(Column* c, std::string layerName, std::string inMode){
   BaseLayer::setParams(c, layerName);

   if(inMode == "sigmoid"){
      activationMode = CUDNN_ACTIVATION_SIGMOID;
   }
   else if(inMode == "reul"){
      activationMode = CUDNN_ACTIVATION_RELU;
   }
   else if(inMode == "tanh"){
      activationMode = CUDNN_ACTIVATION_TANH;
   }
   else{
      std::cerr << "Activation mode " << inMode << " not recognized, must be \"sigmoid\", \"relu\", or \"tanh\".\n";
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
   float alpha = 1;
   float beta = 0;
   cudnnHandle_t handle = col->getCudnnHandle();
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
   return SUCCESS;
}

int Activation::backwardsUpdate(int timestep){
   BaseLayer::backwardsUpdate(timestep);
   float alpha = 1;
   float beta = 0;
   cudnnHandle_t handle = col->getCudnnHandle();
   CudaError(cudaDeviceSynchronize());
   CudnnError(cudnnActivationBackward(
      handle,
      activationMode,
      &alpha,
      layerDescriptor, //Layer src data, preactivation buffer
      d_UData,
      layerDescriptor, //Layer srcDiffData, gradients
      d_GData,
      layerDescriptor, //destData, postactivation buffer
      d_AData,
      &beta,
      layerDescriptor, //Layer destDiffData, gradients
      d_GData));

   return SUCCESS;

}

