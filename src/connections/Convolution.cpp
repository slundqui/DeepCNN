/**
 * Convolution.cpp
 * Author: Sheng Lundquist
 **/

#include "Convolution.hpp"
#include "../layers/BaseLayer.hpp"
#include "../utils.hpp"
#include "../cuda_utils.hpp"
#include "../Column.hpp"

Convolution::Convolution(){
   d_WData = NULL;
   weightLoadFilename = "";
   weightInitVal = 0;
   biasLoadFilename = "";
   biasInitVal = 0;
   workspaceSize = 0;
   d_workspaceMem = NULL;
   biasDataSize = 0;
}

Convolution::~Convolution(){
   CudaError(cudaFree(d_WData));
   if(d_workspaceMem){
      CudaError(cudaFree(d_workspaceMem));
   }
}

int Convolution::setParams(Column* c, std::string connName, int in_nyp, int in_nxp, int in_nfp, int in_ystride, int in_xstride, int in_weightInitType, float in_weightInitVal, std::string in_weightLoadFilename, int in_biasInitType, float in_biasInitVal, std::string in_biasLoadFilename){
   weightInitType = in_weightInitType;
   assert(weightInitType == 0 || weightInitType == 1);
   weightInitVal = in_weightInitVal;
   weightLoadFilename = in_weightLoadFilename;
   biasInitType = in_biasInitType;
   assert(biasInitType == 0 || biasInitType == 1);
   biasInitVal = in_biasInitVal;
   biasLoadFilename = in_biasLoadFilename;
   return BaseConnection::setParams(c, connName, in_nyp, in_nxp, in_nfp, in_ystride, in_xstride);
}

int Convolution::initialize(){
   BaseConnection::initialize();
   if(!paramsSet){
      std::cerr << "Error! Connection did not set parameters before trying to initialize\n";
      exit(UNDEFINED_PARAMS);
   }
   int inNf = prevLayer->getFSize();

   //Set up filter descriptor
   CudnnError(cudnnCreateFilterDescriptor(&filterDescriptor));
   if(DEBUG) std::cout << "Connection " << name << " setting cudnn filter descrptor with " << nfp << ", " << inNf << ", " << nyp << ", " << nxp << "\n";
   CudnnError(cudnnSetFilter4dDescriptor(filterDescriptor, CUDNN_DATA_FLOAT,
      nfp, //Number of output feature maps.
      inNf, //Number of input feature maps
      nyp, //Height of each filter
      nxp) //Width of each filter
   );

   //Set up bias descriptor
   //One bias per output featuremap
   CudnnError(cudnnCreateTensorDescriptor(&biasDescriptor));
   CudnnError(cudnnSetTensor4dDescriptor(biasDescriptor,
      CUDNN_TENSOR_NCHW, //Ordering
      CUDNN_DATA_FLOAT, //Type
      1, //Number of images
      nfp, //Number of feature maps per image
      1, //Height of each feature map
      1//Width of each feature map
   ));

   //Needed to make layer size defined by stride only
   int pady = ystride % 2 == 0 ? (nyp-1)/2 : (nyp/2);
   int padx = xstride % 2 == 0 ? (nxp-1)/2 : (nxp/2);

   //Set up convolution descriptor
   if(DEBUG) std::cout << "Connection " << name << " setting cudnn conv descrptor with " << ystride << ", " << xstride << "\n";
   CudnnError(cudnnCreateConvolutionDescriptor(&convDescriptor));
   CudnnError(cudnnSetConvolution2dDescriptor(convDescriptor,
      pady, //Padding height, makes layer size independent of patch size
      padx,  //Padding width
      ystride, //Vertical filter stride
      xstride, //Horizontal filter stride
      1, 1, //upscale the input in x/y direction
      CUDNN_CONVOLUTION) //Convolution as opposed to cross correlation
   );

   //Set size for this layer
   gpuDataSize = prevLayer->getFSize() * nyp * nxp * nfp * sizeof(float);
   biasDataSize = nfp * sizeof(float);

   return SUCCESS;
}

int Convolution::allocate(){
   //Calculate and set up best forward conv algorithm to use
   cudnnHandle_t handle = col->getCudnnHandle();

   size_t forward_workspaceSize;
   CudnnError(cudnnGetConvolutionForwardAlgorithm(
      handle,
      prevLayer->getLayerDescriptor(),
      filterDescriptor,
      convDescriptor,
      nextLayer->getLayerDescriptor(),
      //TODO make sure we have enough workspace size
      //CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT,
      //CUDNN_CONVOLUTION_FWD_NO_WORKSPACE,
      CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
      0,
      &convAlgo
   ));

   CudnnError(cudnnGetConvolutionForwardWorkspaceSize(
      handle,
      prevLayer->getLayerDescriptor(),
      filterDescriptor,
      convDescriptor,
      nextLayer->getLayerDescriptor(),
      convAlgo,
      &forward_workspaceSize));

   ////Calculate and set up best backward conv altorithm to use for finding gradient vals
   //CudnnError(cudnnGetConvolutionBackwardDataAlgorithm(
   //   handle,
   //   filterDescriptor,
   //   nextLayer->getLayerDescriptor(),
   //   prevLayer->getLayerDescriptor(),
   //   convDescriptor,




   //));

   ////Calculate and set up best backward conv algorithm to use for updating weights
   //CudnnError(cudnnGetConvolutionBackwardFilterAlgorithm(
   //   handle,
   //   nextLayer->getGradientDescriptor(),
   //   prevLayer->getGradientDescriptor(),
   //   convDescriptor,




   //));

   //TODO find maximum of workspace sizes
   workspaceSize = forward_workspaceSize;

   //Allocate weights and bias
   CudaError(cudaMalloc(&d_WData, gpuDataSize));
   CudaError(cudaMalloc(&d_Bias, biasDataSize));

   CudaError(cudaMalloc(&d_workspaceMem, workspaceSize));

   //Initialize data
   assert(initializeWeights() == SUCCESS);
   assert(initializeBias() == SUCCESS);

   return SUCCESS;
}


float* Convolution::getHostW(){
   int inNf = prevLayer->getFSize();
   size_t memSize = nyp * inNf * nxp * nfp * sizeof(float);
   assert(memSize == gpuDataSize);
   CudaError(cudaDeviceSynchronize());
   float * h_outMem = (float*) malloc(gpuDataSize);
   CudaError(cudaMemcpy(h_outMem, d_WData, gpuDataSize, cudaMemcpyDeviceToHost));
   CudaError(cudaDeviceSynchronize());
   return h_outMem;
}

float* Convolution::getHostBias(){
   size_t memSize = nfp * sizeof(float);
   assert(memSize == biasDataSize);
   CudaError(cudaDeviceSynchronize());
   float * h_outMem = (float*) malloc(biasDataSize);
   CudaError(cudaMemcpy(h_outMem, d_Bias, biasDataSize, cudaMemcpyDeviceToHost));
   CudaError(cudaDeviceSynchronize());
   return h_outMem;
}

int Convolution::initializeWeights(){
   int inNf = prevLayer->getFSize();

   if(weightInitType == 0){ //uniform weights
      int count = nfp * inNf * nyp * nxp;
      setArray(d_WData, count, weightInitVal);
   }
   else if(weightInitType == 1){
      int nDims;
      size_t * dims;
      readDataToDevice(weightLoadFilename, d_WData, &nDims, &dims);
      assert(nDims == 4);

      assert(dims[0] == (size_t)nfp);
      assert(dims[1] == (size_t)inNf);
      assert(dims[2] == (size_t)nyp);
      assert(dims[3] == (size_t)nxp);
   }
   else{
      std::cerr << "Weight init type of " << weightInitType << " not recognized\n";
      exit(BAD_PARAM);
   }
   return SUCCESS;
}

int Convolution::initializeBias(){
   if(biasInitType == 0){ //uniform weights
      int count = nfp;
      setArray(d_Bias, count, biasInitVal);
   }
   else if(biasInitType == 1){
      int nDims;
      size_t * dims;
      readDataToDevice(biasLoadFilename, d_Bias, &nDims, &dims);
      assert(nDims == 1);
      assert(dims[0] == (size_t)nfp);
   }
   else{
      std::cerr << "Bias init type of " << biasInitType << " not recognized\n";
      exit(BAD_PARAM);
   }
   return SUCCESS;
}


int Convolution::setNextLayerSize(int* ySize, int* xSize, int* fSize){
   //int bSize = col->getBSize();
   int tempBSize;

   cudnnTensorDescriptor_t inputDesc = prevLayer->getLayerDescriptor();

   //Query output layout and check with PV layout
   CudnnError(cudnnGetConvolution2dForwardOutputDim(
      convDescriptor, //Conv descriptor
      inputDesc, //Input descriptor
      filterDescriptor,
      &tempBSize, //num images
      fSize, //num output features
      ySize, //output height
      xSize) //output width
   );
   assert(tempBSize == prevLayer->getBSize());

   return SUCCESS;
}

//TODO
int Convolution::updateWeights(int timestep){
   return SUCCESS;
}

int Convolution::forwardDeliver(){
   if(DEBUG) std::cout << "Convolution deliver called\n";

   cudnnHandle_t handle = col->getCudnnHandle();
   cudnnTensorDescriptor_t inputDesc = prevLayer->getLayerDescriptor();
   float* inputPtr = prevLayer->getDeviceA();
   cudnnTensorDescriptor_t outputDesc = nextLayer->getLayerDescriptor();
   float* outputPtr = nextLayer->getDeviceA();

   float alpha = 1; //input scaling
   float beta = 0; //output scaling, 0 means do not scale

   CudaError(cudaDeviceSynchronize());
   CudnnError(cudnnConvolutionForward(
      handle, //cudnn handle
      &alpha, //Input scaling factor
      inputDesc, //Input descriptor
      inputPtr, //Input pointer
      filterDescriptor, //Filter descriptor
      d_WData, //Filter pointer
      convDescriptor, //Convolution descriptor
      convAlgo, //Convolution algorithm
      d_workspaceMem, //Workspace memory TODO
      workspaceSize, //Workspace size
      &beta, //Output scaling factor
      outputDesc, //Output descriptor
      outputPtr //Output pointer
   ));

   //Add bias
   CudnnError(cudnnAddTensor(
      handle, //cudnnHandle
      CUDNN_ADD_SAME_C,
      &alpha,
      biasDescriptor, //bias descriptor
      d_Bias, //bias pointer
      &alpha,
      outputDesc, //Output descriptor
      outputPtr //Output pointer
   ));

   return SUCCESS;
}

int Convolution::backwardDeliver(){
   //cudnnHandle_t handle = col->getCudnnHandle();
   //cudnnTensorDescriptor_t inputDesc = nextLayer->getGradientDescriptor();
   //float* inputPtr = nextLayer->getDeviceG();
   //cudnnTensorDescriptor_t outputDesc = prevLayer->getGradientDescriptor();
   //float* outputPtr = prevLayer->getDeviceG();

   //float alpha = 1; //input scaling
   //float beta = 0; //output scaling, 0 means do not scale

   //CudaError(cudaDeviceSynchronize());
   //CudnnError(cudnnConvolutionBackwardFilter_v3(
   //   handle, //cudnn handle
   //   &alpha, //Input scaling factor
   //   inputDesc, //Input descriptor
   //   inputPtr, //Input pointer
   //   outputDesc, //Output descriptor
   //   outputPtr, //Output pointer
   //   convDescriptor, //Convolution descriptor
   //   

   //));
   return SUCCESS;
}


