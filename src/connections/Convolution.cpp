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
   d_Bias = NULL;
   d_GWData = NULL;
   d_GBias = NULL;
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
   CudaError(cudaFree(d_Bias));
   CudaError(cudaFree(d_GWData));
   CudaError(cudaFree(d_GBias));
   if(d_workspaceMem){
      CudaError(cudaFree(d_workspaceMem));
   }
}

int Convolution::setParams(Column* c, std::string connName, int in_nyp, int in_nxp, int in_nfp, int in_ystride, int in_xstride, int in_weightInitType, float in_weightInitVal, std::string in_weightLoadFilename, int in_biasInitType, float in_biasInitVal, std::string in_biasLoadFilename, int in_plasticity, float in_dwRate, float in_dbRate, float in_decay){
   weightInitType = in_weightInitType;
   assert(weightInitType == 0 || weightInitType == 1);
   weightInitVal = in_weightInitVal;
   weightLoadFilename = in_weightLoadFilename;
   biasInitType = in_biasInitType;
   assert(biasInitType == 0 || biasInitType == 1);
   biasInitVal = in_biasInitVal;
   biasLoadFilename = in_biasLoadFilename;

   plasticity = in_plasticity;
   needGrad = plasticity;
   dwRate = in_dwRate;
   dbRate = in_dbRate;
   decay = in_decay;

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
   cudnnHandle_t handle = col->getCudnnHandle();

   //Calculate and set up best conv algorithms to use
   //Forward pass
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
      &forwardConvAlgo
   ));

   CudnnError(cudnnGetConvolutionForwardWorkspaceSize(
      handle,
      prevLayer->getLayerDescriptor(),
      filterDescriptor,
      convDescriptor,
      nextLayer->getLayerDescriptor(),
      forwardConvAlgo,
      &forward_workspaceSize));


   //Backward filter gradient
   size_t backward_filter_workspaceSize;
   CudnnError(cudnnGetConvolutionBackwardFilterAlgorithm(
      handle, 
      prevLayer->getLayerDescriptor(), //Source descriptor (prev layer activation)
      nextLayer->getLayerDescriptor(), //Diff descriptor (Next layer gradient)
      convDescriptor, //Convolution descriptor
      filterDescriptor, //weight gradient descriptor
      //TODO make sure we have enough workspace size
      CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST, 
      0,
      &backwardFilterAlgo
   ));

   CudnnError(cudnnGetConvolutionBackwardFilterWorkspaceSize(
      handle,
      prevLayer->getLayerDescriptor(), //Source descriptor (prev layer activaiton)
      nextLayer->getLayerDescriptor(), //Diff descriptor (Next layer gradient)
      convDescriptor, //Convolution descriptor
      filterDescriptor, //weight gradient descriptor
      backwardFilterAlgo, //Algorithm
      &backward_filter_workspaceSize
   ));

   //Backward data gradient
   size_t backward_data_workspaceSize;
   CudnnError(cudnnGetConvolutionBackwardDataAlgorithm(
      handle,
      filterDescriptor, //Filter descriptor
      nextLayer->getLayerDescriptor(), //Diff descriptor (Next layer gradient)
      convDescriptor, //Convolution descriptor
      prevLayer->getLayerDescriptor(), //Output gradient descriptor (prev layer gradient)
      CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST,
      0, 
      &backwardDataAlgo
   ));

   CudnnError(cudnnGetConvolutionBackwardDataWorkspaceSize(
      handle,
      filterDescriptor, //Filter descriptor
      nextLayer->getLayerDescriptor(), //Diff descriptor (Next layer gradient)
      convDescriptor, //Convolution descriptor
      prevLayer->getLayerDescriptor(), //Output gradient descriptor (prev layer gradient)
      backwardDataAlgo, //Algorithm
      &backward_data_workspaceSize
   ));

   //Find maximum of workspace sizes
   workspaceSize = forward_workspaceSize > backward_filter_workspaceSize ? forward_workspaceSize: backward_filter_workspaceSize;
   workspaceSize = workspaceSize > backward_data_workspaceSize ? workspaceSize : backward_data_workspaceSize;

   //Allocate weights and bias
   CudaError(cudaMalloc(&d_WData, gpuDataSize));
   CudaError(cudaMalloc(&d_GWData, gpuDataSize));
   CudaError(cudaMalloc(&d_Bias, biasDataSize));
   CudaError(cudaMalloc(&d_GBias, biasDataSize));

   CudaError(cudaMalloc(&d_workspaceMem, workspaceSize));

   //Initialize data
   assert(initializeWeights() == SUCCESS);
   assert(initializeBias() == SUCCESS);

   return SUCCESS;
}

int Convolution::setWeight(int idx, float val){
   int inNf = prevLayer->getFSize();
   assert(idx >= 0 && idx < nfp * inNf * nyp * nxp); 
   float* d_offsetWData = &(d_WData[idx]);
   CudaError(cudaMemcpy(d_offsetWData, &val, sizeof(float), cudaMemcpyHostToDevice));
   return SUCCESS;
}

int Convolution::setBias(int idx, float val){
   int inNf = prevLayer->getFSize();
   assert(idx >= 0 && idx < nfp); 
   float* d_offsetBias = &(d_Bias[idx]);
   CudaError(cudaMemcpy(d_offsetBias, &val, sizeof(float), cudaMemcpyHostToDevice));
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

float* Convolution::getHostB(){
   size_t memSize = nfp * sizeof(float);
   assert(memSize == biasDataSize);
   CudaError(cudaDeviceSynchronize());
   float * h_outMem = (float*) malloc(biasDataSize);
   CudaError(cudaMemcpy(h_outMem, d_Bias, biasDataSize, cudaMemcpyDeviceToHost));
   CudaError(cudaDeviceSynchronize());
   return h_outMem;
}

float* Convolution::getHostWGradient(){
   assert(d_GWData);
   int inNf = prevLayer->getFSize();
   size_t memSize = nyp * inNf * nxp * nfp * sizeof(float);
   assert(memSize == gpuDataSize);
   CudaError(cudaDeviceSynchronize());
   float * h_outMem = (float*) malloc(gpuDataSize);
   CudaError(cudaMemcpy(h_outMem, d_GWData, gpuDataSize, cudaMemcpyDeviceToHost));
   CudaError(cudaDeviceSynchronize());
   return h_outMem;
}

float* Convolution::getHostBGradient(){
   size_t memSize = nfp * sizeof(float);
   assert(memSize == biasDataSize);
   CudaError(cudaDeviceSynchronize());
   float * h_outMem = (float*) malloc(biasDataSize);
   CudaError(cudaMemcpy(h_outMem, d_GBias, biasDataSize, cudaMemcpyDeviceToHost));
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

      assert(dims[0] == (size_t)nxp); //Fastest
      assert(dims[1] == (size_t)nyp);
      assert(dims[2] == (size_t)inNf);
      assert(dims[3] == (size_t)nfp); //Slowest
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
   float* outputPtr = nextLayer->getDeviceU();

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
      forwardConvAlgo, //Convolution algorithm
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
   if(!needGrad) return SUCCESS;
   //std::cout << "Layer " << name << " gradient called\n";
   
   //Clear all gradient buffers
   CudaError(cudaMemset(d_GWData, 0, gpuDataSize));
   CudaError(cudaMemset(d_GBias, 0, biasDataSize));

   if(DEBUG) std::cout << "Convolution gradient called\n";
   //std::cout << "Convolution gradient called\n";

   cudnnHandle_t handle = col->getCudnnHandle();
   cudnnTensorDescriptor_t srcDesc = prevLayer->getLayerDescriptor();
   float* srcPtr = prevLayer->getDeviceA();

   cudnnTensorDescriptor_t diffDesc = nextLayer->getLayerDescriptor();
   float* diffPtr = nextLayer->getDeviceG();
   //float* nextAPtr = nextLayer->getDeviceA();

   cudnnTensorDescriptor_t outputDesc = prevLayer->getLayerDescriptor();
   float* outputPtr = prevLayer->getDeviceG();

   float alpha = 1; //input scaling
   float beta = 0; //output scaling, 0 means do not scale

   CudaError(cudaDeviceSynchronize());
   //Bias gradient calculation
   CudnnError(cudnnConvolutionBackwardBias(
      handle,
      &alpha,
      diffDesc, //Prev layer's activations
      diffPtr,
      &beta,
      biasDescriptor, //Bias gradient descriptor
      d_GBias //Bias gradient pointer
   ));

   //TODO need sync barrier here because of workspace?
   CudaError(cudaDeviceSynchronize());
   
   //Weight gradient calculation
   CudnnError(cudnnConvolutionBackwardFilter_v3(
      handle,
      &alpha,
      srcDesc, //Prev layer's activations
      srcPtr,
      diffDesc, //Next layer's gradient
      diffPtr,
      convDescriptor, //Conv descriptor
      backwardFilterAlgo, //Algorithm
      d_workspaceMem, //Workspace pointer
      workspaceSize, //Workspace size
      &beta,
      filterDescriptor, //Weight gradient descriptor
      d_GWData //Weight gradient pointer
   ));

   CudaError(cudaDeviceSynchronize());

   //Data gradient calculation (backpass to previous layer)
   CudnnError(cudnnConvolutionBackwardData_v3(
      handle,
      &alpha,
      filterDescriptor, //Weights 
      d_WData,
      diffDesc, //Next layer's gradient
      diffPtr,
      convDescriptor, //Conv descriptor
      backwardDataAlgo, //Algorithm
      d_workspaceMem, //Workspace pointer
      workspaceSize, //Workspace size
      &beta,
      outputDesc, //Prev layer gradient desc
      outputPtr //Prev layer gradient pointer
   ));

   return SUCCESS;
}

int Convolution::getNumWeights(){
   return prevLayer->getFSize() * nyp * nxp * nfp;
}



