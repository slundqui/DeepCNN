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
   loadFilename = "";
   initVal = 0;
}

Convolution::~Convolution(){
}

int Convolution::setParams(Column* c, std::string connName, int in_nyp, int in_nxp, int in_nfp, int in_ystride, int in_xstride, int in_weightInitType, float in_initVal, std::string in_loadFilename){
   weightInitType = in_weightInitType;
   assert(weightInitType == 0 || weightInitType == 1);
   initVal = in_initVal;
   loadFilename = in_loadFilename;
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

   //Set up convolution descriptor
   if(DEBUG) std::cout << "Connection " << name << " setting cudnn conv descrptor with " << ystride << ", " << xstride << "\n";
   CudnnError(cudnnCreateConvolutionDescriptor(&convDescriptor));
   CudnnError(cudnnSetConvolution2dDescriptor(convDescriptor,
      0,
      0,  //zero-padding height and width
      ystride, //Vertical filter stride
      xstride, //Horizontal filter stride
      1, 1, //upscale the input in x/y direction
      CUDNN_CONVOLUTION) //Convolution as opposed to cross correlation
   );

   //Set size for this layer
   gpuDataSize = prevLayer->getFSize() * nyp * nxp * nfp * sizeof(float);

   return SUCCESS;
}

int Convolution::allocate(){
   //Calculate and set up best forward conv algorithm to use
   cudnnHandle_t handle = col->getCudnnHandle();

   CudnnError(cudnnGetConvolutionForwardAlgorithm(
      handle,
      prevLayer->getOutputDescriptor(),
      filterDescriptor,
      convDescriptor,
      nextLayer->getInputDescriptor(),
      //TODO: use this flag, but we need to calculate how much free space is left on the GPU and pass it in as next argument
      //CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT,
      CUDNN_CONVOLUTION_FWD_NO_WORKSPACE,
      0,
      &convAlgo
   ));
   CudaError( cudaMalloc(&d_WData, gpuDataSize));

   //Initialize data
   assert(initializeWeights() == SUCCESS);

   return SUCCESS;
}


float* Convolution::getHostW(){
   float * h_outMem = (float*) malloc(gpuDataSize);
   CudaError(cudaMemcpy(h_outMem, d_WData, gpuDataSize, cudaMemcpyDeviceToHost));
   CudaError(cudaDeviceSynchronize());
   return h_outMem;
}

int Convolution::initializeWeights(){
   int inNf = prevLayer->getFSize();

   if(weightInitType == 0){ //uniform weights
      std::cout << "Memcpying " << initVal << " for " << gpuDataSize << " bytes to weights\n";
      int count = nfp * inNf * nyp * nxp;
      setArray(d_WData, count, initVal);
   }
   else if(weightInitType == 1){
      int nDims;
      size_t * dims;
      readDataToDevice(loadFilename, d_WData, &nDims, &dims);
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


int Convolution::setNextLayerSize(int* ySize, int* xSize, int* fSize){
   //int bSize = col->getBSize();
   int tempBSize;

   cudnnTensorDescriptor_t inputDesc = prevLayer->getOutputDescriptor();

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

int Convolution::deliver(){
   cudnnHandle_t handle = col->getCudnnHandle();
   cudnnTensorDescriptor_t inputDesc = prevLayer->getOutputDescriptor();
   float* inputPtr = prevLayer->getDeviceA();
   cudnnTensorDescriptor_t outputDesc = nextLayer->getOutputDescriptor();
   float* outputPtr = nextLayer->getDeviceA();

   int alpha = 1; //input scaling
   int beta = 0; //output scaling, 0 means do not scale

   CudnnError(cudnnConvolutionForward(
      handle, //cudnn handle
      &alpha, //Input scaling factor
      inputDesc, //Input descriptor
      inputPtr, //Input pointer
      filterDescriptor, //Filter descriptor
      d_WData, //Filter pointer
      convDescriptor, //Convolution descriptor
      convAlgo, //Convolution algorithm
      NULL, //Workspace memory TODO
      0, //Workspace size
      &beta,
      outputDesc, //Output descriptor
      outputPtr //Output pointer
   ));
   return SUCCESS;
}


