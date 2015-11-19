/**
 * FullyConnected.cpp
 * Author: Sheng Lundquist
 **/

#include "FullyConnected.hpp"
#include "../layers/BaseLayer.hpp"
#include "../utils.hpp"
#include "../cuda_utils.hpp"
#include "../Column.hpp"

FullyConnected::FullyConnected(){
}

FullyConnected::~FullyConnected(){
}

int FullyConnected::setParams(Column* c, std::string connName, int in_nfp, int in_weightInitType, float in_weightInitVal, std::string in_weightLoadFilename, int in_biasInitType, float in_biasInitVal, std::string in_biasLoadFilename, int in_plasticity, float in_dwRate, float in_dbRate, float in_decay){

   return Convolution::setParams(
         c,
         connName,
         1, //This will be changed during setSize
         1,
         in_nfp,
         1,
         1,
         in_weightInitType, //0 means uniform with init_val, 1 means from file with in_loadFilename
         in_weightInitVal,
         in_weightLoadFilename,
         in_biasInitType, //0 means uniform with init_val, 1 means from file with in_loadFilename
         in_biasInitVal,
         in_biasLoadFilename,
         in_plasticity,
         in_dwRate,
         in_dbRate,
         in_decay
   );
}

int FullyConnected::setNextLayerSize(int* ySize, int* xSize, int* fSize){
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

   assert(*ySize == 1);
   assert(*xSize == 1);
   assert(*fSize == nfp);
   assert(tempBSize == prevLayer->getBSize());

   return SUCCESS;
}

int FullyConnected::initialize(){
   //Set nxp and nyp based on previous layer
   nxp = prevLayer->getXSize();
   nyp = prevLayer->getYSize();

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

   //Set up convolution descriptor
   if(DEBUG) std::cout << "Connection " << name << " setting cudnn conv descrptor with " << ystride << ", " << xstride << "\n";
   CudnnError(cudnnCreateConvolutionDescriptor(&convDescriptor));
   CudnnError(cudnnSetConvolution2dDescriptor(convDescriptor,
      0, //Padding height, makes layer size independent of patch size
      0,  //Padding width
      1, //Vertical filter stride
      1, //Horizontal filter stride
      1, 1, //upscale the input in x/y direction
      CUDNN_CONVOLUTION) //Convolution as opposed to cross correlation
   );

   //Set size for this layer
   gpuDataSize = prevLayer->getFSize() * nyp * nxp * nfp * sizeof(float);
   biasDataSize = nfp * sizeof(float);

   return SUCCESS;
}

