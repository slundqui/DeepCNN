/**
 * Convolution.hpp
 *
 * Author: Sheng Lundquist
 **/
#ifndef CONVOLUTION_HPP_ 
#define CONVOLUTION_HPP_ 

#include "includes.hpp"
#include "BaseConnection.hpp"

class Convolution : public BaseConnection{
public:
   Convolution();
   virtual ~Convolution();
   virtual int setParams(
         Column* c,
         std::string connName,
         int in_nyp,
         int in_nxp,
         int in_nfp,
         int in_ystride,
         int in_xstride,
         int in_weightInitType = 0, //0 means uniform with init_val, 1 means from file with in_loadFilename
         float in_weightInitVal = 0,
         std::string in_weightLoadFilename = "",
         int in_biasInitType = 0, //0 means uniform with init_val, 1 means from file with in_loadFilename
         float in_biasInitVal = 0,
         std::string in_biasLoadFilename = "",
         int in_plasticity = 0, //learning or not
         float in_dwRate = .001, //weight learning rate
         float in_dbRate = .001, //bias learning rate
         float in_dwMom = 0,
         float in_dbMom = 0,
         float in_decay = 0 //Decay applied to the weight
         );

   virtual int initialize();
   virtual int setCudnnDescriptors();
   virtual int allocate();
   virtual int updateWeights(int timestep);
   virtual int forwardDeliver();
   virtual int backwardDeliver();
   virtual int setNextLayerSize(int* ySize, int* xSize, int* fSize);
   float* getHostW();
   float* getHostB();
   float* getHostWGradient();
   float* getHostBGradient();
   //Sets the flag for plasticity
   void setGradientCheck(){plasticity = 0; needGrad = 1;}
   //Sets a weight of a specific index to a specific value
   int setWeight(int idx, float val);
   int setBias(int idx, float val);
   int getNumWeights();
   int getNumBias(){return nfp;}

   void printW();
   void printB();
   void printGW();
   void printGB();

   void setDwRate(float inval){dwRate = inval;}
   void setDbRate(float inval){dwRate = inval;}
   float getDwRate(){return dwRate;}
   float getDbRate(){return dbRate;}

protected:
   cudnnFilterDescriptor_t filterDescriptor;
   cudnnTensorDescriptor_t biasDescriptor;

   cudnnConvolutionDescriptor_t convDescriptor;
   cudnnConvolutionFwdAlgo_t forwardConvAlgo;
   cudnnConvolutionBwdFilterAlgo_t backwardFilterAlgo;
   cudnnConvolutionBwdDataAlgo_t backwardDataAlgo;

   size_t workspaceSize;
   void* d_workspaceMem;
   virtual int initializeWeights();
   virtual int initializeBias();

   int weightInitType;
   int biasInitType;
   float weightInitVal;
   float biasInitVal;
   std::string weightLoadFilename;
   std::string biasLoadFilename;

   size_t biasDataSize;

   float* d_WData;
   float* d_Bias;
   float* d_dWData;
   float* d_dBias;

   //Gradients of weights and biases
   float* d_GWData;
   float* d_GBias;

   //Learning parameters
   int plasticity;
   int needGrad;
   float dwRate;
   float dbRate;
   float dwMom;
   float dbMom;
   float decay;

   int weightBlockSize;
   int weightGridSize;
   int biasBlockSize;
   int biasGridSize;



};
#endif 
