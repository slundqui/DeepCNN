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
         float in_initVal = 0,
         std::string in_loadFilename = ""
         );
   virtual int initialize();
   virtual int allocate();
   virtual int updateWeights(int timestep);
   virtual int deliver();
   virtual int setNextLayerSize(int* ySize, int* xSize, int* fSize);
   float* getHostW();

protected:
   cudnnFilterDescriptor_t filterDescriptor;
   cudnnConvolutionDescriptor_t convDescriptor;
   cudnnConvolutionFwdAlgo_t convAlgo;
   size_t workspaceSize;
   void* d_workspaceMem;
   virtual int initializeWeights();

   int weightInitType;
   float initVal;
   std::string loadFilename;

   float* d_WData;

};
#endif 
