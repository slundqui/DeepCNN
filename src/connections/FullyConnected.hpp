/**
 * FullyConnected.hpp
 *
 * Author: Sheng Lundquist
 **/
#ifndef FULLYCONNECTED_HPP_ 
#define FULLYCONNECTED_HPP_ 

#include "includes.hpp"
#include "Convolution.hpp"

class FullyConnected: public Convolution{
public:
   FullyConnected();
   virtual ~FullyConnected();
   virtual int setParams(
         Column* c,
         std::string connName,
         int in_nfp,
         int in_weightInitType = 0, //0 means uniform with init_val, 1 means from file with in_loadFilename
         float in_weightInitVal = 0,
         std::string in_weightLoadFilename = "",
         int in_biasInitType = 0, //0 means uniform with init_val, 1 means from file with in_loadFilename
         float in_biasInitVal = 0,
         std::string in_biasLoadFilename = ""
         );
   virtual int initialize();
   virtual int setNextLayerSize(int* ySize, int* xSize, int* fSize);
};
#endif 
