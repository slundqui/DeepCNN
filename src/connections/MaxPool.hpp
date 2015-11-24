/**
 * MaxPool.hpp
 *
 * Author: Sheng Lundquist
 **/
#ifndef MAXPOOL_HPP_ 
#define MAXPOOL_HPP_ 

#include "includes.hpp"
#include "BaseConnection.hpp"

class MaxPool : public BaseConnection{
public:
   MaxPool();
   virtual ~MaxPool();
   virtual int setParams(
         Column* c,
         std::string connName,
         int in_nyp,
         int in_nxp,
         int in_ystride,
         int in_xstride
         );

   virtual int initialize();
   virtual int setCudnnDescriptors();
   virtual int allocate();
   virtual int forwardDeliver();
   virtual int backwardDeliver();
   virtual int setNextLayerSize(int* ySize, int* xSize, int* fSize);
   void setGradientCheck(){needGrad = 1;}
   
protected:
   cudnnPoolingDescriptor_t poolingDescriptor;
   int needGrad;

};
#endif 
