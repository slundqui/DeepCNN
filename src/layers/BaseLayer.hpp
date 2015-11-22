/**
 * BaseLayer.hpp
 *
 * The base abstract layer
 * Each layer must define 3 methods: parameterSetting, initialization, and update
 * 
 * Each layer must also define a buffer.
 * AData: The unit activation buffer to be passed onto the next layer via convolution
 *
 * Author: Sheng Lundquist
 **/
#ifndef BASELAYER_HPP_ 
#define BASELAYER_HPP_ 

#include "includes.hpp"
#include "../BaseData.hpp"

class BaseConnection;

class BaseLayer: public BaseData {
public:
   BaseLayer();
   virtual ~BaseLayer();
   virtual int initialize();
   virtual int allocate();
   virtual int setParams(
         Column* c,
         std::string layerName);
   void setPrev(BaseConnection* inConn){prevConn = inConn;}
   void setNext(BaseConnection* inConn){nextConn = inConn;}
   BaseConnection* getPrev(){return prevConn;};
   BaseConnection* getNext(){return nextConn;};

   virtual int applyActivation();
   virtual int forwardUpdate(int timestep);
   virtual int applyGradient();
   virtual int backwardsUpdate(int timestep);

   //Note: this function is inefficient, only use for debugging
   //Caller's responsible for freeing memory
   float * getHostU();
   float * getHostA();
   float * getHostG();

   float * getDeviceU(){return d_UData;}
   float * getDeviceA(){return d_AData;}
   float * getDeviceG(){return d_GData;}

   int getBSize(){return bSize;}
   int getYSize(){return ySize;}
   int getXSize(){return xSize;}
   int getFSize(){return fSize;}

   void printU();
   void printA();
   void printG();

   cudnnTensorDescriptor_t getLayerDescriptor(){return layerDescriptor;}
   //cudnnTensorDescriptor_t getGradientDescriptor(){return cudnnGDescriptor;}

protected:
   float * d_UData; //Feedforward preactivation function buffer
   float * d_AData; //Feedforward activity buffer
   float * d_GData; //Backpass gradient buffer
   //float * h_AData; //Host memory
   int bSize, ySize, xSize, fSize;
   cudnnTensorDescriptor_t layerDescriptor;
   //cudnnTensorDescriptor_t cudnnGDescriptor;
   virtual int setSize();
   BaseConnection* prevConn;
   BaseConnection* nextConn;

};
#endif 
