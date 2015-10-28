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
#include "../connections/BaseConnection.hpp"

class BaseLayer: public BaseData {
public:
   BaseLayer();
   virtual ~BaseLayer();
   virtual int initialize();
   virtual int setParams(
         std::string layerName,
         int in_bSize,
         int in_ySize,
         int in_xSize,
         int in_fSize);
   //virtual int updateState(double timef, double dt) = 0;
   void setPrev(BaseConnection* inConn){prevConn = inConn;}
   void setNext(BaseConnection* inConn){nextConn = inConn;}
   BaseConnection* getPrev(){return prevConn;};
   BaseConnection* getNext(){return nextConn;};
protected:
   float * d_AData;
   int bSize, ySize, xSize, fSize;

private:
   BaseConnection* prevConn;
   BaseConnection* nextConn;

};
#endif 
