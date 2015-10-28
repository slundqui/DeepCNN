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

class BaseLayer{
public:
   BaseLayer(std::string layerName);
   virtual ~BaseLayer();
   virtual int initialize();
   virtual int setParams(int in_bSize, int in_ySize, int in_wSize, int in_fSize);
   //virtual int updateState(double timef, double dt) = 0;
protected:
   float * d_AData;
   int bSize, ySize, wSize, fSize;
   std::string name;

private:
   bool paramsSet;

};
#endif 
