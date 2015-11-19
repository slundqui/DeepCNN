/**
 * Activation.hpp
 *
 * The base abstract layer
 * Each layer must define 3 methods: parameterSetting, initialization, and update
 * 
 * Each layer must also define a buffer.
 * AData: The unit activation buffer to be passed onto the next layer via convolution
 *
 * Author: Sheng Lundquist
 **/
#ifndef ACTIVATION_HPP_ 
#define ACTIVATION_HPP_ 

#include "includes.hpp"
#include "BaseLayer.hpp"

class BaseConnection;

class Activation: public BaseLayer{
public:
   Activation();
   virtual ~Activation();
   virtual int initialize();
   virtual int allocate();
   virtual int setParams(
         Column* c,
         std::string layerName,
         std::string activationMode);

   //TODO make virtual, but need to change test
   virtual int applyActivation();
   //virtual int forwardUpdate(int timestep);
   virtual int backwardsUpdate(int timestep);
protected:
   cudnnActivationMode_t activationMode;

};
#endif 
