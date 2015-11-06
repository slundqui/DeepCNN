/**
 * Convolution.hpp
 *
 * Author: Sheng Lundquist
 **/
#ifndef CONVOLUTION_HPP_ 
#define CONVOLUTION_HPP_ 

#include "includes.hpp"
#include "BaseConnection.hpp"

//Forward declaration of BaseLayer
class Convolution : public BaseConnection{
public:
   Convolution();
   virtual ~Convolution();
   virtual int setParams(
         Column* c,
         std::string connName,
         int in_nyp,
         int in_nxp,
         int in_stride);
   virtual int initialize();
   virtual int updateWeights(int timestep);
   virtual int deliver();
};
#endif 
