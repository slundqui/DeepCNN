/**
 * BaseCostFunction.hpp
 *
 * Author: Sheng Lundquist
 **/
#ifndef BASECOSTFUNCTION_HPP_ 
#define BASECOSTFUNCTION_HPP_ 

#include "includes.hpp"
#include "BaseLayer.hpp"
#include "../Column.hpp"
#include <fstream>

class BaseCostFunction: public BaseLayer {
public:
   BaseCostFunction();
   virtual ~BaseCostFunction();
   virtual int initialize();
   virtual int allocate();
   virtual int forwardUpdate(int timestep);
   virtual int backwardsUpdate(int timestep);
   virtual int setParams(Column* c, std::string layerName, 
         std::string outCostFile = "",
         std::string outAccuracyFile = "");

   virtual const float* getHostTotalCost();

   virtual int calcTotalCost() = 0;
   virtual int calcGradient() = 0;
protected:
   float* d_TotalCost;
   float* h_TotalCost;
};
#endif 
