/**
 * SoftmaxCost.hpp
 *
 * Author: Sheng Lundquist
 **/
#ifndef SOFTMAXCOST_HPP_ 
#define SOFTMAXCOST_HPP_ 

#include "includes.hpp"
#include "BaseCostFunction.hpp"
#include "../Column.hpp"
#include <fstream>
#include <cmath>

class SoftmaxCost: public BaseCostFunction{
public:
   SoftmaxCost();
   virtual ~SoftmaxCost();
   virtual int initialize();
   virtual int allocate();
   virtual int setParams(Column* c, std::string layerName, 
         //std::string activationType,
         int writePeriod = 1,
         std::string costFilename = "",
         std::string accuracyFilename = "",
         std::string estFilename = "");

   virtual float calcCost();
   virtual int calcCorrect();

   virtual int calcGradient();
   virtual int applyGradient();
   virtual int applyActivation();
protected:

   //Variables used by subclasses to store optimal gpu size
   int totalCostBlockSize;
   int totalCostGridSize;

   int calcGradBlockSize;
   int calcGradGridSize;


};
#endif 
