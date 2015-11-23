/**
 * BaseCostFunction.hpp
 *
 * Author: Sheng Lundquist
 **/
#ifndef BASECOSTFUNCTION_HPP_ 
#define BASECOSTFUNCTION_HPP_ 

#include "includes.hpp"
#include "Activation.hpp"
#include "../Column.hpp"
#include <fstream>

class BaseCostFunction: public Activation{
public:
   BaseCostFunction();
   virtual ~BaseCostFunction();
   virtual int initialize();
   virtual int allocate();
   virtual int setParams(Column* c, std::string layerName, 
         std::string activationType,
         std::string outCostFile = "",
         std::string outAccuracyFile = "");

   virtual int forwardUpdate(int timestep);
   //virtual int backwardsUpdate(int timestep);
   virtual int applyGradient();

   float getHostTotalCost();
   float getHostAccuracy();

   virtual int calcTotalCost() = 0;
   virtual int calcGradient() = 0;
   virtual void reset(){numCorrect=0; numTests=0;}

   virtual int calcAccuracy() = 0;
protected:
   float* d_TotalCost;
   int numCorrect;
   int numTests;
};
#endif 
