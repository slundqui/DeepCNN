/**
 * LeastSquaresCost.hpp
 *
 * Author: Sheng Lundquist
 **/
#ifndef LEASTSQAURESCOST_HPP_ 
#define LEASTSQAURESCOST_HPP_ 

#include "includes.hpp"
#include "BaseCostFunction.hpp"
#include "../Column.hpp"
#include <fstream>

class LeastSquaresCost: public BaseCostFunction{
public:
   LeastSquaresCost();
   virtual ~LeastSquaresCost();
   virtual int initialize();
   virtual int allocate();
//   virtual int setParams(Column* c, std::string layerName, std::string outCostFile);

   virtual int calcTotalCost();
   virtual int calcGradient();
   virtual int calcAccuracy();
   //virtual int applyActivation();
protected:

   //Variables used by subclasses to store optimal gpu size
   int totalCostGridSize;
   int totalCostBlockSize;

   int calcGradBlockSize;
   int calcGradGridSize;

   
   
   


};
#endif 
