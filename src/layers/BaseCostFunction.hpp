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
         int in_writePeriod = 1,
         std::string in_costFilename = "",
         std::string in_accuracyFilename = "",
         std::string in_estFilename = "");

   virtual int forwardUpdate(int timestep);
   //virtual int backwardsUpdate(int timestep);
   virtual int applyGradient();

   float getHostTotalCost();
   float getHostAccuracy();

   virtual int calcTotalCost() = 0;
   virtual int calcGradient() = 0;

   virtual int calcAccuracy() = 0;
protected:
   virtual void reset(){sumCost=0; numCorrect=0; numTests=0;}
   float* d_TotalCost;
   int numCorrect;
   int numTests;
   float sumCost;
   int writePeriod;
   std::ofstream costFile;
   std::ofstream accuracyFile;
   std::ofstream estFile;
   std::string costFilename;
   std::string accuracyFilename;
   std::string estFilename;

   float* h_estBuf;
   float* h_gtBuf;
private:
   int writeEst();
   int updateHostData();
};
#endif 
