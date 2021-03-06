/**
 * MatInput.hpp
 *
 * Author: Sheng Lundquist
 **/
#ifndef MATINPUT_HPP_ 
#define MATINPUT_HPP_ 

#include "includes.hpp"
#include "BaseLayer.hpp"
#include "../Column.hpp"
#include <algorithm>
#include <fstream>

class MatInput: public BaseLayer {
public:
   MatInput();
   virtual ~MatInput();
   virtual int initialize();
   virtual int allocate();
   virtual int setParams(
         Column* c,
         std::string layerName,
         int in_ySize,
         int in_xSize,
         int num_features,
         std::string in_name,
         bool in_shuffle=false);
   virtual int applyActivation();
   virtual int forwardUpdate(int timestep);
   virtual int backwardsUpdate(int timestep);
   int rewind(){exampleIdx = 0; return SUCCESS;}
   std::vector<int> getOrderVector(){return orderVector;}
   int getCurrentIdx(int batch);
   
protected:
   virtual int setSize();
   int loadMatInput();
   std::string matFilename;
   int readMat();
   float* h_data;
   int numExamples;
   int exampleIdx;
   bool shuffle;
private:
   std::vector<int> orderVector; //A vector keeping track of random permutations of the dataset
   int shuffleOrder();
   
   


};
#endif 
