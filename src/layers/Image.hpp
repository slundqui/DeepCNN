/**
 * Image.hpp
 *
 * Author: Sheng Lundquist
 **/
#ifndef IMAGE_HPP_ 
#define IMAGE_HPP_ 

//Defined for CImg to not need additional libraries
#define cimg_display 0

#include "includes.hpp"
#include "BaseLayer.hpp"
#include "CImg.h"
#include "../Column.hpp"
#include <fstream>

using namespace cimg_library;

class Image: public BaseLayer {
public:
   Image();
   virtual ~Image();
   virtual int initialize();
   virtual int setParams(
         Column* c,
         std::string layerName,
         int num_features,
         std::string inList);
   virtual int forwardUpdate(int timestep);
   virtual int backwardsUpdate(int timestep);
protected:
   virtual int setSize();
   int loadImage(std::string filename, int batchIdx);
   std::string filenameList;
   std::ifstream listFile;
   


};
#endif 
