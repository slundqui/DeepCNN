/**
 * Convolution.cpp
 * Author: Sheng Lundquist
 **/

#include "Convolution.hpp"

Convolution::Convolution(){
}

Convolution::~Convolution(){
}

int Convolution::setParams(Column* c, std::string connName, int in_nyp, int in_nxp, int in_stride){

   return BaseConnection::setParams(c, connName, in_nyp, in_nxp, in_stride);
}

int Convolution::initialize(){
   if(!paramsSet){
      std::cerr << "Error! Connection did not set parameters before trying to initialize\n";
      exit(UNDEFINED_PARAMS);
   }
   //TODO
   return SUCCESS;
}

//TODO
int Convolution::updateWeights(int timestep){
   return SUCCESS;
}

//TODO
int Convolution::deliver(){
   return SUCCESS;
}
