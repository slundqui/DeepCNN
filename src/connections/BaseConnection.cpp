/**
 * BaseConnection.cpp
 * Author: Sheng Lundquist
 **/

#include "BaseConnection.hpp"

BaseConnection::BaseConnection(){
   prevLayer = NULL;
   nextLayer = NULL;
}

BaseConnection::~BaseConnection(){
}

int BaseConnection::setParams(std::string layerName, int in_nyp, int in_nxp){
   name = layerName;
   nyp = in_nyp;
   nxp = in_nxp;
   paramsSet = true;
}

int BaseConnection::initialize(){
   if(!paramsSet){
      std::cerr << "Error! Connection did not set parameters before trying to initialize\n";
      exit(UNDEFINED_PARAMS);
   }
   //TODO
   return SUCCESS;
}
