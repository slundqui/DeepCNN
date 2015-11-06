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

int BaseConnection::setParams(Column* c, std::string connName, int in_nyp, int in_nxp, int in_stride){
   BaseData::setParams(c, connName);
   name = connName;
   nyp = in_nyp;
   nxp = in_nxp;
   stride = in_stride;
   return SUCCESS;
}

//TODO this isn't needed once we make BaseConnection a pure virtual function
int BaseConnection::getNextLayerSize(int* ySize, int* xSize){
   std::cerr << "Error! Can't get connection size\n";
   exit(ERROR);
   return SUCCESS;
}

int BaseConnection::initialize(){
   if(!paramsSet){
      std::cerr << "Error! Connection did not set parameters before trying to initialize\n";
      exit(UNDEFINED_PARAMS);
   }
   //TODO
   return SUCCESS;
}
