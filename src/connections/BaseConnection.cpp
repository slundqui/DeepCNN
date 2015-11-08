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

int BaseConnection::setParams(Column* c, std::string connName, int in_nyp, int in_nxp, int in_nfp, int in_ystride, int in_xstride){
   BaseData::setParams(c, connName);
   name = connName;
   nyp = in_nyp;
   nxp = in_nxp;
   nfp = in_nfp;
   ystride = in_ystride;
   xstride = in_xstride;
   return SUCCESS;
}

int BaseConnection::setNextLayerSize(int* ySize, int* xSize, int* fSize){
   std::cerr << "Error! Can't get connection size\n";
   exit(ERROR);
   return SUCCESS;
}

int BaseConnection::initialize(){
   if(!paramsSet){
      std::cerr << "Error! Connection did not set parameters before trying to initialize\n";
      exit(UNDEFINED_PARAMS);
   }
   return SUCCESS;
}

int BaseConnection::allocate(){
   //TODO
   return SUCCESS;
}
