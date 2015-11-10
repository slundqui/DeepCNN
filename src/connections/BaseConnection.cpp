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

   //If stride is odd, patch size must also be odd (for making a layer with a stride of 1 be the same size)
   if(ystride % 2 != 0 && nyp % 2 == 0){
      std::cout << "Error: Connection " << name << " has an odd y stride, so the patch size must also be odd\n";
      exit(BAD_PARAM);
   }
   if(xstride % 2 != 0 && nxp % 2 == 0){
      std::cout << "Error: Connection " << name << " has an odd y stride, so the patch size must also be odd\n";
      exit(BAD_PARAM);
   }
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
