/**
 * BaseData.cpp
 * Author: Sheng Lundquist
 **/

#include "BaseData.hpp"
#include "Column.hpp"

BaseData::BaseData(){
   paramsSet = false;
   gpuDataSize = 0;
}

BaseData::~BaseData(){
}

int BaseData::setParams(Column* c, std::string in_name){
   col = c;
   name = in_name;
   paramsSet = true;
   return SUCCESS;
}
