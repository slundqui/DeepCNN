/**
 * BaseData.hpp
 *
 * The base abstract class that both layers and connections inherit from
 * Each layer must define initialization
 *
 * Author: Sheng Lundquist
 **/
#ifndef BASEDATA_HPP_ 
#define BASEDATA_HPP_ 

#include "includes.hpp"

class Column;

class BaseData{
public:
   BaseData();
   virtual ~BaseData();
   virtual int initialize() = 0;
   bool isParamsSet(){return paramsSet;}
   virtual int setParams(Column* c, std::string in_name);
   std::string getName(){return name;}
protected:
   bool paramsSet;
   std::string name;
   Column* col;
};
#endif 
