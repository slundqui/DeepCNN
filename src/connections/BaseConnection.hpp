/**
 * BaseConnection.hpp
 *
 * The base abstract layer
 * Each layer must define 2 methods: initialization and update
 *
 * Author: Sheng Lundquist
 **/
#ifndef BASECONNECTION_HPP_ 
#define BASECONNECTION_HPP_ 

#include "includes.hpp"

class BaseConnection{
public:
   BaseConnection();
   virtual ~BaseConnection();
   //virtual int initialize() = 0;
   //virtual int updateState(double timef, double dt) = 0;
};
#endif 
