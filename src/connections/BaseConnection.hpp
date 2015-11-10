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
#include "../BaseData.hpp"

//Forward declaration of BaseLayer
class BaseLayer;

class BaseConnection : public BaseData{
public:
   BaseConnection();
   virtual ~BaseConnection();

   virtual int setParams(
         Column* c,
         std::string connName,
         int in_nyp,
         int in_nxp,
         int in_nfp,
         int in_ystride,
         int in_xstride);

   virtual int initialize();
   virtual int allocate();
   virtual int setNextLayerSize(int* ySize, int* xSize, int* fSize);
   //virtual int updateState(double timef, double dt) = 0;
   void setNext(BaseLayer* inLayer){nextLayer = inLayer;}
   void setPrev(BaseLayer* inLayer){prevLayer = inLayer;}
   BaseLayer* getPrev(){return prevLayer;};
   BaseLayer* getNext(){return nextLayer;};
   virtual int updateWeights(int timestep){return SUCCESS;}
   virtual int deliver(){return SUCCESS;}

   BaseLayer* prevLayer;
   BaseLayer* nextLayer;
protected:
   int nyp, nxp, nfp, xstride, ystride;
private:
};
#endif 
