/**
 * Column.hpp
 *
 * The container object that contains a list of all layers and connections
 *
 * Author: Sheng Lundquist
 **/
#ifndef COLUMN_HPP_ 
#define COLUMN_HPP_ 
#include "layers/BaseLayer.hpp"
#include "connections/BaseConnection.hpp"
#include "layers/MatInput.hpp"

class MatInput;
class Column{
public:
   Column(int in_bSize, unsigned int in_seed = 1234567890);
   virtual ~Column();
   int addLayer(BaseLayer* inLayer);
   int addConn(BaseConnection* inConn);
   int addGroundTruth(MatInput* inLayer);
   MatInput* getGroundTruthLayer(){return groundTruthLayer;}
   int getBSize(){return bSize;}
   //int getXSize(){return xSize;}
   //int getYSize(){return ySize;}

   virtual int initialize();
   virtual int run(int numTimesteps);
   virtual int initAndRun(int numTimesteps);
   cudnnHandle_t getCudnnHandle(){return cudnn_handle;}
   void query_device(int id);
   unsigned int getSeed(){return seed;}
   int getTimestep(){return timestep;}
   //virtual int updateState(double timef, double dt);
private:
   std::vector<BaseLayer*> layerList;
   std::vector<BaseConnection*> connList;

   //Ground truth layer
   MatInput* groundTruthLayer;

   //Run list contains the order in which to achieve forward and backward passes
   std::vector<BaseData*> runList;

   int bSize;
   int timestep;

   //Total size of needed gpu memory
   size_t totalGpuSize;

   //Device properties
   cudnnHandle_t cudnn_handle;
   cudaDeviceProp devProp;
   unsigned int seed;

   

};
#endif 
