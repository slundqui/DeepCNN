/**
 * Column.cpp
 * Author: Sheng Lundquist
 **/

#include "Column.hpp"

Column::Column(int in_bSize, int in_ySize, int in_xSize)
{
   //initialize();
   bSize = in_bSize;
   xSize = in_xSize;
   ySize = in_ySize;
   timestep = 0;
}

Column::~Column(){
}

int Column::addLayer(BaseLayer* inLayer){
   assert(inLayer);
   //Make sure inLayer's parameters have been set
   if(!inLayer->isParamsSet()){
      std::cerr << "Error! Layer did not set parameters before trying to add layer to column\n";
      exit(UNDEFINED_PARAMS);
   }

   //Add to runList
   runList.push_back(inLayer);

   //If not the first element in runList, make sure previous data structure is a connection
   if(runList.size() > 1){
      BaseConnection * prevConn = dynamic_cast <BaseConnection*>(runList[runList.size()-2]);
      if(!prevConn){
         std::cerr << "Error adding layer " << inLayer->getName() << " as last item on runList is not a connection\n";
         exit(INVALID_DATA_ADDITION);
      }
      prevConn->setNext(inLayer);
      inLayer->setPrev(prevConn);
   }

   layerList.push_back(inLayer);

   return SUCCESS;
}

int Column::addConn(BaseConnection* inConn){
   assert(inConn);
   //Make sure inConn's parameters have been set
   if(!inConn->isParamsSet()){
      std::cerr << "Error! Connection did not set parameters before trying to add conn to column\n";
      exit(UNDEFINED_PARAMS);
   }
   //Add to runList
   runList.push_back(inConn);

   //If not the first element in runList, make sure previous data structure is a connection
   if(runList.size() > 1){
      BaseLayer * prevLayer = dynamic_cast <BaseLayer*>(runList[runList.size()-2]);
      if(!prevLayer){
         std::cerr << "Error adding conection " << inConn->getName() << " as last item on runList is not a layer \n";
         exit(INVALID_DATA_ADDITION);
      }
      prevLayer->setNext(inConn);
      inConn->setPrev(prevLayer);
   }
   else{
      std::cerr << "Error adding conection " << inConn->getName() << " as first item on runList can't be a connection\n";
      exit(INVALID_DATA_ADDITION);
   }

   connList.push_back(inConn);

   return SUCCESS;
}

int Column::initialize(){
   for(std::vector<BaseData*>::iterator it = runList.begin(); it != runList.end(); ++it){
      (*it)->initialize();
   }
   return SUCCESS;
}

//TODO
int Column::run(int numTimesteps){
   for(timestep = 0; timestep < numTimesteps; timestep++){
      //Update all connections first (learning)
      for(std::vector<BaseConnection*>::iterator connIt = connList.begin();
            connIt != connList.end(); ++connIt){
         (*connIt)->updateWeights(timestep);
      }
      //Update all layers (feedforward)
      for(std::vector<BaseLayer*>::iterator layerIt = layerList.begin();
            layerIt != layerList.end(); ++layerIt){
         (*layerIt)->forwardUpdate(timestep);
      }
      //Update all gradients (backprop)
      for(std::vector<BaseLayer*>::reverse_iterator rLayerIt = layerList.rbegin();
            rLayerIt != layerList.rend(); ++rLayerIt){
         (*rLayerIt)->backwardsUpdate(timestep);
      }
   }
}


int Column::initAndRun(int numTimesteps){
   initialize();
   run(numTimesteps);
   return SUCCESS;
}






