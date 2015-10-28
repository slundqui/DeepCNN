/**
 * Column.cpp
 * Author: Sheng Lundquist
 **/

#include "Column.hpp"

Column::Column()
{
   //initialize();
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


