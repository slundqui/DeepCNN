/**
 * MatInput.cpp
 * Author: Sheng Lundquist
 **/

#include "MatInput.hpp"
#include "../utils.hpp"
#include "../cuda_utils.hpp"


MatInput::MatInput()
{
   exampleIdx = 0;
   h_data = NULL;
}

MatInput::~MatInput(){
   free(h_data);
}

int MatInput::setParams(Column* c, std::string layerName, int in_ySize, int in_xSize, int num_features, std::string in_name){
   matFilename = in_name;
   ySize = in_ySize;
   xSize = in_xSize;
   fSize = num_features;
   matFilename = in_name; 
   return BaseLayer::setParams(c, layerName);
}

int MatInput::setSize(){
   //MatInput size should already be set, do nothing
   return SUCCESS;
}

int MatInput::readMat(){
   mat_t *matfp;
   matvar_t *matvar;
   size_t size = sizeof(float); 
   //Reading from mat
   matfp = Mat_Open(matFilename.c_str(), MAT_ACC_RDONLY);
   if(matfp == NULL){
      std::cerr << "Error opening MAT file " << matFilename << "\n";
      exit(-1);
   }

   matvar = Mat_VarRead(matfp, (char*)"data");
   if(!matvar){
      Mat_Close(matfp);
      std::cerr << "Error reading var\n";
      exit(-1);
   }

   assert(matvar->data_type == 7); //Single data type

   int nDims = matvar->rank;
   assert(nDims == 4);

   size_t dims[nDims];
   for(int i = 0; i < nDims; i++){
      dims[i] = matvar->dims[i];
      size *= matvar->dims[i];
   }

   assert(dims[0] == (size_t)xSize); 
   assert(dims[1] == (size_t)ySize);
   assert(dims[2] == (size_t)fSize);
   numExamples = dims[3];

   h_data = static_cast<float*>(matvar->data);
   return SUCCESS;
}

int MatInput::initialize(){
   readMat();
   BaseLayer::initialize();
   return SUCCESS;
}

int MatInput::allocate(){
   BaseLayer::allocate();

   //Start off with a forward pass to load 1st image (with -1 timestep)
   forwardUpdate(-1);
   return SUCCESS;
}

int MatInput::loadMatInput(){
   //Load into GPU memory
   int numVals = ySize * xSize * fSize;
   for(int bi = 0; bi < bSize; bi++){
      float * d_batchAData = &(d_AData[bi * numVals]);
      float * h_batchAData = &(h_data[exampleIdx * numVals]);
      CudaError(cudaMemcpy(d_batchAData, h_batchAData, numVals*sizeof(float), cudaMemcpyHostToDevice));
      exampleIdx++;
      //If out of examples, reset
      if(exampleIdx >= numExamples){
         exampleIdx = 0;
      }
   }
   return SUCCESS;
}

int MatInput::applyActivation(){
   //Does nothing, forward update takes care of things
   return SUCCESS;
}

int MatInput::forwardUpdate(int timestep){
   std::string filename;

   //First image was already loaded on first timestep
   if(timestep == 0){
      return SUCCESS;
   }

   //Read image per batch
   loadMatInput();

   return SUCCESS;
}

//Backwards update does nothing, as image does not have a gradient
int MatInput::backwardsUpdate(int timestep){
   BaseLayer::backwardsUpdate(timestep);
   return SUCCESS;
};



