/**
 * Image.cpp
 * Author: Sheng Lundquist
 **/

#include "Image.hpp"
#include "../utils.hpp"
#include "../cuda_utils.hpp"


Image::Image()
{
}

Image::~Image(){
   listFile.close();
}

int Image::setParams(Column* c, std::string layerName, int in_ySize, int in_xSize, int num_features, std::string inList){
   filenameList = inList;
   ySize = in_ySize;
   xSize = in_xSize;
   fSize = num_features;
   return BaseLayer::setParams(c, layerName);
}

int Image::setSize(){
   //Image size should already be set, do nothing
   return SUCCESS;
}


int Image::initialize(){
   BaseLayer::initialize();

   //Open list of filenames
   listFile.open(filenameList.c_str(), std::ifstream::in);
   char c = listFile.get();
   if(listFile.bad()){
      std::cerr << "Error opening file " << filenameList << "\n";
      exit(FILEIO_ERROR);
   }
   listFile.clear();
   listFile.seekg(0);

   return SUCCESS;
}

int Image::allocate(){
   BaseLayer::allocate();
   //Start off with a forward pass to load 1st image (with -1 timestep)
   //forwardUpdate(-1);
   return SUCCESS;
}

//Loads image onto GPU
int Image::loadImage(std::string filename, int batchIdx){
   //Get image
   CImg<float> image(filename.c_str());
   if(!image){
      std::cerr << "Error, file " << filename << " does not exist\n";
      exit(FILEIO_ERROR);
   }
   //Check image size with layer size
   if(image.width() != xSize || image.height() != ySize || image.spectrum() != fSize){
      std::cerr << "Error, image (" << image.height() << ", " <<
         image.width() << ", " << image.spectrum() <<
         ") does not match image layer dimensions (" << ySize << ", " << xSize << ", " << fSize << ")\n";
      exit(FILEIO_ERROR);
   }

   //Load into GPU memory
   int numVals = ySize * xSize * fSize;
   //Offset based on batch
   float * d_batchAData = &(d_AData[batchIdx * numVals]);
   float* h_data = image.data();
   
   for(int i = 0; i < xSize * ySize * fSize; i++){
      h_data[i] = h_data[i] / 255; //Scale from 0 to 1
   }

   CudaError(cudaMemcpy(d_batchAData, h_data, numVals*sizeof(float), cudaMemcpyHostToDevice));
   return SUCCESS;
}

int Image::applyActivation(){
   //Do nothing, as forwardUpdate takes care of everything
   return SUCCESS;
}

int Image::forwardUpdate(int timestep){
   std::string filename;

   //First image was already loaded on first timestep
   //if(timestep == 0){
   //   return SUCCESS;
   //}

   //Read image per batch
   for(int b = 0; b < bSize; b++){
      getline(listFile, filename);
      if(listFile.eof()){
         listFile.clear();
         listFile.seekg(0);
         std::cout << "Rewinding file " << filenameList << "\n";
         getline(listFile, filename);
         if(listFile.eof()){
            std::cerr << "Error, file " << filenameList << " empty\n";
            exit(FILEIO_ERROR);
         }
      }
      //if(DEBUG) std::cout << "Reading image " << filename << " into batch " << b << "\n";
      std::cout << "Reading image " << filename << " into batch " << b << "\n";
      loadImage(filename, b);
   }
   return SUCCESS;
}

int Image::backwardsUpdate(int timestep){
   BaseLayer::backwardsUpdate(timestep);
   
   return SUCCESS;
};



