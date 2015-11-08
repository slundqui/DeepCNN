/**
 * Image.cpp
 * Author: Sheng Lundquist
 **/

#include "Image.hpp"


Image::Image()
{
}

Image::~Image(){
   listFile.close();
}

int Image::setParams(Column* c, std::string layerName, int num_features, std::string inList){
   filenameList = inList;
   ySize = c->getYSize();
   xSize = c->getXSize();
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
   CudaError(cudaMemcpy(d_batchAData, image.data(), numVals*sizeof(float), cudaMemcpyHostToDevice));
   return SUCCESS;
}

int Image::forwardUpdate(int timestep){
   std::string filename;

   //Read image per batch
   for(int b = 0; b < bSize; b++){
      getline(listFile, filename);
      if(listFile.eof()){
         listFile.clear();
         listFile.seekg(0);
         if(DEBUG) std::cout << "Rewinding file " << filenameList << "\n";
         getline(listFile, filename);
         if(listFile.eof()){
            std::cerr << "Error, file " << filenameList << " empty\n";
            exit(FILEIO_ERROR);
         }
      }
      if(DEBUG) std::cout << "Reading image " << filename << " into batch " << b << "\n";
      loadImage(filename, b);
   }
   return SUCCESS;
}

//Backwards update does nothing, as image does not have a gradient
int Image::backwardsUpdate(int timestep){
   return SUCCESS;
};



