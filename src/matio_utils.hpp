#ifndef MATIO_UTILS_HPP_ 
#define MATIO_UTILS_HPP_ 

#include <matio.h>
#include <string>
#include <cuda_runtime.h>
#include "cuda_utils.hpp"

void writeDeviceData(std::string matFilename, int nDims, size_t * dims, float* d_data){
   mat_t *matfp;
   matvar_t *matvar;
   size_t size = sizeof(float);
   for(int i = 0;  i < nDims; i++){
      size *= dims[i];
   }
   float* h_data = (float*) malloc(size);
   CudaError(cudaMemcpy(h_data, d_data, size, cudaMemcpyDeviceToHost));

   matfp = Mat_CreateVer(matFilename.c_str(), NULL, MAT_FT_DEFAULT);
   if(matfp == NULL){
      std::cerr << "Error opening MAT file " << matFilename<< "\n";
      exit(-1);
   }

   matvar = Mat_VarCreate("data", MAT_C_SINGLE, MAT_T_SINGLE, nDims, dims, h_data, MAT_F_DONT_COPY_DATA);
   if(!matvar){
      Mat_Close(matfp);
      std::cerr << "Error creating var\n";
      exit(-1);
   }

   Mat_VarWrite(matfp, matvar, MAT_COMPRESSION_NONE);

   Mat_VarFree(matvar);
   Mat_Close(matfp);
   free(h_data);
}

void readDataToDevice(std::string matFilename, float* d_data, int* nDims, size_t** dims){
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

   *nDims = matvar->rank;
   *dims = (size_t*)malloc(*nDims * sizeof(size_t));
   for(int i = 0; i < *nDims; i++){
      (*dims)[i] = matvar->dims[i];
      size *= matvar->dims[i];
   }

   float *h_data = static_cast<float*>(matvar->data);
   
   CudaError(cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice));
   Mat_VarFree(matvar);
   Mat_Close(matfp);
}

#endif
