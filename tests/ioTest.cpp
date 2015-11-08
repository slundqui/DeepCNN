#include "gtest/gtest.h"
#include <iostream>
#include <src/utils.hpp>

//Fixture for testing auto size generation
class mat_writing: public ::testing::Test{
   protected:
      virtual void SetUp(){
      }
      virtual void TearDown(){
      }
};

//For use in writing to a mat file 
TEST_F(mat_writing, matTest){
   //Writing to mat
   //mat_t *matfp;
   //matvar_t *matvar;
   const char* outName = "tests/out/test.mat";
   int nb = 2;
   int nf = 3;
   int ny = 5;
   int nx = 7;
   float* h_data = (float*) malloc(nb*nf*nx*ny * sizeof(float));
   float* d_data;
   CudaError(cudaMalloc(&d_data, nb*nf*nx*ny*sizeof(float)));

   for(int i = 0; i < nb*ny*nx*nf; i++){
      h_data[i] = (float)i; //Initializing data
   }

   CudaError(cudaMemcpy(d_data, h_data, nb*nf*nx*ny*sizeof(float), cudaMemcpyHostToDevice));
   cudaDeviceSynchronize();
   size_t array_dim[4] = {(size_t)nb, (size_t)nf, (size_t)ny, (size_t)nx};

   writeDeviceData(outName, 4, array_dim, d_data);

   int numDim;
   size_t* dims;
   int outNb, outNf, outNy, outNx;
   float* h_outdata = (float*) malloc(nb*nf*nx*ny *sizeof(float));
   float* d_outdata;
   CudaError(cudaMalloc(&d_outdata, nb*nf*nx*ny*sizeof(float)));
   readDataToDevice(outName, d_outdata, &numDim, &dims);

   EXPECT_EQ(numDim, 4);
   EXPECT_EQ(dims[0], 2);
   EXPECT_EQ(dims[1], 3);
   EXPECT_EQ(dims[2], 5);
   EXPECT_EQ(dims[3], 7);

   CudaError(cudaMemcpy(h_outdata, d_outdata, nb*nf*nx*ny*sizeof(float), cudaMemcpyDeviceToHost));

   for(int i = 0; i < nb*ny *nx*nf; i++){
      ASSERT_EQ(h_outdata[i], h_data[i]);
   }

   free(h_data);
   free(h_outdata);
   cudaFree(d_data);
   free(dims);
   cudaFree(d_outdata);
   
   return;
}

