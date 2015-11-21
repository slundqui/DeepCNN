#include "gtest/gtest.h"
#include <iostream>
#include <src/utils.hpp>
#include <src/layers/MatInput.hpp>
#include <connections/Convolution.hpp>

//Fixture for testing auto size generation
class mat_fileio: public ::testing::Test{
   protected:
      virtual void SetUp(){
         myCol = new Column(3 //batch
                            );

         input = new MatInput();
         input->setParams(myCol, //column name
                               "input", //name
                               16, //ny
                               8, //nx
                               1, //features
                               "/home/sheng/workspace/DeepCNN/tests/testMats/idxcount_8x16x1x5.mat");//input mat

         conv = new Convolution();
         conv->setParams(myCol, //column
                         "conv", //name
                         4, //nyp
                         2, //nxp
                         5, //nfp
                         2, //ystride
                         2, //xstride
                         2, //from file init val
                         0, //initVal, not used
                         "/home/sheng/workspace/DeepCNN/tests/testMats/idxcount_2x4x1x5.mat" //filename
                         );

         testLayer = new BaseLayer();
         testLayer->setParams(myCol, //column
                              "test"); //name
         myCol->addLayer(input);
         myCol->addConn(conv);
         myCol->addLayer(testLayer);
      }
      virtual void TearDown(){
         delete myCol;
         delete input;
         delete testLayer;
         delete conv;
      }

      Column* myCol;
      MatInput* input;
      BaseLayer* testLayer;
      Convolution* conv;
};

//For reading a mat 
TEST_F(mat_fileio, loadMat){
   myCol->initialize();
   myCol->run(1);
   float* h_AData = input->getHostA();
   for(int i = 0; i < 8*16*3; i++){
      ASSERT_EQ(h_AData[i], i+1);
   }

   //Run to update mats
   free(h_AData);
   myCol->run(1);
   h_AData = input->getHostA();
   int offset = 8*16*3;
   //First 2 batches should continue counting
   for(int i = 0; i < 8*16*2; i++){
      ASSERT_EQ(h_AData[i], offset+i+1);
   }
   //Last batch should have been reset
   offset = 8*16*2;
   for(int i = 0; i < 8*16; i++){
      ASSERT_EQ(h_AData[offset+i], i+1);
   }

}

TEST_F(mat_fileio, loadWeights){
   myCol->initialize();
   myCol->run(1);
   float* h_WData = conv->getHostW();
   for(int i = 0; i < 4*2*5; i++){
      ASSERT_EQ(h_WData[i], i+1);
   }
   free(h_WData);
}

//For use in writing to a mat file 
TEST(mattest, matTest){
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
   //Matlab arrays spin from fastest to slowest
   size_t array_dim[4] = {(size_t)nx, (size_t)ny, (size_t)nf, (size_t)nb};

   writeDeviceData(outName, 4, array_dim, d_data);

   int numDim;
   size_t* dims;
   int outNb, outNf, outNy, outNx;
   float* h_outdata = (float*) malloc(nb*nf*nx*ny *sizeof(float));
   float* d_outdata;
   CudaError(cudaMalloc(&d_outdata, nb*nf*nx*ny*sizeof(float)));
   readDataToDevice(outName, d_outdata, &numDim, &dims);

   EXPECT_EQ(numDim, 4);
   EXPECT_EQ(dims[0], 7);
   EXPECT_EQ(dims[1], 5);
   EXPECT_EQ(dims[2], 3);
   EXPECT_EQ(dims[3], 2);

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

