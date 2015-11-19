#include <limits.h>
#include "gtest/gtest.h"
#include <utils.hpp>
#include <src/layers/Image.hpp>
#include <connections/Convolution.hpp>

//Fixture for testing auto size generation
class convTests: public ::testing::Test{
   protected:
      virtual void SetUp(){
         myColNoBias = new Column(1 //batch
                            );

         myColBias = new Column(1 //batch
                            );

         inputNoBias = new Image();
         inputNoBias->setParams(myColNoBias, //column name
                               "input", //name
                               16, //ny
                               8, //nx
                               1, //features
                               "/home/sheng/workspace/DeepCNN/tests/testImgs/square_16x8x1.txt");//list of images

         inputBias = new Image();
         inputBias->setParams(myColBias, //column name
                               "input", //name
                               16, //ny
                               8, //nx
                               1, //features
                               "/home/sheng/workspace/DeepCNN/tests/testImgs/square_16x8x1.txt");//list of images

         convNoBias = new Convolution();
         convNoBias->setParams(myColNoBias, //column
                         "conv", //name
                         4, //nyp
                         4, //nxp
                         5, //nfp
                         2, //ystride
                         2, //xstride
                         0, //uniform init of weights
                         .5, //initVal of weights
                         "" //filename, not used
                         );

         convBias = new Convolution();
         convBias->setParams(myColBias, //column
                         "conv", //name
                         4, //nyp
                         4, //nxp
                         5, //nfp
                         2, //ystride
                         2, //xstride
                         0, //uniform init of weights
                         .5, //initVal of weights
                         "", //filename, not used
                         0, //uniform init of bias
                         5, //initVal of bias
                         "" //filename of bias, not used
                         );

         testLayerNoBias = new BaseLayer();
         testLayerNoBias->setParams(myColNoBias, //column
                              "test"); //name

         testLayerBias = new BaseLayer();
         testLayerBias->setParams(myColBias, //column
                              "test"); //name

         myColNoBias->addLayer(inputNoBias);
         myColNoBias->addConn(convNoBias);
         myColNoBias->addLayer(testLayerNoBias);

         myColBias->addLayer(inputBias);
         myColBias->addConn(convBias);
         myColBias->addLayer(testLayerBias);
      }
      virtual void TearDown(){
         delete myColNoBias;
         delete myColBias;
         delete inputNoBias;
         delete inputBias;
         delete testLayerNoBias;
         delete testLayerBias;
         delete convNoBias;
         delete convBias;
      }

      Column* myColNoBias;
      Column* myColBias;
      Image* inputNoBias;
      Image* inputBias;
      BaseLayer* testLayerNoBias;
      BaseLayer* testLayerBias;
      Convolution* convNoBias;
      Convolution* convBias;
};

TEST_F(convTests, sizeSetTest){
   myColNoBias->initialize();
   EXPECT_EQ(testLayerNoBias->getYSize(), 8);
   EXPECT_EQ(testLayerNoBias->getXSize(), 4);
   EXPECT_EQ(testLayerNoBias->getFSize(), 5);
}

TEST_F(convTests, initWeightsTest){
   myColNoBias->initialize();
   float* h_WData = convNoBias->getHostW();
   for(int i = 0; i < 2*1*2*5; i++){
      //std::cout << "idx " << i << " val " << h_WData[i] << "\n";
      ASSERT_EQ(h_WData[i], 0.5);
   }
   free(h_WData);
}

TEST_F(convTests, initBiasTest){
   myColBias->initialize();
   float* h_Bias = convBias->getHostB();
   for(int i = 0; i < 5; i++){
      //std::cout << "idx " << i << " val " << h_WData[i] << "\n";
      ASSERT_EQ(h_Bias[i], 5);
   }
   free(h_Bias);
}

TEST_F(convTests, feedforwardNoBiasTest){
   myColNoBias->initialize();

   //std::cout << "Input data before: \n";
   //float* h_inData = input->getHostA();
   //printMat(h_inData, 1, 16, 8, 1);
   //free(h_inData);

   myColNoBias->run(1);

   //h_inData = input->getHostA();
   //std::cout << "Input data after: \n";
   //printMat(h_inData, 1, 16, 8, 1);

   //std::cout << "Weights data: \n";
   //float* h_wData = conv->getHostW();
   //printMat(h_wData, 1, 2, 2, 5);
   ////for(int i = 0; i < 2*2*5*1; i++){
   ////   std::cout << "idx " << i << " val " << h_wData[i] << "\n";
   ////}
   //free(h_wData);

   float* h_outData = testLayerNoBias->getHostA();
   int idx = 0;
   for(int fi = 0; fi < 5; fi++){
      for(int yi = 0; yi < 8; yi++){
         for(int xi = 0; xi < 4; xi++){
            //Center of data
            if(xi >= 1 && xi < 3 && yi >= 3 && yi < 5){
               ASSERT_EQ(h_outData[idx], 2.0);
            }
            else{
               ASSERT_EQ(h_outData[idx], 0);
            }
            idx++;
         }
      }
   }
   free(h_outData);
}

TEST_F(convTests, feedforwardBiasTest){
   myColBias->initialize();

   //std::cout << "Input data before: \n";
   //float* h_inData = input->getHostA();
   //printMat(h_inData, 1, 16, 8, 1);
   //free(h_inData);

   myColBias->run(1);

   //h_inData = input->getHostA();
   //std::cout << "Input data after: \n";
   //printMat(h_inData, 1, 16, 8, 1);

   //std::cout << "Weights data: \n";
   //float* h_wData = conv->getHostW();
   //printMat(h_wData, 1, 2, 2, 5);
   ////for(int i = 0; i < 2*2*5*1; i++){
   ////   std::cout << "idx " << i << " val " << h_wData[i] << "\n";
   ////}
   //free(h_wData);

   float* h_outData = testLayerBias->getHostA();
   int idx = 0;
   for(int fi = 0; fi < 5; fi++){
      for(int yi = 0; yi < 8; yi++){
         for(int xi = 0; xi < 4; xi++){
            //Center of data
            if(xi >= 1 && xi < 3 && yi >= 3 && yi < 5){
               ASSERT_EQ(h_outData[idx], 7.0);
            }
            else{
               ASSERT_EQ(h_outData[idx], 5);
            }
            idx++;
         }
      }
   }
   free(h_outData);
}
