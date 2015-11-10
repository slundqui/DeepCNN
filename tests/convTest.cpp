#include <limits.h>
#include "gtest/gtest.h"
#include <utils.hpp>
#include <src/layers/Image.hpp>
#include <connections/Convolution.hpp>

//Fixture for testing auto size generation
class convTests: public ::testing::Test{
   protected:
      virtual void SetUp(){
         myCol = new Column(1, //batch
                            16, //ny
                            8); //nx
         input = new Image();
         input->setParams(myCol, //column name
                               "input", //name
                               1, //features
                               "/home/sheng/workspace/DeepCNN/tests/testImgs/square_16x8x1.txt");//list of images

         conv = new Convolution();
         conv->setParams(myCol, //column
                         "conv", //name
                         4, //nyp
                         4, //nxp
                         5, //nfp
                         2, //ystride
                         2, //xstride
                         0, //uniform init
                         .5, //initVal
                         "" //filename, not used
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
      Image* input;
      BaseLayer* testLayer;
      Convolution* conv;
};

TEST_F(convTests, sizeSetTest){
   myCol->initialize();
   EXPECT_EQ(testLayer->getYSize(), 8);
   EXPECT_EQ(testLayer->getXSize(), 4);
   EXPECT_EQ(testLayer->getFSize(), 5);
}

TEST_F(convTests, initWeightsTest){
   myCol->initialize();
   float* h_WData = conv->getHostW();
   for(int i = 0; i < 2*1*2*5; i++){
      //std::cout << "idx " << i << " val " << h_WData[i] << "\n";
      ASSERT_EQ(h_WData[i], 0.5);
   }
   free(h_WData);
}

TEST_F(convTests, feedforwardTest){
   myCol->initialize();

   //std::cout << "Input data before: \n";
   //float* h_inData = input->getHostA();
   //printMat(h_inData, 1, 16, 8, 1);
   //free(h_inData);

   myCol->run(1);

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

   float* h_outData = testLayer->getHostA();
   int idx = 0;
   for(int fi = 0; fi < 5; fi++){
      for(int yi = 0; yi < 8; yi++){
         for(int xi = 0; xi < 4; xi++){
            //Center of data
            if(xi >= 1 && xi < 3 && yi >= 3 && yi < 5){
               //ASSERT_EQ(h_outData[idx], 2.0);
            }
            else{
               //ASSERT_EQ(h_outData[idx], 0);
            }
            idx++;
         }
      }
   }
   free(h_outData);

}
