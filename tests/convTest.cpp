#include <limits.h>
#include "gtest/gtest.h"
#include <src/layers/Image.hpp>
#include <connections/Convolution.hpp>

//Fixture for testing auto size generation
class convTests: public ::testing::Test{
   protected:
      virtual void SetUp(){
         myCol = new Column(3, //batch
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
                         2, //nyp
                         2, //nxp
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
}
