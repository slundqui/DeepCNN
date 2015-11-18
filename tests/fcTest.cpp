#include <limits.h>
#include "gtest/gtest.h"
#include <utils.hpp>
#include <src/layers/Image.hpp>
#include <connections/FullyConnected.hpp>

//Fixture for testing auto size generation
class fcTests: public ::testing::Test{
   protected:
      virtual void SetUp(){
         myCol = new Column(1 //batch
                            );

         input = new Image();
         input->setParams(myCol, //column name
                               "input", //name
                               16, //ny
                               8, //nx
                               1, //features
                               "/home/sheng/workspace/DeepCNN/tests/testImgs/square_16x8x1.txt");//list of images

         fc = new FullyConnected();
         fc->setParams(myCol, //column
                         "fc", //name
                         5, //nfp
                         0, //uniform init of weights
                         .5, //initVal of weights
                         "" //filename, not used
                         );

         testLayer = new BaseLayer();
         testLayer->setParams(myCol, //column
                              "test"); //name

         myCol->addLayer(input);
         myCol->addConn(fc);
         myCol->addLayer(testLayer);
      }
      virtual void TearDown(){
         delete myCol;
         delete input;
         delete testLayer;
         delete fc;
      }

      Column* myCol;
      Image* input;
      BaseLayer* testLayer;
      FullyConnected* fc;
};

TEST_F(fcTests, sizeSetTest){
   myCol->initialize();
   EXPECT_EQ(testLayer->getYSize(), 1);
   EXPECT_EQ(testLayer->getXSize(), 1);
   EXPECT_EQ(testLayer->getFSize(), 5);
}

TEST_F(fcTests, feedforward){
   myCol->initialize();
   myCol->run(1);
   float* h_outData = testLayer->getHostA();
   int idx = 0;
   for(int fi = 0; fi < 5; fi++){
      ASSERT_EQ(h_outData[idx], 2.0);
   }
   free(h_outData);
}
