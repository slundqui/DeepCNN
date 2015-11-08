#include <limits.h>
#include "gtest/gtest.h"
#include <iostream>
#include <src/Column.hpp>
#include <src/layers/Image.hpp>
#include <connections/Convolution.hpp>

//Fixture for testing auto size generation
class SizeTests: public ::testing::Test{
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
                         2); //xstride

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

TEST_F(SizeTests, sizeSetTest){
   myCol->initialize();
   EXPECT_EQ(testLayer->getYSize(), 8);
   EXPECT_EQ(testLayer->getXSize(), 4);
   EXPECT_EQ(testLayer->getFSize(), 5);
}
