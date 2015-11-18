#include <limits.h>
#include "gtest/gtest.h"
#include <iostream>
#include <src/layers/Image.hpp>

//Fixture for testing baseLayer class
class ImageTests: public ::testing::Test {
   protected:
      virtual void SetUp(){
         //::testing::FLAGS_gtest_death_test_style = "threadsafe"; //To suppress test warning
         singleCol = new Column(1 //batch
                                );

         batchCol = new Column(4  //batch
                               );

         imageLayerSingle = new Image();
         imageLayerSingle->setParams(singleCol,
                               "input",
                               5, //ny
                               4, //nx
                               3, //features
                               "/home/sheng/workspace/DeepCNN/tests/testImgs/idxcount_5x4x3.txt");//list of images

         imageLayerBatch = new Image();
         imageLayerBatch->setParams(batchCol,
                               "input",
                               5, //ny
                               4, //nx
                               3, //features
                               "/home/sheng/workspace/DeepCNN/tests/testImgs/idxcount_5x4x3.txt");//list of images

         //Add to columns
         singleCol->addLayer(imageLayerSingle);
         batchCol->addLayer(imageLayerBatch);
      }

      virtual void TearDown(){
         delete imageLayerSingle;
         delete imageLayerBatch;
         delete singleCol;
         delete batchCol;
      }
      Column* singleCol, * batchCol;
      Image *imageLayerSingle, *imageLayerBatch;
      float* h_check_memory;
};

TEST_F(ImageTests, singleTest){
  singleCol->initialize();
  singleCol->run(1); //image 1
  //Grab and copy device activity
  float * h_imgData = imageLayerSingle->getHostA();
  for(int i = 0; i < 5*4*3; i++){
     ASSERT_EQ(h_imgData[i], (float)i/255);
  }
  free(h_imgData);

  singleCol->run(1); //image 1
  h_imgData = imageLayerSingle->getHostA();
  int offset = 60;
  for(int i = 0; i < 5*4*3; i++){
     ASSERT_EQ(h_imgData[i], (float)(offset + i)/255);
  }
  free(h_imgData);

  singleCol->run(1); //image 2
  h_imgData = imageLayerSingle->getHostA();
  offset = 120;
  for(int i = 0; i < 5*4*3; i++){
     ASSERT_EQ(h_imgData[i], (float)(offset + i)/255);
  }
  free(h_imgData);
}

TEST_F(ImageTests, batchTest){
  batchCol->initialize(); //This should load the first image
  singleCol->run(1); //image 2
  //Grab and copy device activity
  float * h_imgData = imageLayerBatch->getHostA();
  for(int i = 0; i < 4*5*4*3; i++){
     //Last batch is a repeat of the first batch
     if(i < 3*5*4*3){
        ASSERT_EQ(h_imgData[i], (float)i / 255);
     }
     else{
        ASSERT_EQ(h_imgData[i], (float)(i-180)/255);
     }
  }
  free(h_imgData);
}



