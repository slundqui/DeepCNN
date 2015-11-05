#include <limits.h>
#include "gtest/gtest.h"
#include <iostream>
#include <src/Column.hpp>
#include <src/layers/Image.hpp>

//Fixture for testing baseLayer class
class ImageTests: public ::testing::Test {
   protected:
      virtual void SetUp(){
         //::testing::FLAGS_gtest_death_test_style = "threadsafe"; //To suppress test warning
         singleCol = new Column(1, //batch
                                5, //ny
                                4);//nx

         batchCol = new Column(4,  //batch
                               5,  //ny
                               4); //nx

         imageLayerSingle = new Image();
         imageLayerSingle->setParams(singleCol,
                               "input",
                               3, //features
                               "/home/sheng/workspace/DeepCNN/tests/testImgs/fileList.txt");//list of images

         imageLayerBatch = new Image();
         imageLayerBatch->setParams(batchCol,
                               "input",
                               3, //features
                               "/home/sheng/workspace/DeepCNN/tests/testImgs/fileList.txt");//list of images

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
  singleCol->run(1); //Run for 1 timestep, image 0
  //Grab and copy device activity
  float * h_imgData = imageLayerSingle->getDeviceA();
  for(int i = 0; i < 5*4*3; i++){
     ASSERT_EQ(h_imgData[i], i);
  }
  free(h_imgData);

  singleCol->run(1); //image 1
  h_imgData = imageLayerSingle->getDeviceA();
  int offset = 60;
  for(int i = 0; i < 5*4*3; i++){
     ASSERT_EQ(h_imgData[i], offset + i);
  }
  free(h_imgData);

  singleCol->run(1); //image 2
  h_imgData = imageLayerSingle->getDeviceA();
  offset = 120;
  for(int i = 0; i < 5*4*3; i++){
     ASSERT_EQ(h_imgData[i], offset + i);
  }
  free(h_imgData);
}

TEST_F(ImageTests, batchTest){
  batchCol->initialize();
  batchCol->run(1); //Run for 1 timestep, image 0
  //Grab and copy device activity
  float * h_imgData = imageLayerBatch->getDeviceA();
  for(int i = 0; i < 4*5*4*3; i++){
     //Last batch is a repeat of the first batch
     if(i < 3*5*4*3){
        ASSERT_EQ(h_imgData[i], i);
     }
     else{
        ASSERT_EQ(h_imgData[i], i-180);
     }
  }
  free(h_imgData);
}



