#include <limits.h>
#include "gtest/gtest.h"
#include <utils.hpp>
#include <layers/MatInput.hpp>
#include <layers/LeastSquaresCost.hpp>
#include <layers/Activation.hpp>
#include <connections/FullyConnected.hpp>


//Fixture for testing auto size generation
class xorTests: public ::testing::Test{
   protected:
      virtual void SetUp(){
         //Simple 2 layer network that has 2 inputs, 2 hidden units, and 1 output
         
         myCol = new Column(4 //batch
                            );

         input= new MatInput();
         input->setParams(myCol, //column name
                               "input", //name
                               1, //ny
                               1, //nx
                               2, //features
                               "/home/sheng/workspace/DeepCNN/tests/testMats/xorInput.mat");//list of images

         gt = new MatInput();
         gt->setParams(myCol, //column name
                               "gt", //name
                               1, //ny
                               1, //nx
                               1, //features
                               "/home/sheng/workspace/DeepCNN/tests/testMats/xorGT.mat");//list of images

         fc1 = new FullyConnected();
         fc1->setParams(myCol, //column
                         "fc1", //name
                         2, //nfp
                         0, //uniform init of weights
                         .1, //initVal of weights
                         "" //filename, not used
                         );

         hidden = new Activation();
         hidden->setParams(myCol,
                           "hidden",
                           "tanh"); //tanh for -1 to 1 range

         fc2 = new FullyConnected();
         fc2->setParams(myCol, //column
                         "fc2", //name
                         1, //nfp
                         0, //uniform init of weights
                         .1, //initVal of weights
                         "" //filename, not used
                         );

         cost = new LeastSquaresCost();
         cost->setParams(myCol,
                         "cost");


         myCol->addLayer(input);
         myCol->addConn(fc1);
         myCol->addLayer(hidden);
         myCol->addConn(fc2);
         myCol->addLayer(cost);
         myCol->addGroundTruth(gt);
      }
      virtual void TearDown(){
         delete myCol;
         delete input;
         delete gt;
         delete fc1;
         delete fc2;
         delete cost;
      }

      Column* myCol;
      MatInput* input;
      MatInput* gt;
      Activation* hidden;
      LeastSquaresCost* cost;
      FullyConnected* fc1;
      FullyConnected* fc2;
};

TEST_F(xorTests, forwardPass){
   myCol->initialize();
   myCol->run(1);

   //float* h_inData = input->getHostA();
   //printMat(h_inData, 4, 1, 1, 2);
   //free(h_inData);

   //h_inData = gt->getHostA();
   //std::cout << "GT data : \n";
   //printMat(h_inData, 4, 1, 1, 1);
   //free(h_inData);

   float* h_inData = hidden->getHostU();
   for(int i = 0; i < 8; i++){
      int ib = i / 2;
      if(ib == 0){
         ASSERT_FLOAT_EQ(h_inData[i], -.2);
      }
      else if(ib == 1 || ib == 2){
         ASSERT_FLOAT_EQ(h_inData[i], 0);
      }
      else if(ib == 3){
         ASSERT_FLOAT_EQ(h_inData[i], .2);
      }
   }
   free(h_inData);

   h_inData = hidden->getHostA();
   for(int i = 0; i < 8; i++){
      int ib = i / 2;
      if(ib == 0){
         ASSERT_FLOAT_EQ(h_inData[i], -.19737533);
      }
      else if(ib == 1 || ib == 2){
         ASSERT_FLOAT_EQ(h_inData[i], 0);
      }
      else if(ib == 3){
         ASSERT_FLOAT_EQ(h_inData[i], .19737533);
      }
   }
   free(h_inData);

   h_inData = cost->getHostA();
   for(int ib = 0; ib < 4; ib++){
      if(ib == 0){
         ASSERT_FLOAT_EQ(h_inData[ib], -.039475065);
      }
      else if(ib == 1 || ib == 2){
         ASSERT_FLOAT_EQ(h_inData[ib], 0);
      }
      else if(ib == 3){
         ASSERT_FLOAT_EQ(h_inData[ib], .039475065);
      }
   }
   free(h_inData);

   const float* h_cost = cost->getHostTotalCost();
   for(int ib = 0; ib < 4; ib++){
      if(ib == 0){
         ASSERT_FLOAT_EQ(h_cost[ib], 0.46130407);
      }
      else if(ib == 1 || ib == 2){
         ASSERT_FLOAT_EQ(h_cost[ib], .5);
      }
      else if(ib == 3){
         ASSERT_FLOAT_EQ(h_cost[ib], .54025424);
      }
   }
}

//This test calculates gradients emperically and compares them with backprop gradients
TEST_F(xorTests, gradientCheck){
   float epsilon = 10e-5;
   //Do not update weights
   fc1->setGradientCheck();
   fc2->setGradientCheck();

   myCol->initialize();
   myCol->run(1);
   //Only checking batch 1
   const float* h_cost = cost->getHostTotalCost();
   
   //Get base cost and weights
   float baseCost = h_cost[0];
   float* h_fc1_base_weights = fc1->getHostW();
   float* h_fc1_base_bias = fc1->getHostB();
   float* h_fc2_base_weights = fc2->getHostW();
   float* h_fc2_base_bias = fc2->getHostB();

   float* h_fc1_weight_grad = fc1->getHostWGradient();
   float* h_fc1_bias_grad = fc1->getHostBGradient();
   float* h_fc2_weight_grad = fc2->getHostWGradient();
   float* h_fc2_bias_grad = fc2->getHostBGradient();

   //Get number of weights in each layer
   int fc1_numWeights = 4;
   int fc1_numBias = 2;
   int fc2_numWeights = 2;
   int fc2_numBias = 1;

   //Check fc2 gradients
   for(int weightIdx = 0; weightIdx < fc2_numWeights; weightIdx++){
      //Set weight + epsilon
      fc2->setWeight(weightIdx, h_fc2_base_weights[weightIdx] + epsilon);
      //Run network for 1 timestep
      myCol->run(1);
      //Grab new cost
      h_cost = cost->getHostTotalCost();
      float epCost = h_cost[0];
      float empGrad = (epCost - baseCost)/epsilon;
      float actGrad = h_fc2_weight_grad[weightIdx];
      std::cout << "Idx: " << weightIdx << " EmpGrad: " << empGrad << " ActGrad: " << actGrad << "\n";

      //Reset weight
      fc2->setWeight(weightIdx, h_fc2_base_weights[weightIdx]);
   }





}

