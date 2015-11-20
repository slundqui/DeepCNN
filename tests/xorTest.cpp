#include <limits.h>
#include "gtest/gtest.h"
#include <utils.hpp>
#include <layers/MatInput.hpp>
#include <layers/LeastSquaresCost.hpp>
#include <layers/Activation.hpp>
#include <connections/FullyConnected.hpp>
#include <cmath>


//Fixture for testing auto size generation
class xorTests: public ::testing::Test{
   protected:
      virtual void SetUp(){
         //Simple 2 layer network that has 2 inputs, 2 hidden units, and 1 output
         batch = 4;
         
         myCol = new Column(batch //batch
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

      void gradientCheck(Convolution* checkConn, float epsilon, float* h_actualWGrad, float* h_actualBGrad){
         float tolerance = 1e-3;

         int numWeights = checkConn->getNumWeights();
         int numBias = checkConn->getNumBiases();
         float* baseWeights = checkConn->getHostW();
         float* baseBias = checkConn->getHostB();
         int batch = input->getBSize();

         //std::cout << "numWeights " << numWeights << "numBiases" << numBias << "\n";

         //Check weights
         for(int weightIdx = 0; weightIdx < numWeights; weightIdx++){
            //Rewind matInput layers
            input->rewind();
            gt->rewind();
            //Set weight + epsilon
            checkConn->setWeight(weightIdx, baseWeights[weightIdx] + epsilon);
            //Run network for 1 timestep
            myCol->run(1);

            //float* h_data = hidden->getHostA();
            //std::cout << "Pos run\n";
            //printMat(h_data, 1, 2, 1, 1);

            const float* h_cost = cost->getHostTotalCost();
            float posCost = 0;
            for(int b = 0; b < batch; b++){
               posCost += h_cost[b];
            }

            //Grab neg cost
            input->rewind();
            gt->rewind();
            //Set weight - epsilon
            checkConn->setWeight(weightIdx, baseWeights[weightIdx] - epsilon);
            //Run network for 1 timestep
            myCol->run(1);

            //h_data = hidden->getHostA();
            //std::cout << "Neg run\n";
            //printMat(h_data, 1, 2, 1, 1);

            //Grab neg cost
            h_cost = cost->getHostTotalCost();
            float negCost = 0;
            for(int b = 0; b < batch; b++){
               negCost += h_cost[b];
            }
            //std::cout << "posCost " << posCost << " negCost " << negCost << "\n";
            
            float empGrad = (posCost - negCost)/(2*epsilon);
            float actGrad = h_actualWGrad[weightIdx];
            ASSERT_TRUE(fabs(empGrad - actGrad) < tolerance);
            //std::cout << "Weight Idx: " << weightIdx << " EmpGrad: " << empGrad << " ActGrad: " << actGrad << "\n";

            //Reset weight
            checkConn->setWeight(weightIdx, baseWeights[weightIdx]);
         }

         //Check bias
         for(int biasIdx = 0; biasIdx < numBias; biasIdx++){
            //Rewind matInput layers
            input->rewind();
            gt->rewind();
            //Set weight + epsilon
            checkConn->setBias(biasIdx, baseBias[biasIdx] + epsilon);
            //Run network for 1 timestep
            myCol->run(1);

            const float* h_cost = cost->getHostTotalCost();
            float posCost = 0;
            for(int b = 0; b < batch; b++){
               posCost += h_cost[b];
            }

            //Grab neg cost
            input->rewind();
            gt->rewind();
            //Set weight - epsilon
            checkConn->setBias(biasIdx, baseBias[biasIdx] - epsilon);
            //Run network for 1 timestep
            myCol->run(1);
            //Grab neg cost
            h_cost = cost->getHostTotalCost();
            float negCost = 0;
            for(int b = 0; b < batch; b++){
               negCost += h_cost[b];
            }
            
            //std::cout << "posCost " << posCost << " negCost " << negCost << "\n";
            float empGrad = (posCost - negCost)/(2*epsilon);
            float actGrad = h_actualBGrad[biasIdx];

            ASSERT_TRUE(fabs(empGrad - actGrad) < tolerance);
            //std::cout << "Bias Idx: " << biasIdx << " EmpGrad: " << empGrad << " ActGrad: " << actGrad << "\n";

            //Reset weight
            checkConn->setBias(biasIdx, baseBias[biasIdx]);
         }



      }

      Column* myCol;
      MatInput* input;
      MatInput* gt;
      Activation* hidden;
      LeastSquaresCost* cost;
      FullyConnected* fc1;
      FullyConnected* fc2;
      int batch;
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
   //Do not update weights but calculate gradients
   fc1->setGradientCheck();
   fc2->setGradientCheck();

   myCol->initialize();

   //Rewind matInput layers
   input->rewind();
   gt->rewind();

   myCol->run(1);
   //Grab actual gradients calculated
   float* h_fc1_weight_grad = fc1->getHostWGradient();
   float* h_fc1_bias_grad = fc1->getHostBGradient();
   float* h_fc2_weight_grad = fc2->getHostWGradient();
   float* h_fc2_bias_grad = fc2->getHostBGradient();

   ////Check fc2 gradients
   gradientCheck(fc2, epsilon, h_fc2_weight_grad, h_fc2_bias_grad);
   //Check fc1 gradients
   gradientCheck(fc1, epsilon, h_fc1_weight_grad, h_fc1_bias_grad);

}

