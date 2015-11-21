#include <limits.h>
#include "gtest/gtest.h"
#include <utils.hpp>
#include <layers/MatInput.hpp>
#include <layers/LeastSquaresCost.hpp>
#include <layers/Activation.hpp>
#include <connections/FullyConnected.hpp>
#include <cmath>


//Fixture for testing auto size generation
class nandTests: public ::testing::Test{
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
                               "/home/sheng/workspace/DeepCNN/tests/testMats/binaryInput.mat");//list of images

         gt = new MatInput();
         gt->setParams(myCol, //column name
                               "gt", //name
                               1, //ny
                               1, //nx
                               1, //features
                               "/home/sheng/workspace/DeepCNN/tests/testMats/nandGT.mat");//list of images

         fc = new FullyConnected();
         fc->setParams(myCol, //column
                         "fc", //name
                         1, //nfp
                         0, //uniform init of weights
                         .1, //initVal of weights
                         "", //filename, not used
                         0, //uniform init of bias
                         0, //initVal of bias
                         "", //filename, not used
                         1, //Plasticity is on
                         .1, //dw rate
                         .1, //db rate
                         0, //dw momentum
                         0, //db momentum
                         0 //decay
                         );

         cost = new LeastSquaresCost();
         cost->setParams(myCol,
                         "cost");


         myCol->addLayer(input);
         myCol->addConn(fc);
         myCol->addLayer(cost);
         myCol->addGroundTruth(gt);
      }
      virtual void TearDown(){
         delete myCol;
         delete input;
         delete gt;
         delete fc;
         delete cost;
      }

      void gradientCheck(Convolution* checkConn, float epsilon, float* h_actualWGrad, float* h_actualBGrad){
         float tolerance = 1e-2;

         int numWeights = checkConn->getNumWeights();
         int numBias = checkConn->getNumBias();
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
            
            float empGrad = -(posCost - negCost)/(2*epsilon);
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
            float empGrad = -(posCost - negCost)/(2*epsilon);
            float actGrad = h_actualBGrad[biasIdx];

            ASSERT_TRUE(fabs(empGrad - actGrad) < tolerance);
            //std::cout << "Bias Idx: " << biasIdx << " EmpGrad: " << empGrad << " ActGrad: " << actGrad << "\n";

            //Reset weight
            checkConn->setBias(biasIdx, baseBias[biasIdx]);
         }

         free(baseWeights);
         free(baseBias);
      }

      Column* myCol;
      MatInput* input;
      MatInput* gt;
      Activation* hidden;
      LeastSquaresCost* cost;
      FullyConnected* fc;
      int batch;
};

//This test calculates gradients emperically and compares them with backprop gradients
TEST_F(nandTests, gradientCheck){
   float epsilon = 10e-5;
   //Do not update weights but calculate gradients
   fc->setGradientCheck();

   myCol->initialize();

   //Rewind matInput layers
   input->rewind();
   gt->rewind();

   myCol->run(1);

   //Grab actual gradients calculated float* h_fc_weight_grad = fc->getHostWGradient();
   float* h_fc_weight_grad = fc->getHostWGradient();
   float* h_fc_bias_grad = fc->getHostBGradient();

   ////Check fc gradients
   gradientCheck(fc, epsilon, h_fc_weight_grad, h_fc_bias_grad);

}

////This test attempts to solve the xor problem
//TEST_F(nandTests, nandLearn){
//   myCol->initialize();
//   //Rewind matInput layers
//   input->rewind();
//   gt->rewind();
//
//   myCol->run(20);
//   //for(int i = 0; i < 4; i++){
//      myCol->run(1);
//      std::cout << "input\n";
//      float* h_input = input->getHostA();
//      printMat(h_input, batch, 2, 1, 1);
//      free(h_input);
//
//      std::cout << "fc weights\n";
//      float* h_weight = fc->getHostW();
//      printMat(h_weight, 1, 2, 1, 1 );
//      free(h_weight);
//
//      std::cout << "fc bias\n";
//      h_weight = fc->getHostB();
//      printMat(h_weight, 1, 1, 1, 1 );
//      free(h_weight);
//
//      std::cout << "EST\n";
//      float* h_est = cost->getHostA();
//      printMat(h_est, batch, 1, 1, 1);
//      free(h_est);
//
//      std::cout << "GT\n";
//      float* h_gt = gt->getHostA();
//      printMat(h_gt, batch, 1, 1, 1);
//      free(h_gt);
//   //}
//
//}




