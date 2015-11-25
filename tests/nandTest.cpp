#include <limits.h>
#include "gtest/gtest.h"
#include <utils.hpp>
//#include <layers/MatInput.hpp>
#include <layers/LeastSquaresCost.hpp>
#include <layers/Activation.hpp>
#include <connections/FullyConnected.hpp>
#include "test_utils.hpp"


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
                         1, //uniform init of weights
                         .01, //initVal of weights
                         "", //filename, not used
                         1, //uniform init of bias
                         .01, //initVal of bias
                         "", //filename, not used
                         1, //Plasticity is on
                         .01, //dw rate
                         .01, //db rate
                         0, //dw momentum
                         0, //db momentum
                         0 //decay
                         );

         cost = new LeastSquaresCost();
         cost->setParams(myCol,
                         "cost",
                         "sigmoid");


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

      Column* myCol;
      MatInput* input;
      MatInput* gt;
      Activation* hidden;
      LeastSquaresCost* cost;
      FullyConnected* fc;
      int batch;
};

//This test calculates gradients emperically and compares them with backprop gradients
TEST_F(nandTests, nandCheckGradient){
   float tolerance = 10e-3;
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
   EXPECT_TRUE(gradientCheck(myCol, fc, input, gt, cost, tolerance, h_fc_weight_grad, h_fc_bias_grad));

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




