#include <limits.h>
#include "gtest/gtest.h"
#include <utils.hpp>
//#include <layers/MatInput.hpp>
#include <layers/LeastSquaresCost.hpp>
#include <layers/Activation.hpp>
#include <connections/FullyConnected.hpp>
#include "test_utils.hpp"


//Fixture for testing auto size generation
class xorTests: public ::testing::Test{
   protected:
      virtual void SetUp(){
         //Simple 2 layer network that has 2 inputs, 2 hidden units, and 1 output
         batch = 4;
         
         myCol = new Column(batch, //batch
                            12948589//seed1
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
                               "/home/sheng/workspace/DeepCNN/tests/testMats/xorGT_zeros.mat");//list of images

         fc1 = new FullyConnected();
         fc1->setParams(myCol, //column
                         "fc1", //name
                         2, //nfp
                         1, //uniform random weights
                         1, //range of weights
                         "", //filename, not used
                         0, //uniform init of bias
                         0, //initVal of bias
                         "", //filename, not used
                         1, //Plasticity is on
                         1, //dw rate
                         2, //db rate
                         0, //dw momentum
                         0, //db momentum
                         0 //decay
                         );

         hidden = new Activation();
         hidden->setParams(myCol,
                           "hidden",
                           "sigmoid");

         fc2 = new FullyConnected();
         fc2->setParams(myCol, //column
                         "fc2", //name
                         1, //nfp
                         1, //uniform random weights
                         1, //range of weights
                         "", //filename, not used
                         0, //uniform init of bias
                         0, //initVal of bias
                         "", //filename, not used
                         1, //Plasticity is on
                         1, //dw rate
                         2, //db rate
                         0, //dw momentum
                         0, //db momentum
                         0 //decay
                         );

         cost = new LeastSquaresCost();
         cost->setParams(myCol,
                         "cost",
                         "sigmoid");

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
      int batch;
};

//TEST_F(xorTests, forwardPass){
//   //Do not update weights but calculate gradients
//   fc1->setGradientCheck();
//   fc2->setGradientCheck();
//
//   myCol->initialize();
//   myCol->run(1);
//
//   //float* h_inData = input->getHostA();
//   //printMat(h_inData, 4, 1, 1, 2);
//   //free(h_inData);
//
//   //h_inData = gt->getHostA();
//   //std::cout << "GT data : \n";
//   //printMat(h_inData, 4, 1, 1, 1);
//   //free(h_inData);
//
//   float* h_inData = hidden->getHostU();
//   for(int i = 0; i < 8; i++){
//      int ib = i / 2;
//      if(ib == 0){
//       ASSERT_FLOAT_EQ(h_inData[i], -.2);
//      }
//      else if(ib == 1 || ib == 2){
//         ASSERT_FLOAT_EQ(h_inData[i], 0);
//      }
//      else if(ib == 3){
//         ASSERT_FLOAT_EQ(h_inData[i], .2); //      }
//   }
//   free(h_inData);
//
//   h_inData = hidden->getHostA();
//   for(int i = 0; i < 8; i++){
//      int ib = i / 2;
//      if(ib == 0){
//       ASSERT_FLOAT_EQ(h_inData[i], -.19737533);
//      }
//      else if(ib == 1 || ib == 2){
//         ASSERT_FLOAT_EQ(h_inData[i], 0);
//      }
//      else if(ib == 3){
//         ASSERT_FLOAT_EQ(h_inData[i], .19737533);
//      }
//   }
//   free(h_inData);
//
//   h_inData = cost->getHostA();
//   for(int ib = 0; ib < 4; ib++){
//      if(ib == 0){
//       ASSERT_FLOAT_EQ(h_inData[ib], -.039475065);
//      }
//      else if(ib == 1 || ib == 2){
//         ASSERT_FLOAT_EQ(h_inData[ib], 0);
//      }
//      else if(ib == 3){
//         ASSERT_FLOAT_EQ(h_inData[ib], .039475065);
//      }
//   }
//   free(h_inData);
//
//   const float* h_cost = cost->getHostTotalCost();
//   for(int ib = 0; ib < 4; ib++){
//      if(ib == 0){
//         ASSERT_FLOAT_EQ(h_cost[ib], 0.46130407);
//      }
//      else if(ib == 1 || ib == 2){
//         ASSERT_FLOAT_EQ(h_cost[ib], .5);
//      }
//      else if(ib == 3){
//         ASSERT_FLOAT_EQ(h_cost[ib], .54025424);
//      }
//   }
//}

//This test calculates gradients emperically and compares them with backprop gradients
TEST_F(xorTests, checkGradient){
   float tolerance = 10e-3;
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
   EXPECT_TRUE(gradientCheck(myCol, fc2, input, gt, cost, tolerance, h_fc2_weight_grad, h_fc2_bias_grad));
   //Check fc1 gradients
   EXPECT_TRUE(gradientCheck(myCol, fc1, input, gt, cost, tolerance, h_fc1_weight_grad, h_fc1_bias_grad));

}

//This test attempts to solve the xor problem
//This seed with parameters should work to find a solution
TEST_F(xorTests, xorLearn){
   myCol->initialize();

   //myCol->run(5000);

   bool verbose = false;

   myCol->run(500);

   float* h_est = cost->getHostA();
   float* h_truth = gt->getHostA();

   for(int b = 0; b < 4/batch; b++){
      for(int i = 0; i < batch; i++){
         float t_est = h_est[i] < .5 ? 0 : 1;
         EXPECT_EQ(t_est, h_truth[i]);
      }
      myCol->run(1);
   }
   
   //for(int i = 0; i < 4; i++){
   //   std::cout << "---------------\ninput\n";
   //   input->printA();
   //   if(verbose){
   //      std::cout << "---------------\nhidden U\n";
   //      hidden->printU();
   //      std::cout << "---------------\nhidden A\n";
   //      hidden->printA();
   //   }
   //   std::cout << "---------------\nEST U\n";
   //   cost->printU();
   //   std::cout << "---------------\nEST A\n";
   //   cost->printA();
   //   std::cout << "---------------\nGT\n";
   //   gt->printA();
   //   if(verbose){
   //      std::cout << "---------------\nEST A gradient\n";
   //      cost->printGA();
   //      std::cout << "---------------\nEST U gradient\n";
   //      cost->printGU();
   //      std::cout << "---------------\nfc2 w gradient\n";
   //      fc2->printGW();
   //      std::cout << "---------------\nfc2 b gradient\n";
   //      fc2->printGB();
   //      std::cout << "---------------\nfc2 weights\n";
   //      fc2->printW();
   //      std::cout << "---------------\nfc2 bias\n";
   //      fc2->printB();
   //      std::cout << "---------------\nhidden A gradient\n";
   //      hidden->printGA();
   //      std::cout << "---------------\nhidden U gradient\n";
   //      hidden->printGU();
   //      std::cout << "---------------\nfc1 w gradient\n";
   //      fc1->printGW();
   //      std::cout << "---------------\nfc1 b gradient\n";
   //      fc1->printGB();
   //      std::cout << "---------------\nfc1 weights\n";
   //      fc1->printW();
   //      std::cout << "---------------\nfc1 bias\n";
   //      fc1->printB();
   //   }
   //   myCol->run(1);
   //}

   //float* h_est= cost->getHostA();
   //float* h_gt= gt->getHostA();
   //
   //float tolerance = 1e-5;
   //for(int i = 0; i < batch; i++){
   //   float h_thresh_est = h_est[i] < .5 ? 0 : 1;
   //   ASSERT_TRUE(fabs(h_gt[i]-h_thresh_est) < tolerance);
   //}
}

