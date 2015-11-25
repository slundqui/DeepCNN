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
                            29383058//seed1
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
                         .01, //range of weights
                         "", //filename, not used
                         1, //uniform init of bias
                         .01, //initVal of bias
                         "", //filename, not used
                         1, //Plasticity is on
                         .01, //dw rate
                         .02, //db rate
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
                         .01, //range of weights
                         "", //filename, not used
                         1, //uniform init of bias
                         .01, //initVal of bias
                         "", //filename, not used
                         1, //Plasticity is on
                         .01, //dw rate
                         .02, //db rate
                         0, //dw momentum
                         0, //db momentum
                         0 //decay
                         );

         cost = new LeastSquaresCost();
         cost->setParams(myCol,
                         "cost",
                         "sigmoid"
                         //1, //writePeriod
                         //"/home/sheng/workspace/DeepCNNData/xor/totalCost.txt" //Out cost file
                         );

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

//This test calculates gradients emperically and compares them with backprop gradients
TEST_F(xorTests, xorCheckGradient){
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

////This test attempts to solve the xor problem
////This seed with parameters should work to find a solution
//TEST_F(xorTests, xorLearn){
//   myCol->initialize();
//
//   //myCol->run(5000);
//
//   bool verbose = true;
//
//   myCol->run(5000);
//
//   float* h_est = cost->getHostA();
//   float* h_truth = gt->getHostA();
//
//   for(int b = 0; b < 4/batch; b++){
//      for(int i = 0; i < batch; i++){
//         //std::cout << "---------------\ninput\n";
//         //input->printA();
//         //std::cout << "---------------\nGT\n";
//         //gt->printA();
//         //std::cout << "---------------\nEST\n";
//         //cost->printA();
//
//         float t_est = h_est[i] < .5 ? 0 : 1;
//         EXPECT_EQ(t_est, h_truth[i]);
//      }
//      myCol->run(1);
//   }
//   
//   //for(int i = 0; i < 1; i++){
//   //   std::cout << "---------------\ninput\n";
//   //   input->printA();
//   //   if(verbose){
//   //      std::cout << "---------------\nhidden U\n";
//   //      hidden->printU();
//   //      std::cout << "---------------\nhidden A\n";
//   //      hidden->printA();
//   //   }
//   //   std::cout << "---------------\nGT\n";
//   //   gt->printA();
//   //   std::cout << "---------------\nEST\n";
//   //   cost->printA();
//   //   if(verbose){
//   //      std::cout << "---------------\nEST A gradient\n";
//   //      cost->printGA();
//   //      std::cout << "---------------\nEST U gradient\n";
//   //      cost->printGU();
//   //      std::cout << "---------------\nfc2 w gradient\n";
//   //      fc2->printGW();
//   //      std::cout << "---------------\nfc2 b gradient\n";
//   //      fc2->printGB();
//   //      std::cout << "---------------\nfc2 weights\n";
//   //      fc2->printW();
//   //      std::cout << "---------------\nfc2 bias\n";
//   //      fc2->printB();
//   //      std::cout << "---------------\nhidden A gradient\n";
//   //      hidden->printGA();
//   //      std::cout << "---------------\nhidden U gradient\n";
//   //      hidden->printGU();
//   //      std::cout << "---------------\nfc1 w gradient\n";
//   //      fc1->printGW();
//   //      std::cout << "---------------\nfc1 b gradient\n";
//   //      fc1->printGB();
//   //      std::cout << "---------------\nfc1 weights\n";
//   //      fc1->printW();
//   //      std::cout << "---------------\nfc1 bias\n";
//   //      fc1->printB();
//   //   }
//   //   myCol->run(1);
//   //}
//
//   //float* h_est= cost->getHostA();
//   //float* h_gt= gt->getHostA();
//   //
//   //float tolerance = 1e-5;
//   //for(int i = 0; i < batch; i++){
//   //   float h_thresh_est = h_est[i] < .5 ? 0 : 1;
//   //   ASSERT_TRUE(fabs(h_gt[i]-h_thresh_est) < tolerance);
//   //}
//}

