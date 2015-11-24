#include <limits.h>
#include "gtest/gtest.h"
#include <utils.hpp>
//#include <layers/MatInput.hpp>
#include <layers/SoftmaxCost.hpp>
#include <layers/LeastSquaresCost.hpp>
#include <layers/Activation.hpp>
#include <connections/FullyConnected.hpp>
#include <cmath>
#include "test_utils.hpp"


//Fixture for testing auto size generation
class cifarTests: public ::testing::Test{
   protected:
      virtual void SetUp(){
         //Simple 3 layer network, no pooling
         batch = 4;
         
         myCol = new Column(batch, //batch
                            1234567890//seed
                            );

         input= new MatInput();
         input->setParams(myCol, //column name
                               "input", //name
                               32, //ny
                               32, //nx
                               3, //features
                               "/home/sheng/workspace/DeepCNNData/cifar/formatted/trainData.mat");//list of images

         gt = new MatInput();
         gt->setParams(myCol, //column name
                               "gt", //name
                               1, //ny
                               1, //nx
                               10, //features
                               "/home/sheng/workspace/DeepCNNData/cifar/formatted/trainLabels.mat");//list of images

         conv1 = new Convolution();
         conv1->setParams(myCol, //column
                         "conv1", //name
                         3, //nyp
                         3, //nxp
                         10, //nfp
                         2, //ystride
                         2, //xstride
                         1, //uniform random weights
                         .01, //range of weights
                         "", //filename, not used
                         0, //uniform init of bias
                         0, //initVal of bias
                         "", //filename, not used
                         1, //Plasticity is on
                         .01, //dw rate
                         .02, //db rate
                         0.9, //dw momentum
                         0.9, //db momentum
                         0.004 //decay
                         );

         hidden1 = new Activation();
         hidden1->setParams(myCol,
                           "hidden1",
                           "relu");

         conv2 = new Convolution();
         conv2->setParams(myCol, //column
                         "conv2", //name
                         3, //nyp
                         3, //nxp
                         10, //nfp
                         2, //ystride
                         2, //xstride
                         1, //uniform random weights
                         .01, //range of weights
                         "", //filename, not used
                         0, //uniform init of bias
                         0, //initVal of bias
                         "", //filename, not used
                         1, //Plasticity is on
                         .01, //dw rate
                         .02, //db rate
                         0.9, //dw momentum
                         0.9, //db momentum
                         0.004 //decay
                         );

         hidden2 = new Activation();
         hidden2->setParams(myCol,
                           "hidden2",
                           "relu");

         fc = new FullyConnected();
         fc->setParams(myCol, //column
                         "fc", //name
                         10, //nfp
                         1, //uniform random weights
                         .01, //range of weights
                         "", //filename, not used
                         0, //uniform init of bias
                         0, //initVal of bias
                         "", //filename, not used
                         1, //Plasticity is on
                         .1, //dw rate
                         .2, //db rate
                         0.9, //dw momentum
                         0.9, //db momentum
                         0.03 //decay
                         );

         //cost = new LeastSquaresCost();
         //cost->setParams(myCol,
         //                "cost",
         //                "sigmoid");

         cost = new SoftmaxCost();
         cost->setParams(myCol,
                         "cost");

         myCol->addLayer(input);
         myCol->addConn(conv1);
         myCol->addLayer(hidden1);
         myCol->addConn(conv2);
         myCol->addLayer(hidden2);
         myCol->addConn(fc);
         myCol->addLayer(cost);
         myCol->addGroundTruth(gt);
      }
      virtual void TearDown(){
         delete myCol;
         delete input;
         delete gt;
         delete conv1;
         delete hidden1;
         delete conv2;
         delete hidden2;
         delete fc;
         delete cost;
      }

      Column* myCol;
      MatInput* input;
      MatInput* gt;
      Activation* hidden1;
      Activation* hidden2;
      SoftmaxCost* cost;
      //LeastSquaresCost* cost;
      Convolution* conv1;
      Convolution* conv2;
      FullyConnected* fc;
      int batch;
};

///This test calculates gradients emperically and compares them with backprop gradients
TEST_F(cifarTests, checkGradient){
   float tolerance = 10e-3;

   //Do not update weights but calculate gradients
   conv1->setGradientCheck();
   conv2->setGradientCheck();
   fc->setGradientCheck();

   myCol->initialize();

   //Rewind matInput layers
   input->rewind();
   gt->rewind();

   myCol->run(1);
   //Grab actual gradients calculated
   float* h_conv1_weight_grad = conv1->getHostWGradient();
   float* h_conv1_bias_grad = conv1->getHostBGradient();
   float* h_conv2_weight_grad = conv2->getHostWGradient();
   float* h_conv2_bias_grad = conv2->getHostBGradient();
   float* h_fc_weight_grad = fc->getHostWGradient();
   float* h_fc_bias_grad = fc->getHostBGradient();

   //cost->printDim();

   //input->printDims();
   //conv2->printDims();
   //hidden2->printDims();
   //fc->printDims();
   //cost->printDims();

   //std::cout << "---------------\nEST U\n";
   //cost->printU();
   //std::cout << "---------------\nEST A\n";
   //cost->printA();
   //std::cout << "---------------\nGT\n";
   //gt->printA();
   //std::cout << "---------------\nEST A gradient\n";
   //cost->printGA();
   //std::cout << "---------------\nEST U gradient\n";
   //cost->printGU();


   //Check gradients
   EXPECT_TRUE(gradientCheck(myCol, fc, input, gt, cost, tolerance, h_fc_weight_grad, h_fc_bias_grad));
   EXPECT_TRUE(gradientCheck(myCol, conv2, input, gt, cost, tolerance, h_conv2_weight_grad, h_conv2_bias_grad));
   EXPECT_TRUE(gradientCheck(myCol, conv1, input, gt, cost, tolerance, h_conv1_weight_grad, h_conv1_bias_grad));


   //std::cout << "input vals: \n";
   //input->printA();

   //std::cout << "conv1 weights: \n";
   //conv1->printW();

   //std::cout << "hidden 1 vals: \n";
   //hidden1->printA();

   //std::cout << "hidden 2 vals: \n";
   //hidden2->printA();

   //std::cout << "est U vals: \n";
   //cost->printU();

   //std::cout << "est A vals: \n"; 
   //cost->printA();
}

//TEST_F(cifarTests, cifarLearn){
//   myCol->initialize();
//
//   int outerRunTime = 50;
//   int innerRunTime = 1000;
//
//   for(int i = 0; i < outerRunTime; i++){
//      myCol->run(innerRunTime);
//      //Get accuracy
//      float accuracy = cost->getHostAccuracy();
//      std::cout << "Run " << i*innerRunTime << " out of " << outerRunTime * innerRunTime << " accuracy: " << accuracy << "\n";
//      
//      //reset accuracy
//      cost->reset();
//   }
//
//   //float* h_data;
//   //std::cout << "---------------\ninput\n";
//   //h_data= input->getHostA();
//   //printMat(h_data, batch, 2, 1, 1);
//   //free(h_data);
//
//   //std::cout << "---------------\nhidden U\n";
//   //h_data= hidden->getHostU();
//   //printMat(h_data, batch, 2, 1, 1);
//   //free(h_data);
//
//   //std::cout << "---------------\nhidden A\n";
//   //h_data= hidden->getHostA();
//   //printMat(h_data, batch, 2, 1, 1);
//   //free(h_data);
//
//   //std::cout << "---------------\nEST\n";
//   //h_data= cost->getHostA();
//   //printMat(h_data, batch, 1, 1, 1);
//   //free(h_data);
//
//   //std::cout << "---------------\nGT\n";
//   //h_data = gt->getHostA();
//   //printMat(h_data, batch, 1, 1, 1);
//   //free(h_data);
//
//   //std::cout << "---------------\nEST gradient\n";
//   //h_data= cost->getHostG();
//   //printMat(h_data, batch, 1, 1, 1);
//   //free(h_data);
//
//   //std::cout << "---------------\nfc2 w gradient\n";
//   //h_data= fc2->getHostWGradient();
//   //printMat(h_data, 1, 2, 1,1 );
//   //free(h_data);
//
//   //std::cout << "---------------\nfc2 b gradient\n";
//   //h_data= fc2->getHostBGradient();
//   //printMat(h_data, 1, 1, 1,1 );
//   //free(h_data);
//
//   //std::cout << "---------------\nfc2 weights\n";
//   //h_data= fc2->getHostW();
//   //printMat(h_data, 1, 2, 1,1 );
//   //free(h_data);
//
//   //std::cout << "---------------\nfc2 bias\n";
//   //h_data= fc2->getHostB();
//   //printMat(h_data, 1, 1, 1,1 );
//   //free(h_data);
//
//   //std::cout << "---------------\nhidden G\n";
//   //h_data= hidden->getHostG();
//   //printMat(h_data, batch, 2, 1, 1);
//   //free(h_data);
//
//   //std::cout << "---------------\nfc1 w gradient\n";
//   //h_data= fc1->getHostWGradient();
//   //printMat(h_data, 2, 2, 1,1 );
//   //free(h_data);
//
//   //std::cout << "---------------\nfc1 b gradient\n";
//   //h_data= fc1->getHostBGradient();
//   //printMat(h_data, 1, 2, 1,1 );
//   //free(h_data);
//
//   //std::cout << "---------------\nfc1 weights\n";
//   //h_data= fc1->getHostW();
//   //printMat(h_data, 2, 2, 1,1 );
//   //free(h_data);
//
//   //std::cout << "---------------\nfc1 bias\n";
//   //h_data= fc1->getHostB();
//   //printMat(h_data, 1, 2, 1,1 );
//   //free(h_data);
//
//}
