//A run file that runs the DCN on cifar data

#include <Column.hpp>
#include <layers/MatInput.hpp>
#include <layers/SoftmaxCost.hpp>
#include <connections/FullyConnected.hpp>

int main(void){
   //Simple 3 layer network, no pooling
   int batch = 128;
   
   Column* myCol = new Column(batch, //batch
                      1234567890//seed
                      );

   MatInput* input= new MatInput();
   input->setParams(myCol, //column name
                         "input", //name
                         32, //ny
                         32, //nx
                         3, //features
                         "/home/sheng/workspace/DeepCNNData/cifar/formatted/trainData.mat");//list of images

   MatInput* gt = new MatInput();
   gt->setParams(myCol, //column name
                         "gt", //name
                         1, //ny
                         1, //nx
                         10, //features
                         "/home/sheng/workspace/DeepCNNData/cifar/formatted/trainLabels.mat");//list of images

   Convolution* conv1 = new Convolution();
   conv1->setParams(myCol, //column
                   "conv1", //name
                   5, //nyp
                   5, //nxp
                   32, //nfp
                   2, //ystride
                   2, //xstride
                   1, //uniform random weights
                   .0001, //range of weights
                   "", //filename, not used
                   0, //uniform init of bias
                   0, //initVal of bias
                   "", //filename, not used
                   1, //Plasticity is on
                   .001, //dw rate
                   .002, //db rate
                   0.9, //dw momentum
                   0.9, //db momentum
                   0.004 //decay
                   );

   Activation* hidden1 = new Activation();
   hidden1->setParams(myCol,
                     "hidden1",
                     "relu");

   Convolution* conv2 = new Convolution();
   conv2->setParams(myCol, //column
                   "conv2", //name
                   5, //nyp
                   5, //nxp
                   32, //nfp
                   2, //ystride
                   2, //xstride
                   1, //uniform random weights
                   .01, //range of weights
                   "", //filename, not used
                   0, //uniform init of bias
                   0, //initVal of bias
                   "", //filename, not used
                   1, //Plasticity is on
                   .001, //dw rate
                   .002, //db rate
                   0.9, //dw momentum
                   0.9, //db momentum
                   0.004 //decay
                   );

   Activation * hidden2 = new Activation();
   hidden2->setParams(myCol,
                     "hidden2",
                     "relu");

   Convolution* conv3 = new Convolution();
   conv3->setParams(myCol, //column
                   "conv3", //name
                   5, //nyp
                   5, //nxp
                   64, //nfp
                   2, //ystride
                   2, //xstride
                   1, //uniform random weights
                   .01, //range of weights
                   "", //filename, not used
                   0, //uniform init of bias
                   0, //initVal of bias
                   "", //filename, not used
                   1, //Plasticity is on
                   .001, //dw rate
                   .002, //db rate
                   0.9, //dw momentum
                   0.9, //db momentum
                   0.004 //decay
                   );

   Activation * hidden3 = new Activation();
   hidden3->setParams(myCol,
                     "hidden3",
                     "relu");

   FullyConnected* fc64 = new FullyConnected();
   fc64->setParams(myCol, //column
                   "fc64", //name
                   64, //nfp
                   1, //uniform random weights
                   .1, //range of weights
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

   Activation * hidden4 = new Activation();
   hidden4->setParams(myCol,
                     "hidden4",
                     "relu");
   
   FullyConnected* fc10 = new FullyConnected();
   fc10->setParams(myCol, //column
                   "fc10", //name
                   10, //nfp
                   1, //uniform random weights
                   .1, //range of weights
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

   SoftmaxCost* cost = new SoftmaxCost();
   cost->setParams(myCol,
                   "cost");

   myCol->addLayer(input);
   myCol->addConn(conv1);
   myCol->addLayer(hidden1);
   myCol->addConn(conv2);
   myCol->addLayer(hidden2);
   myCol->addConn(conv3);
   myCol->addLayer(hidden3);
   myCol->addConn(fc64);
   myCol->addLayer(hidden4);
   myCol->addConn(fc10);
   myCol->addLayer(cost);
   myCol->addGroundTruth(gt);


   //Run
   myCol->initialize();

   int outerRunTime = 5000;
   int innerRunTime = 1000;

   for(int i = 0; i < outerRunTime; i++){
      myCol->run(innerRunTime);
      //Get accuracy
      float accuracy = cost->getHostAccuracy();
      std::cout << "Run " << i*innerRunTime << " out of " << outerRunTime * innerRunTime << " accuracy: " << accuracy << "\n";
      
      //reset accuracy
      cost->reset();
   }



   
}
