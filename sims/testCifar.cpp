//A run file that runs the DCN on cifar data

#include <Column.hpp>
#include <layers/MatInput.hpp>
#include <layers/SoftmaxCost.hpp>
#include <layers/LeastSquaresCost.hpp>
#include <connections/FullyConnected.hpp>
#include <connections/MaxPool.hpp>
#include <sstream>

void readConn(Convolution* conn, std::string weightsInDir, int time){
   std::stringstream ss;
   ss << weightsInDir << "/" << conn->getName() << "_time" << time;
   std::string connStr = ss.str();
   std::cout << "Reading conn " << connStr << "\n";
   conn->loadWeights(connStr + "_W.mat");
   conn->loadBias(connStr + "_B.mat");
}

int main(void){
   int batch = 128;
   //Each inner run time is one time through the dataset
   int epochTime = 400; //Each inner run time is one time through the dataset

   std::string weightsInDir= "/home/sheng/workspace/DeepCNNData/cifar/out/weights/";

   int timePeriod = 400;
   
   Column* myCol = new Column(batch, //batch
                      1292749337//seed
                      );

   MatInput* input= new MatInput();
   input->setParams(myCol, //column name
                         "input", //name
                         32, //ny
                         32, //nx
                         3, //features
                         "/home/sheng/workspace/DeepCNNData/cifar/formatted/testData.mat");//list of images

   MatInput* gt = new MatInput(); gt->setParams(myCol, //column name
                         "gt", //name
                         1, //ny
                         1, //nx
                         10, //features
                         "/home/sheng/workspace/DeepCNNData/cifar/formatted/testLabels.mat");//list of images

   Convolution* conv1 = new Convolution();
   conv1->setParams(myCol, //column
                   "conv1", //name
                   5, //nyp
                   5, //nxp
                   32, //nfp
                   1, //ystride
                   1, //xstride
                   2, //load weights
                   .0001, //not used
                   weightsInDir + "conv1_time400_W.mat", //filename
                   2, //uniform init of bias
                   0, //initVal of bias
                   weightsInDir + "conv1_time400_B.mat", //filename
                   0, //Plasticity is off
                   .001, //dw rate
                   .002, //db rate
                   0.9, //dw momentum
                   0.9, //db momentum
                   0.004 //decay
                   );

   Activation* conv1Buf = new Activation();
   conv1Buf->setParams(myCol,
                     "conv1Buf",
                     "linear");

   MaxPool* pool1 = new MaxPool();
   pool1->setParams(myCol, //column
                   "pool1", //name
                   3, //nyp
                   3, //nxp
                   2, //ystride
                   2 //xstride
                   );

   Activation* relu1 = new Activation();
   relu1->setParams(myCol,
                     "relu1",
                     "relu");

   Convolution* conv2 = new Convolution();
   conv2->setParams(myCol, //column
                   "conv2", //name
                   5, //nyp
                   5, //nxp
                   32, //nfp
                   1, //ystride
                   1, //xstride
                   2, //load weights
                   .01, //range of weights
                   weightsInDir + "conv2_time400_W.mat", //filename
                   2, //load bias
                   0, //initVal of bias
                   weightsInDir + "conv2_time400_B.mat", //filename
                   0, //Plasticity is on
                   .001, //dw rate
                   .002, //db rate
                   0.9, //dw momentum
                   0.9, //db momentum
                   0.004 //decay
                   );

   Activation * conv2Buf = new Activation();
   conv2Buf->setParams(myCol,
                     "conv2Buf",
                     "linear");

   MaxPool* pool2 = new MaxPool();
   pool2->setParams(myCol, //column
                   "pool2", //name
                   3, //nyp
                   3, //nxp
                   2, //ystride
                   2 //xstride
                   );

   Activation* relu2 = new Activation();
   relu2->setParams(myCol,
                     "relu2",
                     "relu");

   Convolution* conv3 = new Convolution();
   conv3->setParams(myCol, //column
                   "conv3", //name
                   5, //nyp
                   5, //nxp
                   64, //nfp
                   1, //ystride
                   1, //xstride
                   2, //init from file
                   .01, //range of weights
                   weightsInDir + "conv3_time400_W.mat", //filename
                   2, //init from file
                   0, //initVal of bias
                   weightsInDir + "conv3_time400_B.mat", //filename
                   0, //Plasticity is on
                   .001, //dw rate
                   .002, //db rate
                   0.9, //dw momentum
                   0.9, //db momentum
                   0.004 //decay
                   );

   Activation * conv3Buf = new Activation();
   conv3Buf->setParams(myCol,
                     "conv3Buf",
                     "linear");

   MaxPool* pool3 = new MaxPool();
   pool3->setParams(myCol, //column
                   "pool3", //name
                   3, //nyp
                   3, //nxp
                   2, //ystride
                   2 //xstride
                   );

   Activation* relu3 = new Activation();
   relu3->setParams(myCol,
                     "relu3",
                     "relu");

   FullyConnected* fc64 = new FullyConnected();
   fc64->setParams(myCol, //column
                   "fc64", //name
                   64, //nfp
                   2, //init from file
                   .1, //range of weights
                   weightsInDir + "fc64_time400_W.mat", //filename, not used
                   2, //init from file
                   0, //initVal of bias
                   weightsInDir + "fc64_time400_B.mat", //filename, not used
                   0, //Plasticity is off
                   .001, //dw rate
                   .002, //db rate
                   0.9, //dw momentum
                   0.9, //db momentum
                   0.03 //decay
                   );

   Activation * relu4 = new Activation();
   relu4->setParams(myCol,
                     "hidden4",
                     "relu");
   
   FullyConnected* fc10 = new FullyConnected();
   fc10->setParams(myCol, //column
                   "fc10", //name
                   10, //nfp
                   2, //init from file
                   .1, //range of weights
                   weightsInDir + "fc10_time400_W.mat", //filename
                   2, //init from file
                   0, //initVal of bias
                   weightsInDir + "fc10_time400_B.mat", //filename
                   0, //Plasticity is off
                   .001, //dw rate
                   .002, //db rate
                   0.9, //dw momentum
                   0.9, //db momentum
                   0.03 //decay
                   );

   SoftmaxCost* cost = new SoftmaxCost();
   cost->setParams(myCol,
                   "cost",
                   epochTime, //once per almost epoch through dataset
                   "/home/sheng/workspace/DeepCNNData/cifar/out/totalTestCost.txt", //cost file
                   "/home/sheng/workspace/DeepCNNData/cifar/out/testAccuracy.txt"); //Cost accuracy
                   //"/home/sheng/workspace/DeepCNNData/cifar/out/testEst_time400.txt"); //cost est file

   myCol->addLayer(input);
   myCol->addConn(conv1);
   myCol->addLayer(conv1Buf);
   myCol->addConn(pool1);
   myCol->addLayer(relu1);
   myCol->addConn(conv2);
   myCol->addLayer(conv2Buf);
   myCol->addConn(pool2);
   myCol->addLayer(relu2);
   myCol->addConn(conv3);
   myCol->addLayer(conv3Buf);
   myCol->addConn(pool3);
   myCol->addLayer(relu3);
   myCol->addConn(fc64);
   myCol->addLayer(relu4);
   myCol->addConn(fc10);
   myCol->addLayer(cost);
   myCol->addGroundTruth(gt);

   //Initialize network
   myCol->initialize();

   int numEpochs = 258; //Running for 100 epcohs

   //Run and write to mat file, TODO put this into layers
   for(int i = 2; i < numEpochs; i++){
      myCol->run(epochTime);
      int time = (i) * timePeriod;
      readConn(conv1, weightsInDir, time);
      readConn(conv2, weightsInDir, time);
      readConn(conv3, weightsInDir, time);
      readConn(fc64, weightsInDir, time);
      readConn(fc10, weightsInDir, time);
   }
   
}
