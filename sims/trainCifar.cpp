//A run file that runs the DCN on cifar data

#include <Column.hpp>
#include <layers/MatInput.hpp>
#include <layers/SoftmaxCost.hpp>
#include <layers/LeastSquaresCost.hpp>
#include <connections/FullyConnected.hpp>
#include <connections/MaxPool.hpp>
#include <sstream>

void writeConn(Convolution* conn, std::string weightsOutDir, int time){
   std::stringstream ss;
   ss << weightsOutDir << "/" << conn->getName() << "_time" << time;
   std::string connStr = ss.str();

   conn->writeWeights(connStr + "_W.mat");
   conn->writeBias(connStr + "_B.mat");
}

int main(void){
   int batch = 128;
   //Each inner run time is one time through the dataset
   
   int epochTime = 400; //Each inner run time is one time through the dataset
   std::string weightsOutDir = "/home/sheng/workspace/DeepCNNData/cifar/out/weights/";
   
   Column* myCol = new Column(batch, //batch
                      289745937//seed
                      );

   MatInput* input= new MatInput();
   input->setParams(myCol, //column name
                         "input", //name
                         32, //ny
                         32, //nx
                         3, //features
                         "/home/sheng/workspace/DeepCNNData/cifar/formatted/trainData.mat");//list of images

   MatInput* gt = new MatInput(); gt->setParams(myCol, //column name
                         "gt", //name
                         1, //ny
                         1, //nx
                         10, //features
                         "/home/sheng/workspace/DeepCNNData/cifar/formatted/trainLabels.mat",//list of images
                         true //shuffle
         );

   Convolution* conv1 = new Convolution();
   conv1->setParams(myCol, //column
                   "conv1", //name
                   5, //nyp
                   5, //nxp
                   32, //nfp
                   1, //ystride
                   1, //xstride
                   1, //uniform random weights
                   .0001, //range of weights
                   "", //filename, not used
                   0, //uniform init of bias
                   0, //initVal of bias
                   "", //filename, not used
                   1, //Plasticity is on
                   .0001, //dw rate
                   .0002, //db rate
                   0.9, //dw momentum
                   0.9, //db momentum
                   0.0004 //decay
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
                   1, //uniform random weights
                   .01, //range of weights
                   "", //filename, not used
                   0, //uniform init of bias
                   0, //initVal of bias
                   "", //filename, not used
                   1, //Plasticity is on
                   .0001, //dw rate
                   .0002, //db rate
                   0.9, //dw momentum
                   0.9, //db momentum
                   0.0004 //decay
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
                   1, //uniform random weights
                   .01, //range of weights
                   "", //filename, not used
                   0, //uniform init of bias
                   0, //initVal of bias
                   "", //filename, not used
                   1, //Plasticity is on
                   .0001, //dw rate
                   .0002, //db rate
                   0.9, //dw momentum
                   0.9, //db momentum
                   0.0004 //decay
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
                   1, //uniform random weights
                   .1, //range of weights
                   "", //filename, not used
                   0, //uniform init of bias
                   0, //initVal of bias
                   "", //filename, not used
                   1, //Plasticity is on
                   .0001, //dw rate
                   .0002, //db rate
                   0.9, //dw momentum
                   0.9, //db momentum
                   0.003 //decay
                   );

   Activation * relu4 = new Activation();
   relu4->setParams(myCol,
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
                   .0001, //dw rate
                   .0002, //db rate
                   0.9, //dw momentum
                   0.9, //db momentum
                   0.003 //decay
                   );

   SoftmaxCost* cost = new SoftmaxCost();
   cost->setParams(myCol,
                   "cost",
                   epochTime, //once per almost epoch through dataset
                   "/home/sheng/workspace/DeepCNNData/cifar/out/train_totalCost.txt",
                   "/home/sheng/workspace/DeepCNNData/cifar/out/train_accuracy.txt");

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

   //Run
   myCol->initialize();

   //for(int i = 0; i < 10; i++){
   //   input->rewind();
   //   gt->rewind();
   //   myCol->run(1);
   //}

   //std::cout << "Estimate A" << "\n";
   //cost->printA();
   //std::cout << "GT A" << "\n";
   //gt->printA();

   int iterationCount = 0;
   int iterationMax = 40;

   std::cout << "Running for 10 epochs\n";
   
   int numEpochs = 10; //Running for 100 epcohs
   for(int i = 0; i < numEpochs; i++){
      myCol->run(epochTime);
      writeConn(conv1, weightsOutDir, myCol->getTimestep());
      writeConn(conv2, weightsOutDir, myCol->getTimestep());
      writeConn(conv3, weightsOutDir, myCol->getTimestep());
      writeConn(fc64, weightsOutDir, myCol->getTimestep());
      writeConn(fc10, weightsOutDir, myCol->getTimestep());
      std::cout << "(" << ((float)100 * iterationCount)/iterationMax << "\%) Epoch " << iterationCount << " out of " << iterationMax << "\n";
      iterationCount++;
   }

   std::cout << "Reducing learning rate by 10 for 10 epochs\n";
   //Reduce learning rate by factor of 10 after 8 epochs
   conv1->setDwRate(conv1->getDwRate()/10);
   conv2->setDwRate(conv2->getDwRate()/10);
   conv3->setDwRate(conv3->getDwRate()/10);
   fc64->setDwRate(fc64->getDwRate()/10);
   fc10->setDwRate(fc10->getDwRate()/10);

   conv1->setDbRate(conv1->getDbRate()/10);
   conv2->setDbRate(conv2->getDbRate()/10);
   conv3->setDbRate(conv3->getDbRate()/10);
   fc64->setDbRate(fc64->getDbRate()/10);
   fc10->setDbRate(fc10->getDbRate()/10);

   //Rerun for longer
   numEpochs = 10;
   for(int i = 0; i < numEpochs; i++){
      myCol->run(epochTime);
      writeConn(conv1, weightsOutDir, myCol->getTimestep());
      writeConn(conv2, weightsOutDir, myCol->getTimestep());
      writeConn(conv3, weightsOutDir, myCol->getTimestep());
      writeConn(fc64, weightsOutDir, myCol->getTimestep());
      writeConn(fc10, weightsOutDir, myCol->getTimestep());
      std::cout << "(" << ((float)100 * iterationCount)/iterationMax << "\%) Epoch " << iterationCount << " out of " << iterationMax << "\n";
      iterationCount++;
   }

   //Reduce learning rate by factor of 10 
   conv1->setDwRate(conv1->getDwRate()/10);
   conv2->setDwRate(conv2->getDwRate()/10);
   conv3->setDwRate(conv3->getDwRate()/10);
   fc64->setDwRate(fc64->getDwRate()/10);
   fc10->setDwRate(fc10->getDwRate()/10);
   conv1->setDbRate(conv1->getDbRate()/10);
   conv2->setDbRate(conv2->getDbRate()/10);
   conv3->setDbRate(conv3->getDbRate()/10);
   fc64->setDbRate(fc64->getDbRate()/10);
   fc10->setDbRate(fc10->getDbRate()/10);

   std::cout << "Reducing learning rate by 10 for 10 epochs\n";
   //Rerun for much longer
   numEpochs = 10; //Running for 100 epcohs
   for(int i = 0; i < numEpochs; i++){
      myCol->run(epochTime);
      writeConn(conv1, weightsOutDir, myCol->getTimestep());
      writeConn(conv2, weightsOutDir, myCol->getTimestep());
      writeConn(conv3, weightsOutDir, myCol->getTimestep());
      writeConn(fc64, weightsOutDir, myCol->getTimestep());
      writeConn(fc10, weightsOutDir, myCol->getTimestep());
      std::cout << "(" << ((float)100 * iterationCount)/iterationMax << "\%) Epoch " << iterationCount << " out of " << iterationMax << "\n";
      iterationCount++;
   }

   //Final reduction of learning rate
   conv1->setDwRate(conv1->getDwRate()/10);
   conv2->setDwRate(conv2->getDwRate()/10);
   conv3->setDwRate(conv3->getDwRate()/10);
   fc64->setDwRate(fc64->getDwRate()/10);
   fc10->setDwRate(fc10->getDwRate()/10);

   conv1->setDbRate(conv1->getDbRate()/10);
   conv2->setDbRate(conv2->getDbRate()/10);
   conv3->setDbRate(conv3->getDbRate()/10);
   fc64->setDbRate(fc64->getDbRate()/10);
   fc10->setDbRate(fc10->getDbRate()/10);


   std::cout << "Reducing learning rate by 10 for 10 epochs\n";
   //Rerun for much longer
   numEpochs = 10; //Running for 100 epcohs
   for(int i = 0; i < numEpochs; i++){
      myCol->run(epochTime);
      writeConn(conv1, weightsOutDir, myCol->getTimestep());
      writeConn(conv2, weightsOutDir, myCol->getTimestep());
      writeConn(conv3, weightsOutDir, myCol->getTimestep());
      writeConn(fc64, weightsOutDir, myCol->getTimestep());
      writeConn(fc10, weightsOutDir, myCol->getTimestep());
      std::cout << "(" << ((float)100 * iterationCount)/iterationMax << "\%) Epoch " << iterationCount << " out of " << iterationMax << "\n";
      iterationCount++;
   }
}

