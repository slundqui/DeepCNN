#include "test_utils.hpp"

bool gradientCheck(Column* myCol, Convolution* checkConn, MatInput* input, MatInput* gt, BaseCostFunction* cost, float tolerance, float* h_actualWGrad, float* h_actualBGrad){

   float epsilon = 1e-4;

   int numWeights = checkConn->getNumWeights();
   int numBias = checkConn->getNumBias();
   float* baseWeights = checkConn->getHostW();
   float* baseBias = checkConn->getHostB();
   int batch = input->getBSize();

   bool passed = true;

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

      float posCost = cost->getHostTotalCost();

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
      float negCost = cost->getHostTotalCost();
      //std::cout << "posCost " << posCost << " negCost " << negCost << "\n";
      
      float empGrad = -(posCost - negCost)/(2*epsilon);
      float actGrad = h_actualWGrad[weightIdx];
      //printf("Weight idx: %d  EmpGrad: %.10f EmpGrad  ActGrad: %.10f\n", weightIdx, empGrad, actGrad);
      //std::cout << "Weight Idx: " << weightIdx << " EmpGrad: " << empGrad << " ActGrad: " << actGrad << "\n";
      //nan check
      if(empGrad != empGrad || actGrad != actGrad){
         passed = false;
         std::cout << "Weight nan\n";
         return passed;
      }
      if(fabs(empGrad - actGrad) > tolerance){
         std::cout << "Weight Idx: " << weightIdx << " EmpGrad: " << empGrad << " ActGrad: " << actGrad << "\n";
         passed = false;
         return passed;
      }

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

      float posCost = cost->getHostTotalCost();

      //Grab neg cost
      input->rewind();
      gt->rewind();
      //Set weight - epsilon
      checkConn->setBias(biasIdx, baseBias[biasIdx] - epsilon);
      //Run network for 1 timestep
      myCol->run(1);
      //Grab neg cost
      float negCost = cost->getHostTotalCost();
      
      //std::cout << "posCost " << posCost << " negCost " << negCost << "\n";
      float empGrad = -(posCost - negCost)/(2*epsilon);
      float actGrad = h_actualBGrad[biasIdx];

      //printf("Bias idx: %d  EmpGrad: %.10f EmpGrad  ActGrad: %.10f\n", biasIdx, empGrad, actGrad);
      //nan check
      if(empGrad != empGrad || actGrad != actGrad){
         passed = false;
         std::cout << "Bias nan\n";
         return passed;
      }
      if(fabs(empGrad - actGrad) > tolerance){
         std::cout << "Bias Idx: " << biasIdx << " EmpGrad: " << empGrad << " ActGrad: " << actGrad << "\n";
         passed = false;
         return passed;
      }

      //Reset weight
      checkConn->setBias(biasIdx, baseBias[biasIdx]);
   }

   free(baseWeights);
   free(baseBias);
   return passed;
}

