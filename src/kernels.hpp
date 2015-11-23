extern "C" void leastSqTotalCost(float* truth, float* estimate, int batchcount, float* out, int n_blocks, int block_size);
extern "C" void leastSqTotalCostRunSize(int* gridSize, int* blockSize, int batchcount);

extern "C" void leastSqCalcGrad(float* truth, float* estimate, int batchcount, float* out, int gridSize, int blockSize);
extern "C" void leastSqCalcGradRunSize(int* gridSize, int* blockSize, int batchcount);

extern "C" void softmaxTotalCost(float* truth, float* estimate, int batchcount, int bSize, float* out, int gridSize, int blockSize);
extern "C" void softmaxTotalCostRunSize(int* gridSize, int* blockSize, int batchcount);

extern "C" void convLearningRule(float* d_Weight, float* d_dWeight, float* d_GWeight, int count, float eps, float mom, float decay, int n_blocks, int block_size); 
extern "C" void convLearningRuleRunSize(int* gridSize, int* blockSize, int count);
