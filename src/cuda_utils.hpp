
#ifndef CUDA_UTILS_HPP_ 
#define CUDA_UTILS_HPP_ 

#define CudaError(ans) { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true){
   if(code != cudaSuccess){
      std::cerr << "Cuda error: " << cudaGetErrorString(code) << " " << file << " " << line << "\n";
      if(abort) exit(code);
   }
}

#endif
