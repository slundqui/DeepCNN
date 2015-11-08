#ifndef CUDA_UTILS_HPP_ 
#define CUDA_UTILS_HPP_ 

#include <cudnn.h>
#include <cstdlib>
#include <cassert>

#define CudaError(ans) { gpuAssert((ans), __FILE__, __LINE__); }
#define CudnnError(ans) { cudnnAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char *file, int line, bool term=true){
   if(code != cudaSuccess){
      std::cerr << "Cuda error: " << cudaGetErrorString(code) << " " << file << " " << line << "\n";
      if(term) abort();
   }
}

inline void cudnnAssert(cudnnStatus_t status, const char *file, int line, bool term=true){
   if(status != CUDNN_STATUS_SUCCESS){
      std::cerr << "CuDNN error: " << cudnnGetErrorString(status) << " " << file << " " << line << "\n";
      if(term) abort();
   }
}

inline void cudnnPrintTensorDesc(cudnnTensorDescriptor_t desc){
   cudnnDataType_t dataType;
   int n, c, h, w, nStride, cStride, hStride, wStride;
   if(!desc) std::cerr << "Input tensor desc is null\n";
   CudnnError(cudnnGetTensor4dDescriptor(desc,
      &dataType,
      &n, //dataType
      &c, //channels
      &h, //height
      &w, //width
      &nStride, //strides of dims
      &cStride, 
      &hStride,
      &wStride));
   assert(dataType == CUDNN_DATA_FLOAT);
   std::cout << "Tensor descriptor dims (batch, feature, y, x): " << n << ", " << c << ", " << h << ", " << w << "\n";
}

inline void cudnnPrintConvDesc(cudnnConvolutionDescriptor_t desc){
   cudnnDataType_t dataType;
   int pad_h, pad_w, u, v, upscalex, upscaley;
   cudnnConvolutionMode_t mode;
   if(!desc) std::cerr << "Input conv desc is null\n";

   CudnnError(cudnnGetConvolution2dDescriptor(desc,
      &pad_h, //Padding y
      &pad_w, //Padding x
      &u, //y stride
      &v, //x stride
      &upscalex, //upscale x
      &upscaley, //upscale y
      &mode)); //conv mode

   assert(mode == CUDNN_CONVOLUTION);
   assert(upscalex == 1);
   assert(upscaley == 1);
   std::cout << "Conv desc (ypad, xpad, ystride, xstride): " << pad_h << ", " << pad_w << ", " << u << ", " << v << "\n";
}

#endif
