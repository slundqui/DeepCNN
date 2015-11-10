#ifndef UTILS_HPP_ 
#define UTILS_HPP_ 
#include "includes.hpp"

#define CudaError(ans) { gpuAssert((ans), __FILE__, __LINE__); }
#define CudnnError(ans) { cudnnAssert((ans), __FILE__, __LINE__); }
#define CudaCallError() { lastCallError(__FILE__, __LINE__); }

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

inline void lastCallError(const char *file, int line, bool term=true){
   gpuAssert(cudaGetLastError(), file, line, term);
}

inline void printMat(float* array, int nb, int ny, int nx, int nf){
   for(int bi = 0; bi < nb; bi++){
      printf("Batch %d: \n", bi);

      for(int fi = 0; fi < nf; fi++){
         printf("\tFeature %d\n", fi); 

         for(int yi = 0; yi < ny; yi++){
            printf("\t");
            for(int xi = 0; xi < nx; xi++){
               int idx = (bi*ny*nx*nf) + (fi*ny*nx) + (yi*nx) + xi;
               printf("%f ", array[idx]);
            }
            printf("\n");
         }
      }

   }
   
}


inline void cudnnPrintTensorDesc(cudnnTensorDescriptor_t desc){
   cudnnDataType_t dataType;
   int n, c, h, w, nStride, cStride, hStride, wStride;
   if(!desc) std::cerr << "Input tensor desc is null\n";
   CudnnError(cudnnGetTensor4dDescriptor(desc,
      &dataType, //dataType
      &n, //batch
      &c, //channels
      &h, //height
      &w, //width
      &nStride, //strides of dims
      &cStride, 
      &hStride,
      &wStride));
   assert(dataType == CUDNN_DATA_FLOAT);
   std::cout << "Tensor descriptor dims (batch, y, x, feature): " << n << ", " << h << ", " << w << ", " << c << "\n";
}

inline void cudnnPrintFilterDesc(cudnnFilterDescriptor_t desc){
   cudnnDataType_t dataType;
   int k, c, h, w;
   CudnnError(cudnnGetFilter4dDescriptor(desc,
      &dataType, //dataType
      &k, //outputFeatures
      &c, //inputFeatures
      &h, //height
      &w //width
      ));
   assert(dataType == CUDNN_DATA_FLOAT);
   std::cout << "Tensor descriptor dims (kernels, y, x, feature): " << k << ", " << h << ", " << w << ", " << c << "\n";
}

inline void cudnnPrintConvDesc(cudnnConvolutionDescriptor_t desc){
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

inline void writeDeviceData(std::string matFilename, int nDims, size_t * dims, float* d_data){
   mat_t *matfp;
   matvar_t *matvar;
   size_t size = sizeof(float);
   for(int i = 0;  i < nDims; i++){
      size *= dims[i];
   }
   float* h_data = (float*) malloc(size);
   CudaError(cudaMemcpy(h_data, d_data, size, cudaMemcpyDeviceToHost));

   matfp = Mat_CreateVer(matFilename.c_str(), NULL, MAT_FT_DEFAULT);
   if(matfp == NULL){
      std::cerr << "Error opening MAT file " << matFilename<< "\n";
      exit(-1);
   }

   matvar = Mat_VarCreate("data", MAT_C_SINGLE, MAT_T_SINGLE, nDims, dims, h_data, MAT_F_DONT_COPY_DATA);
   if(!matvar){
      Mat_Close(matfp);
      std::cerr << "Error creating var\n";
      exit(-1);
   }

   Mat_VarWrite(matfp, matvar, MAT_COMPRESSION_NONE);

   Mat_VarFree(matvar);
   Mat_Close(matfp);
   free(h_data);
}

inline void readDataToDevice(std::string matFilename, float* d_data, int* nDims, size_t** dims){
   mat_t *matfp;
   matvar_t *matvar;
   size_t size = sizeof(float);

   //Reading from mat
   matfp = Mat_Open(matFilename.c_str(), MAT_ACC_RDONLY);
   if(matfp == NULL){
      std::cerr << "Error opening MAT file " << matFilename << "\n";
      exit(-1);
   }

   matvar = Mat_VarRead(matfp, (char*)"data");
   if(!matvar){
      Mat_Close(matfp);
      std::cerr << "Error reading var\n";
      exit(-1);
   }

   assert(matvar->data_type == 7); //Single data type

   *nDims = matvar->rank;
   *dims = (size_t*)malloc(*nDims * sizeof(size_t));
   for(int i = 0; i < *nDims; i++){
      (*dims)[i] = matvar->dims[i];
      size *= matvar->dims[i];
   }

   float *h_data = static_cast<float*>(matvar->data);
   
   CudaError(cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice));
   Mat_VarFree(matvar);
   Mat_Close(matfp);
}

#endif
