#include "gtest/gtest.h"
#include <iostream>
#include <matio.h>

//For use in writing to a mat file 
TEST(mat_writing, writeMat){
   //Writing to mat
   mat_t *matfp;
   matvar_t *matvar;
   const char* outName = "tests/out/test.mat";
   int nb = 2;
   int nf = 3;
   int ny = 5;
   int nx = 7;
   float data[nb*nf*nx*ny];

   for(int i = 0; i < nb*ny*nx*nf; i++){
      data[i] = i; //Initializing data
   }
   size_t array_dim[4] = {(size_t)nb, (size_t)nf, (size_t)ny, (size_t)nx};

   matfp = Mat_CreateVer(outName, NULL, MAT_FT_DEFAULT);
   if(matfp == NULL){
      std::cerr << "Error opening MAT file " << outName << "\n";
      return;
   }

   matvar = Mat_VarCreate("data", MAT_C_SINGLE, MAT_T_SINGLE, 4, array_dim, data, MAT_F_DONT_COPY_DATA);
   if(!matvar){
      Mat_Close(matfp);
      std::cerr << "Error creating var\n";
      return;
   }
   Mat_VarWrite(matfp, matvar, MAT_COMPRESSION_NONE);
   Mat_VarFree(matvar);
   Mat_Close(matfp);

   //Reading from mat
   matfp = Mat_Open(outName, MAT_ACC_RDONLY);
   if(matfp == NULL){
      std::cerr << "Error opening MAT file " << outName << "\n";
      return;
   }
   matvar = Mat_VarRead(matfp, (char*)"data");
   if(!matvar){
      Mat_Close(matfp);
      std::cerr << "Error reading var\n";
      return;
   }
   EXPECT_EQ(matvar->rank, 4);
   EXPECT_EQ(matvar->dims[0], 2);
   EXPECT_EQ(matvar->dims[1], 3);
   EXPECT_EQ(matvar->dims[2], 5);
   EXPECT_EQ(matvar->dims[3], 7);

   const float* outData = static_cast<const float*>(matvar->data);
   for(int i = 0; i < nb*ny *nx*nf; i++){
      ASSERT_EQ(outData[i], data[i]);
   }
   return;
}
