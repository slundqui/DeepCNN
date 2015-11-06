#include <limits.h>
#include "gtest/gtest.h"
#include <iostream>
#include <src/Column.hpp>
//#include <matio.h>


//Fixture for testing baseLayer class
class ListTests: public ::testing::Test {
   protected:
      virtual void SetUp(){
         ::testing::FLAGS_gtest_death_test_style = "threadsafe"; //To suppress test warning
         myCol = new Column(1, 1, 1);

         myLayer0 = new BaseLayer();
         myLayer0->setParams(myCol, "layer0", 1);

         myLayer1 = new BaseLayer();
         myLayer1->setParams(myCol, "layer1", 1);

         myLayer2 = new BaseLayer();
         myLayer2->setParams(myCol, "layer2", 1);

         myConn0 = new BaseConnection();
         myConn0->setParams(myCol, "conn0", 1, 1, 1);

         myConn1 = new BaseConnection();
         myConn1->setParams(myCol, "conn1", 1, 1, 1);
      }

      virtual void TearDown(){
         delete myCol;
         delete myLayer0;
         delete myLayer1;
         delete myLayer2;
         delete myConn0;
         delete myConn1;
      }
      Column* myCol;
      BaseLayer *myLayer0, *myLayer1, *myLayer2;
      BaseConnection *myConn0, *myConn1;
};

TEST(various, ErrorUndefinedParameters){
   ::testing::FLAGS_gtest_death_test_style = "threadsafe"; //To suppress test warning
   BaseLayer* myLayer = new BaseLayer();
   EXPECT_EXIT(myLayer->initialize(), ::testing::ExitedWithCode(UNDEFINED_PARAMS), "");
}

TEST_F(ListTests, ErrorAdding2Layers){
   myCol->addLayer(myLayer0);
   EXPECT_EXIT(myCol->addLayer(myLayer1), ::testing::ExitedWithCode(INVALID_DATA_ADDITION), "");
}

TEST_F(ListTests, ErrorAddingFirstConn){
   EXPECT_EXIT(myCol->addConn(myConn0), ::testing::ExitedWithCode(INVALID_DATA_ADDITION), "");
}

TEST_F(ListTests, ErrorAdding2Conns){
   myCol->addLayer(myLayer0);
   myCol->addConn(myConn0);
   EXPECT_EXIT(myCol->addConn(myConn1), ::testing::ExitedWithCode(INVALID_DATA_ADDITION), "");
}

TEST_F(ListTests, DoubleLinkedList){
   myCol->addLayer(myLayer0);
   myCol->addConn(myConn0);
   myCol->addLayer(myLayer1);
   myCol->addConn(myConn1);
   myCol->addLayer(myLayer2);
   ASSERT_EQ(myLayer0->getPrev(), (void*)NULL);
   ASSERT_EQ(myLayer0->getNext(), myConn0);
   ASSERT_EQ(myConn0->getPrev(), myLayer0);
   ASSERT_EQ(myConn0->getNext(), myLayer1);
   ASSERT_EQ(myLayer1->getPrev(), myConn0);
   ASSERT_EQ(myLayer1->getNext(), myConn1);
   ASSERT_EQ(myConn1->getPrev(), myLayer1);
   ASSERT_EQ(myConn1->getNext(), myLayer2);
   ASSERT_EQ(myLayer2->getPrev(), myConn1);
   ASSERT_EQ(myLayer2->getNext(), (void*)NULL);
}

////For use in writing to a mat file 
//TEST(mat_writing, writeMat){
//   mat_t *matfp;
//   matvar_t *matvar;
//   matvar_t * field;
//   const char* outName = "test.mat";
//   const char *fields[4] = {"ny", "nx", "nf", "data"};
//   int ny = 5;
//   int nx = 7;
//   int nf = 3;
//   double data[5*7*3];
//   for(int i = 0; i < ny*nx*nf; i++){
//      data[i] = i; //Initializing data
//   }
//   size_t scalar_dim[2] = {1, 1};
//   size_t array_dim[2] = {ny*nx*nf, 1};
//
//   matfp = Mat_CreateVer(outName, NULL, MAT_FT_DEFAULT);
//   if(matfp == NULL){
//      std::cerr << "Error opening MAT file " << outName << "\n";
//      return;
//   }
//
//   matvar = Mat_VarCreateStruct("testVar",2,scalar_dim,fields,4);
//   if(!matvar){
//      Mat_Close(matfp);
//      std::cerr << "Error creating var\n";
//      return;
//   }
//
//   field = Mat_VarCreate(NULL, MAT_C_INT32, MAT_T_INT32, 2, scalar_dim, &ny, MAT_F_DONT_COPY_DATA);
//   Mat_VarSetStructFieldByName(matvar, fields[0], 0, field);
//
//   field = Mat_VarCreate(NULL, MAT_C_INT32, MAT_T_INT32, 2, scalar_dim, &nx, MAT_F_DONT_COPY_DATA);
//   Mat_VarSetStructFieldByName(matvar, fields[1], 0, field);
//
//   field = Mat_VarCreate(NULL, MAT_C_INT32, MAT_T_INT32, 2, scalar_dim, &nf, MAT_F_DONT_COPY_DATA);
//   Mat_VarSetStructFieldByName(matvar, fields[2], 0, field);
//
//   field = Mat_VarCreate(NULL, MAT_C_SINGLE, MAT_T_SINGLE, 2, array_dim, data, MAT_F_DONT_COPY_DATA);
//   Mat_VarSetStructFieldByName(matvar, fields[3], 0, field);
//
//   Mat_VarWrite(matfp, matvar, MAT_COMPRESSION_NONE);
//
//   Mat_VarFree(matvar);
//
//   Mat_Close(matfp);
//   return;
//}


