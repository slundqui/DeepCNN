#include <limits.h>
#include "gtest/gtest.h"
#include <iostream>
#include <src/Column.hpp>

//Fixture for testing baseLayer class
class ListTests: public ::testing::Test {
   protected:
      virtual void SetUp(){
         ::testing::FLAGS_gtest_death_test_style = "threadsafe"; //To suppress test warning
         myCol = new Column(1, 1, 1);

         myLayer0 = new BaseLayer();
         myLayer0->setParams(myCol, "layer0");

         myLayer1 = new BaseLayer();
         myLayer1->setParams(myCol, "layer1");

         myLayer2 = new BaseLayer();
         myLayer2->setParams(myCol, "layer2");

         myConn0 = new BaseConnection();
         myConn0->setParams(myCol, "conn0", 1, 1, 1, 1, 1);

         myConn1 = new BaseConnection();
         myConn1->setParams(myCol, "conn1", 1, 1, 1, 1, 1);
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



