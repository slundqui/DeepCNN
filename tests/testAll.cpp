#include <limits.h>
#include "gtest/gtest.h"
#include <iostream>
#include "../src/BaseLayer.hpp"


//namespace {
//   //Fixture for testing baseLayer class
//   class BaseLayerTest : public ::testing::Test {
//      protected:
//         //virtual void SetUp(){
//
//         //}
//   };
//}

TEST(BaseLayerTest, ErrorUndefinedParameters){
   ::testing::FLAGS_gtest_death_test_style = "threadsafe"; //To suppress test warning
   BaseLayer* myLayer = new BaseLayer("testLayer");
   EXPECT_EXIT(myLayer->initialize(), ::testing::ExitedWithCode(UNDEFINED_PARAMS), "");
}


int main(int argc, char **argv){
   ::testing::InitGoogleTest(&argc, argv);
   return RUN_ALL_TESTS();
}
