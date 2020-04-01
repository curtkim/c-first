#include "gtest/gtest.h"
#include "utils.hpp"

TEST(GreaterTest,AisGreater){
    EXPECT_EQ(3,GreatestOfThree(3,1,2));
};
TEST(GreaterTest,BisGreater){
    EXPECT_EQ(3,GreatestOfThree(1,3,2));
};
TEST(GreaterTest,CisGreater){
    EXPECT_EQ(3,GreatestOfThree(1,2,3));
};

int main(int argc, char**argv)
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}