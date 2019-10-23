#include "gtest/gtest.h"
#include "mymath.hpp"


TEST(mymath, add){
    EXPECT_EQ(add(1,2),3);
}
