#include "gtest/gtest.h"

#include "lib/hello-math.h"

TEST(hello_math_test,add)
{
  EXPECT_EQ(add(1,2),4);
}