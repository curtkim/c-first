#include <tuple>
#include "gtest/gtest.h"
#include "mymath.hpp"


TEST(mymath, add){
    EXPECT_EQ(add(1,2),3);
}

class MyMathParametersTests :public ::testing::TestWithParam<std::tuple<int, int, int>> {

};

TEST_P(MyMathParametersTests, add) {
    int expected = std::get<0>(GetParam());
    int a = std::get<1>(GetParam());
    int b = std::get<2>(GetParam());
    ASSERT_EQ(expected, add(a,b));
}

INSTANTIATE_TEST_CASE_P(
        MyMathParametersTests,
        add,
        ::testing::Values(
                std::make_tuple(3,1,2),
                std::make_tuple(7,3,4));