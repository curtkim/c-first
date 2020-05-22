// Catch has two natural expression assertion macro's:
// - REQUIRE() stops at first failure.
// - CHECK() continues after failure.

// There are two variants to support decomposing negated expressions:
// - REQUIRE_FALSE() stops at first failure.
// - CHECK_FALSE() continues after failure.

// main() provided in 000-CatchMain.cpp

#include <catch2/catch.hpp>

std::string one() {
  return "1";
}

TEST_CASE( "Assert that something is true (pass)", "[require]" ) {
  REQUIRE( one() == "1" );
}

TEST_CASE( "Assert that something is true (fail)", "[require]" ) {
  REQUIRE( one() == "x" );
}

TEST_CASE( "Assert that something is true (stop at first failure)", "[require]" ) {
  WARN( "REQUIRE stops at first failure:" );

  REQUIRE( one() == "x" );
  REQUIRE( one() == "1" );
}

TEST_CASE( "Assert that something is true (continue after failure)", "[check]" ) {
  WARN( "CHECK continues after failure:" );

  CHECK(   one() == "x" );
  REQUIRE( one() == "1" );
}

TEST_CASE( "Assert that something is false (stops at first failure)", "[require-false]" ) {
  WARN( "REQUIRE_FALSE stops at first failure:" );

  REQUIRE_FALSE( one() == "1" );
  REQUIRE_FALSE( one() != "1" );
}

TEST_CASE( "Assert that something is false (continue after failure)", "[check-false]" ) {
  WARN( "CHECK_FALSE continues after failure:" );

  CHECK_FALSE(   one() == "1" );
  REQUIRE_FALSE( one() != "1" );
}