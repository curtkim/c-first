#include <cstdint>
#include <cstring>
#include "doctest/doctest.h"

TEST_CASE ("test bitwize op") {
  uint32_t a = 800 * 600 * 4;

  char char_len[4];
  char_len[0] = (a >> 0);
  char_len[1] = (a >> 8);
  char_len[2] = (a >> 16);
  char_len[3] = (a >> 24);

  CHECK((int) char_len[0] == 0);
  CHECK((int) char_len[1] == 0X4C); // 76
  CHECK((int) char_len[2] == 0X1D); // 29
  CHECK((int) char_len[3] == 0);

  SUBCASE("char[] to unit32") {
    uint32_t b = 0;
    std::memcpy(&b, char_len, sizeof(uint32_t));
    CHECK(a == b);
  }
}

