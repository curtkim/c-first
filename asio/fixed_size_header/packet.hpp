#ifndef FIXED_SIZE_PACKAGE_PACKAGE_H_
#define FIXED_SIZE_PACKAGE_PACKAGE_H_

#include <stdint.h>

struct MyHeader {
  uint32_t body_len;
  uint32_t check_sum;
  uint32_t other;
};

static const uint32_t MAX_BODY_LEN = 4096;

struct MyPacket {
  MyHeader header;
  char body[MAX_BODY_LEN];
};

#endif /* FIXED_SIZE_PACKAGE_PACKAGE_H_ */