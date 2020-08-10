#include <iostream>
#include <cstdint>
#include <vector>
#include <array>
#include <memory>
#include <assert.h>

struct SensorData {
  uint64_t timestamp;
  uint64_t frame;
  uint16_t type;

  SensorData(uint64_t timestamp, uint64_t frame, uint16_t type): timestamp(timestamp), frame(frame), type(type) {}
};

using Cell = std::array<unsigned char, 3>;  // RGB

struct CameraData : public SensorData {
  int height;
  int width;
  std::vector<Cell> data; // row major

  CameraData(uint64_t timestamp, uint64_t frame, uint16_t type, int height, int width, std::vector<Cell> data):
    SensorData(timestamp, frame, type), height(height), width(width), data(std::move(data)){
  }
};

SensorData make() {
  std::vector<Cell> data;
  return CameraData{1, 1, 1, 1, 1, data};
}

std::shared_ptr<SensorData> make2() {
  std::vector<Cell> data;
  return std::make_shared<CameraData>(1, 1, 1, 1, 1, data);
}

int main() {
  SensorData a = {1,1,1};
  assert(a.frame == 1);

  std::vector<Cell> data;
  CameraData c = {1, 1, 1, 1, 1, data};
  assert(c.frame == 1);

  SensorData b = make();
  assert(b.frame == 1);

  std::shared_ptr<SensorData> d = make2();
  assert(d->frame == 1);

  std::shared_ptr<CameraData> e = std::static_pointer_cast<CameraData>(d);
  assert(e->width == 1);
}