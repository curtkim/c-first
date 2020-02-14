https://github.com/conan-io/examples/tree/master/libraries/protobuf/serialization

~/.conan/profiles/default에
compiler.libcxx=libstdc++11
를 적용하기 전까지 에러가 발생했었음.

- make를 실행하면 sensor.pb.h, sensor.b.cc를 생성한다.
- 그때 clion에서 error가 참조 에러가 사라진다.