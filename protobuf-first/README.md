https://github.com/conan-io/examples/tree/master/libraries/protobuf/serialization

~/.conan/profiles/default에
compiler.libcxx=libstdc++11
를 적용하기 전까지 에러가 발생했었음.

- make를 실행하면 sensor.pb.h, sensor.b.cc를 생성한다.
- 그때 clion에서 error가 참조 에러가 사라진다.


## proto의 fields가 같고 class name이 다른 경우 read / write할 수 있는가?
main.cc가 sensor.proto로 쓰고, 
read2.cc가 sensor2.proto로 읽는다. (성공)
