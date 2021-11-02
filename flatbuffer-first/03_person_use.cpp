using namespace std;

#include <iostream>

// 헤더 선언
#include "flatbuffers/flatbuffers.h"
#include "person_generated.h"


void * operator new(size_t size){
  int *p=(int*)malloc(size);
  cout<<*p<<" "<<p<<endl;
  return p;
}
void operator delete(void *p)
{   free(p);
}
void * operator new[](size_t size){
  void *p=malloc(size);
  return p;
}
void operator delete[](void *p){
  free(p);
}


int main()
{
  // flatbuffer 빌더 객체 선언
  // 초기size가 충분히 크다면 이후에 malloc을 호출하지 않는다.
  flatbuffers::FlatBufferBuilder builder(64);

  // ----------------------------------------------------------------
  // 직렬화할 객체 데이터를 작성하여 직렬화한다.

  auto name = builder.CreateString("홍길동");
  //std::cout << "size=" << builder.GetSize() << "\n";

  int age = 25;

  // 직렬화 완료
  builder.Finish(CreatePerson(builder, name, age));

  // 직렬화된 버퍼를 가져온다. 이 버퍼값을 네트워크로 전송하거나 파일로 저장할 수 있다.
  auto data = builder.GetBufferPointer();
  //std::cout << "size=" << builder.GetSize() << "\n";
    

  // ----------------------------------------------------------------
  // 역직렬화를 진행한다.

  // 역직렬화하여 person 객체로 가져온다.
  const Person* person = GetPerson(data);

  // 객체 출력 확인
  const flatbuffers::String* person_name = person->name();
  std::cout << "이름 : " << person_name->c_str() << "\n";
  std::cout << "나이 : " << person->age() << "\n";

  builder.Release();
  std::cout << "size=" << builder.GetSize() << "\n";

  builder.Clear();

  return 0;
}