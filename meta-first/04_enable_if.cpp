// from https://hellobird.tistory.com/136
#include <type_traits>
#include <iostream>

using namespace std;

////////////////////////////////////////////////////////////////////////////////
/// enable_if가 반환값으로 사용된 경우
////////////////////////////////////////////////////////////////////////////////
// T가 float, doule, long double 인 경우 T 타입을 반환한다.
template <typename T>
typename std::enable_if_t<is_floating_point_v<T>, T>
foo1(T t)
{
  cout << "foo1: 실수형\n";
  cout << t << endl;
  return t;
}

// enable_if_t는 C++11에서 지원하는 Aliasing template을 이용한 헬퍼!
// Aliasing template은 VS2013에서부터 지원된다.
// typename enable_if<is_integral<T>::value, T>::type과 동일하다.
// T가 정수형인 경우 T 타입을 반환한다
template <typename T>
enable_if_t<is_integral_v<T>, T>
foo1(T t)
{
  cout << "foo1: 정수형\n";
  cout << t << endl;
  return t;
}

////////////////////////////////////////////////////////////////////////////////
/// enable_if가 함수 인자로 사용된 경우
////////////////////////////////////////////////////////////////////////////////
// T가 정수형이 아닌 경우 컴파일 에러가 발생하며,
// T가 만약 int 타입인 경우 아래 함수는 다음과 같다.
/*
int foo2(int num, void* p = 0)
{
    return num;
}
*/
// enable_if의 템플릿 2번째 인자를 생략함으로써, defalut template parameter인 void 타입이 된다.
template <typename T>
T foo2(T t, typename std::enable_if_t<is_integral_v<T> >* = 0)
{
  return t;
}

////////////////////////////////////////////////////////////////////////////////
/// enable_if가 함수 템플릿 인자로 사용된 경우
////////////////////////////////////////////////////////////////////////////////
// enable_if의 템플릿 2번째 인자를 생략함으로써, defalut template parameter인 void 타입이 된다
template <typename T, typename = typename std::enable_if_t<is_integral_v<T> > >
T foo3(T t)
{
  return t;
}

////////////////////////////////////////////////////////////////////////////////
/// enable_if가 클래스 템플릿 인자로 사용된 경우
////////////////////////////////////////////////////////////////////////////////
template <typename T, typename Enable = void>
class A; // undefined

// 템플릿 부분 특수화에 사용됨
// enable_if의 템플릿 2번째 인자를 생략함으로써, defalut template parameter인 void 타입이 된다
// 역시 실수형이 아니면 컴파일 에러
template <typename T>
class A<T, typename std::enable_if_t<is_floating_point_v<T> > >
{
};

int main()
{
  // OK, 정수형 foo1 호출
  foo1(10);
  // OK, 실수형 foo1 호출
  // 만약, 실수형 foo1 함수가 정의되어 있지 않다면, 컴파일 에러
  foo1(1.2);

  // foo2는 정수형만 지원한다.
  // foo2(0.1)은 컴파일 에러
  foo2(7);

  // foo3도 정수형만 지원한다
  // foo3(1.2);은 컴파일 에러
  foo3(34); // OK

  // A<int> a1; 컴파일 에러
  A<double> a1; // OK
}