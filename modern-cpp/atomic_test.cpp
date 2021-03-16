#include <atomic>
#include <iostream>

using namespace std;

int main() {

  atomic<int> intAtomic;
  intAtomic = 2;


  // 값 더하기 : 멤버 함수 fetch_add
  intAtomic.fetch_add(1);
  // 값 더하기 : 멤버 함수 operator++(), operator++(int)
  intAtomic++;
  // 값 더하기 : 일반 함수 atomic_fetch_add
  atomic_fetch_add(&intAtomic, 1);
  cout << intAtomic << "\n";


  // 값 빼기 : 멤버 함수 fetch_sub
  intAtomic.fetch_sub(1);
  // 값 빼기 : 멤버 함수 operator--(), operator--(int)
  intAtomic--;
  // 값 빼기 : 일반 함수 atomic_fetch_sub
  atomic_fetch_sub(&intAtomic, 1);
  cout << intAtomic << "\n";


  // 값 로드 : 멤버 함수 load
  int value = intAtomic.load();
  // 값 로드 : 멤버 함수 Operator T
  value = intAtomic;
  // 값 로드 : 일반 함수 atomic_load
  value = atomic_load(&intAtomic);

  cout << value << "\n";


  // 값 교환 : 멤버 함수 exchange
  int oldValue = intAtomic.exchange(5);
  // 값 교환 : 일반 함수 atomic_exchange
  oldValue = atomic_exchange(&intAtomic, 3);
  cout << intAtomic << "\n";


  int comparand = 5;
  int newValue = 10;

  // 값 비교 교환 : 멤버 함수 (value = 3, 5와 같다면, 10으로 value를 바꾸어라)
  // 수행 후 comparand는 원래 value인 3로 바뀐다.
  bool exchanged = intAtomic.compare_exchange_weak(comparand, newValue);
  // 값 비교 교환 : 일반 함수
  // 앞서 comparand가 3로 바뀌었기에, 값이 10으로 바뀐다
  exchanged = atomic_compare_exchange_weak(&intAtomic, &comparand, newValue);

}