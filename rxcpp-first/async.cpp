#include <rxcpp/rx.hpp>

using namespace std;

// 300ms sleep 하지만, multithread를 사용해서 전체가 300ms 정도에서 끝난다.
int main() {

  cout << "thread " << this_thread::get_id() << endl;
  chrono::steady_clock::time_point begin = chrono::steady_clock::now();

  rxcpp::serialize_one_worker s1 = rxcpp::serialize_new_thread();
  rxcpp::observe_on_one_worker o1 = rxcpp::observe_on_new_thread();

  rxcpp::sources::range(1, 5)
    .map([o1](int v) {
      return rxcpp::sources::just(v * 2)
        .tap([](int v) {
          cout << v << " " << this_thread::get_id() << endl; // 새로운 thread에서 실행된다.
          this_thread::sleep_for(chrono::milliseconds(300));
        })
        .subscribe_on(o1);
    })
    .flat_map([](auto observable) { return observable; })
    .observe_on(o1)
    .subscribe(
      [](int v) {
        cout << v << " " << this_thread::get_id() << endl;
      },
      [begin]() {
        chrono::steady_clock::time_point end = chrono::steady_clock::now();
        cout << "Time difference = " << chrono::duration_cast<chrono::milliseconds>(end - begin).count()
                  << "[ms]" << endl;
      });

  this_thread::sleep_for(chrono::seconds(2));

  return 0;
}