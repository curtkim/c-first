#include <rxcpp/rx.hpp>

using namespace std;

// 300ms sleep 하지만, multithread를 사용해서 전체가 300ms 정도에서 끝난다.
// cpu개수보다 많은 경우에도 newthread는 계속 thread를 만든다.
// pool은 어떻게 구현해야 하나?
//
// observe_on_event_loop를 사용하면 cpu개수에 bound된다.
int main() {

  cout << "main thread : " << this_thread::get_id() << endl;
  chrono::steady_clock::time_point start_time = chrono::steady_clock::now();

  //rxcpp::identity_one_worker i1 = rxcpp::identity_current_thread();
  rxcpp::serialize_one_worker s1 = rxcpp::serialize_new_thread();
  //rxcpp::serialize_one_worker s2 = rxcpp::serialize_new_thread();
  //rxcpp::observe_on_one_worker o1 = rxcpp::observe_on_new_thread();
  auto o1 = rxcpp::observe_on_event_loop();


  rxcpp::sources::range(1, 20)
    .map([o1](int v) {
      return rxcpp::sources::just(v)
        .map([](int v) {
          cout << "map " << v << " " << this_thread::get_id() << endl; // 새로운 thread에서 실행된다.
          this_thread::sleep_for(chrono::milliseconds(300));
          return v;
        })
        .subscribe_on(o1);
    })
    .flat_map([](auto observable) { return observable; })
    .observe_on(s1)
    .subscribe(
      [](int v) {
        cout << "sub " << v << " " << this_thread::get_id() << endl;
      },
      [start_time]() {
        long diff = chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now() - start_time).count();
        cout << "Time difference = " << diff << "[ms]" << endl;
      });

  this_thread::sleep_for(chrono::seconds(1));

  return 0;
}