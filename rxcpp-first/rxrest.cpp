#include <rxcpp/rx.hpp>

using namespace std;

#include <cpprest/http_client.h>

using namespace web;                        // Common features like URIs.
using namespace web::http;                  // Common HTTP functionality
using namespace web::http::client;          // HTTP client features
using namespace concurrency::streams;       // Asynchronous streams


auto get(http_client client, string url) {
  //return rxcpp::sources::just(url);

  return rxcpp::sources::create<string>(
    [&](rxcpp::subscriber<string> s){
      pplx::task<web::http::http_response> resp = client.request(methods::GET, url);

      concurrency::streams::stringstreambuf sbuffer;
      pplx::task<void> requestTask = resp.then([=](http_response response) {
          cout << 2 << " " << this_thread::get_id() << endl;
          cout << response.status_code() << std::endl;
          for(auto h : response.headers())
            printf("%s : %s \n", h.first.c_str(), h.second.c_str());

          return response.body().read_to_end(sbuffer);
        })
        .then([=](size_t a){
          s.on_next(sbuffer.collection().c_str());
          s.on_completed();
          return;
        });

      requestTask.wait();
    });
}

int main() {

  chrono::steady_clock::time_point begin = chrono::steady_clock::now();

  //rxcpp::serialize_one_worker s1 = rxcpp::serialize_new_thread();
  //rxcpp::observe_on_one_worker o1 = rxcpp::observe_on_new_thread();

  http_client client("http://dapi.kakao.com");

  rxcpp::sources::range(1, 5)
    .map([client](int v) {
      return get(client, "/region-code/v2/hcode.json?x=501200&y=1121569&inputCoordSystem=WCONGNAMUL&outputCoordSystem=WCONGNAMUL&format=simple");
    })
    .flat_map([](auto observable) { return observable; })
    //.observe_on(s1)
    .subscribe(
      [](string v) {
        cout << v << " " << this_thread::get_id() << endl;
      },
      [begin]() {
        chrono::steady_clock::time_point end = chrono::steady_clock::now();
        cout << "Time difference = " << chrono::duration_cast<chrono::milliseconds>(end - begin).count()
             << "[ms]" << endl;
      });

  this_thread::sleep_for(chrono::seconds(3));
  return 0;
}
