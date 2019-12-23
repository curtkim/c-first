#include <cpprest/http_client.h>
#include <iostream>
#include <thread>


using namespace utility;                    // Common utilities like string conversions
using namespace web;                        // Common features like URIs.
using namespace web::http;                  // Common HTTP functionality
using namespace web::http::client;          // HTTP client features
using namespace concurrency::streams;       // Asynchronous streams

using namespace std;

int main() {

  http_client client(U("http://dapi.kakao.com"));

  concurrency::streams::stringstreambuf sbuffer;

  pplx::task<web::http::http_response> resp = client.request(methods::GET, "/region-code/v2/hcode.json?x=501200&y=1121569&inputCoordSystem=WCONGNAMUL&outputCoordSystem=WCONGNAMUL&format=simple");

  pplx::task<void> requestTask = resp.then([=](http_response response) {
    cout << 2 << " " << this_thread::get_id() << endl;
    cout << response.status_code() << std::endl;
    for(auto h : response.headers())
      printf("%s : %s \n", h.first.c_str(), h.second.c_str());

    return response.body().read_to_end(sbuffer);
  })
  .then([=](size_t a){
    cout << "body: " << a << " " << sbuffer.collection().c_str() << endl;
    return;
  });

  try {
    cout << 1 << " " << this_thread::get_id() << endl;
    requestTask.wait();
  }
  catch (const std::exception &e) {
    printf("Error exception:%s\n", e.what());
  }

  return 0;
}
