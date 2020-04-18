#include <string>
#include <thread>
#include <chrono>
#include <memory>

#include <rxcpp/rx.hpp>

#include <rxterm/terminal.hpp>
#include <rxterm/style.hpp>
#include <rxterm/image.hpp>
#include <rxterm/reflow.hpp>
#include <rxterm/components/text.hpp>
#include <rxterm/components/stacklayout.hpp>
#include <rxterm/components/flowlayout.hpp>
#include <rxterm/components/progress.hpp>
#include <rxterm/components/maxwidth.hpp>

using namespace rxterm;

namespace rx {
using namespace rxcpp;
using namespace rxcpp::sources;
using namespace rxcpp::operators;
using namespace rxcpp::util;
}
using namespace rx;

unsigned const TerminalWidth = 80;

int main() {

  auto progressbar = interval(std::chrono::milliseconds(250))
    | take_while([](long x) { return x < 100; })
    | rx::map([](auto x) { return x / 100.0;})
    | rx::map([](auto x) { return Progress(x);});

  /*
  auto app = zipWith(
      rx::observable<>::just<Text>("progressbar example"),
      progressbar,
      StackLayout);
  */

  auto a = progressbar
      | rx::scan(
        VirtualTerminal(),
        [](auto const& vt, Component const& c) {
          return vt.flip(c.render(TerminalWidth).toString());
        });

  a.subscribe();

  /*
  app.scan(
      VirtualTerminal(),
      [](auto const& vt, Component const& c) {
        return vt.flip(renderToTerm(vt, TerminalWidth, c));
      });
      */
}