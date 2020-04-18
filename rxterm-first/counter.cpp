#include <string>
#include <thread>
#include <chrono>
#include <memory>

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

using namespace std::chrono_literals;
using namespace std::string_literals;


unsigned const TerminalWidth = 80;

auto renderToTerm = [](auto const& vt, unsigned const w, Component const& c) {
  // TODO: get actual terminal width
  return vt.flip(c.render(w).toString());
};


auto fancy_counter(int start, int end, int i) {
  return FlowLayout<> {
      Text("counting from "),
      Text({ Font::Bold }, start),
      Text(" to "),
      Text({ Font::Bold }, end),
      Text(": i="),
      Text({ FontColor::Red, Font::Underline }, i),
      MaxWidth(20, Progress(i * 0.01)),
  };
}


int main() {
  VirtualTerminal vt;

  /*
  auto const fancyCounter = [](auto start, auto end, auto i) -> FlowLayout<> {
    return {
        Text("counting from "), Text({ Font::Bold }, start),
        Text(" to "), Text({ Font::Bold }, end),
        Text(": i="), Text({ FontColor::Red, Font::Underline }, i),
        MaxWidth(20, Progress(i * 0.01)),
    };
  };
  */

  for (int i = 0; i < 101; ++i) {
    vt = renderToTerm(vt, TerminalWidth, fancy_counter(0, 100, i)); // Render to terminal
    std::this_thread::sleep_for(10ms);
  }

  return 0;
}