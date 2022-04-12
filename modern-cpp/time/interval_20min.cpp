// https://akrzemi1.wordpress.com/2022/04/11/using-stdchrono/
#define FMT_HEADER_ONLY
#include <fmt/chrono.h>
#include <chrono>
#include <iostream>

namespace t = std::chrono;

int main() {
	using TwentyMins = t::duration<int, std::ratio<20*60>>;
	t::time_point p1 = t::floor<TwentyMins>(t::system_clock::now());
	std::cout << fmt::format("{:%Y-%m-%d %H:%M}\n", p1);
}
