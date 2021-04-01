// https://github.com/fschuetz04/simcpp20
#include <coroutine>
#include <iostream>

#include "31_simcpp.hpp"

simcpp20::process clock_proc(simcpp20::simulation &sim, std::string name,
                             double delay) {
  while (true) {
    std::cout << name << " " << sim.now() << std::endl;
    co_await sim.timeout(delay);
  }
}

int main() {
  simcpp20::simulation sim;
  clock_proc(sim, "fast", 1);
  clock_proc(sim, "slow", 2);
  sim.run_until(5);
}