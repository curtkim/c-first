#include "spdlog/spdlog.h"
#include "spdlog/sinks/stdout_color_sinks.h"

void stdout_example()
{
  // logger_name으로 동록하고, spdlog::get(logger_name)인것 같다.
  auto console = spdlog::stdout_color_mt("console");
  auto err_logger = spdlog::stderr_color_mt("stderr");

  spdlog::get("console")->info("loggers can be retrieved from a global registry using the spdlog::get(logger_name)");
}

int main() {
  stdout_example();
}