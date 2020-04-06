#include "spdlog/spdlog.h"
#include "spdlog/sinks/basic_file_sink.h"
#include "spdlog/sinks/stdout_color_sinks.h"


void multi_sink_example()
{
  auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
  console_sink->set_level(spdlog::level::warn);
  console_sink->set_pattern("[multi_sink_example] [%^%l%$] %v");

  auto file_sink = std::make_shared<spdlog::sinks::basic_file_sink_mt>("logs/multisink.txt", true);
  file_sink->set_level(spdlog::level::trace);

  spdlog::logger logger("multi_sink", {console_sink, file_sink});
  logger.set_level(spdlog::level::debug);
  logger.warn("this should appear in both console and file");
  logger.info("this message should not appear in the console, only in the file");
}

int main() {
  multi_sink_example();
}