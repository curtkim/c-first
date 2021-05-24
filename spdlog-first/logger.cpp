#include "spdlog/spdlog.h"
#include "spdlog/sinks/stdout_color_sinks.h"

void stdout_example()
{
    auto logger = spdlog::get("console");
    logger->info("loggers can be retrieved from a global registry using the spdlog::get(logger_name)");
}

int main() {
    // logger_name으로 동록하고, spdlog::get(logger_name)인것 같다.
    auto console = spdlog::stdout_color_mt("console");          // _mt는 multithread
    //auto err_logger = spdlog::stderr_color_mt("stderr");

    stdout_example();
}