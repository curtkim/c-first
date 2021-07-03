#include <chrono>
#include <string>
//#include <source_location>
#include <experimental/source_location>
#include <iostream>
#include <format>

enum class log_level : char
{
   Info = 'I',
   Warning = 'W',
   Error = 'E'
};

auto as_local(std::chrono::system_clock::time_point const tp)
{
   return std::chrono::zoned_time{ std::chrono::current_zone(), tp };
}

std::string to_string(auto tp)
{
   return std::format("{:%F %T %Z}", tp);
}


std::string to_string(std::source_location const source)
{
   return std::format("{}:{}:{}", 
      std::filesystem::path(source.file_name()).filename().string(),
      source.function_name(),
      source.line());
}


void log(log_level const level, 
         std::string_view const message, 
         std::source_location const source = std::source_location::current())
{
   std::cout
      << std::format("[{}] {} | {} | {}", 
                     static_cast<char>(level), 
                     to_string(as_local(std::chrono::system_clock::now())), 
                     to_string(source), 
                     message)
      << '\n';
}

void execute(int, double)
{
   log(log_level::Error, "Error in execute!");
}

int main()
{
   log(log_level::Info, "Logging from main!");
   execute(0, 0);
}

