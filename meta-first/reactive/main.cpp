
// Standard library
#include <iostream>
#include <string>

// JSON library
#include <nlohmann/json.hpp>
using json = nlohmann::json;

// Utilities
#include "expected.h"
#include "mtry.h"
#include "trim.h"

// Our reactive stream implementation
#include "filter.h"
#include "join.h"
#include "sink.h"
#include "transform.h"
#include "values.h"

// Service implementation
#include "service.h"

/**
 * For funcitons that return json objects, but which can fail
 * and return an exception
 */
using expected_json = expected<json, std::exception_ptr>;

/**
 * Basic bookmark information. Holds an URL and a title
 */
struct bookmark_t {
    std::string url;
    std::string text;
};

/**
 * Creates a string from the bookmark info in the following format:
 * [text](url)
 */
std::string to_string(const bookmark_t& page)
{
    return "[" + page.text + "](" + page.url + ")";
}

/**
 * Writes the bookmark info in the following format:
 * [text](url)
 */
std::ostream& operator<<(std::ostream& out, const bookmark_t& page)
{
    return out << "[" << page.text << "](" << page.url << ")";
}

/**
 * Type that contains a bookmark or an error (an exception)
 */
using expected_bookmark = expected<bookmark_t, std::exception_ptr>;

/**
 * Tries to read the bookmark info from JSON. In the case
 * of failure, it will return an exception.
 */
expected_bookmark bookmark_from_json(const json& data)
{
    return mtry([&] {
            return bookmark_t { data.at("FirstURL"), data.at("Text") };
        });
}


int main(int argc, char *argv[])
{
    using namespace reactive::operators;

    // We are lifting the transform and filter functions
    // to work on the with_client<T> type that adds the
    // socket information to the given value so that we
    // can reply to the client
    auto transform = [](auto f) {
            return reactive::operators::transform(lift_with_client(f));
        };
    auto filter = [](auto f) {
            return reactive::operators::filter(apply_with_client(f));
        };

    asio::io_service event_loop;

    auto pipeline =
        service(event_loop)
            | transform(trim)

            // Ignoring comments and empty messages
            | filter([] (const std::string &message) {
                return message.length() > 0 && message[0] != '#';
            })

            // Trying to parse the input
            | transform([] (const std::string &message) {
                return mtry([&] {
                    return json::parse(message);
                });
            })

            // Converting the result into the bookmark
            | transform([] (const auto& exp) {
                return mbind(exp, bookmark_from_json);
            })

            | sink([] (const auto &message) {
                const auto exp_bookmark = message.value;

                if (!exp_bookmark) {
                    message.reply("ERROR: Request not understood\n");
                    return;
                }

                if (exp_bookmark->text.find("C++") != std::string::npos) {
                    message.reply("OK: " + to_string(exp_bookmark.get()) + "\n");
                } else {
                    message.reply("ERROR: Not a C++-related link\n");
                }
            });

    // Starting the Boost.ASIO service
    std::cerr << "Service is running...\n";
    event_loop.run();
}
