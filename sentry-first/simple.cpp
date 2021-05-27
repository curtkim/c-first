#include "sentry.h"
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef NDEBUG
#    undef NDEBUG
#endif

#include <assert.h>

#ifdef SENTRY_PLATFORM_WINDOWS
#    include <synchapi.h>
#    define sleep_s(SECONDS) Sleep((SECONDS)*1000)
#else

#    include <signal.h>
#    include <unistd.h>

#    define sleep_s(SECONDS) sleep(SECONDS)
#endif


int main() {

    sentry_options_t *options = sentry_options_new();

    if(const char* dsn = std::getenv("SENTRY_DSN")) {
        printf("dsn = %s", dsn);
        sentry_options_set_dsn(options, dsn);
    }
    else {
        std::exit(1);
    }

    sentry_init(options);

    sentry_value_t user = sentry_value_new_object();
    sentry_value_set_by_key(user, "id", sentry_value_new_int32(1234));
    sentry_value_set_by_key(user, "username", sentry_value_new_string("car1"));
    sentry_set_user(user);

//    sentry_capture_event(sentry_value_new_message_event(
//            /*   level */ SENTRY_LEVEL_INFO,
//            /*  logger */ "custom",
//            /* message */ "It works! 2"
//    ));

    sentry_capture_event(sentry_value_new_message_event(
            /*   level */ SENTRY_LEVEL_WARNING,
            /*  logger */ "topic monitoring",
            /* message */ "센서 데이터가 3초 이상 들어오지 않았습니다"
    ));

    sentry_close();
}