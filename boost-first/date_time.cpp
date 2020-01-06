#include <boost/date_time/gregorian/gregorian.hpp>

#include <iostream>

int main() {

    boost::gregorian::date dateObj{2016, 3, 21};
    std::cout << "Date = " << dateObj << '\n';

    // Initializing Date object with today's in Local Time Zone
    boost::gregorian::date localToday =
            boost::gregorian::day_clock::local_day();

    // Printing complete Date object on console
    std::cout << "Date in System's Time Zone= " << localToday << '\n';

    // Initializing Date object with today's in UTC Time Zone
    boost::gregorian::date utcToday =
            boost::gregorian::day_clock::universal_day();

    // Printing complete Date object on console
    std::cout << "Date in UTC Time Zone = " << utcToday << '\n';

    return 0;
}
