#include <iostream>
#include "Gregorian.h"

int main()
{
    GregorianDate gregDate;
    gregDate.SetMonth(12);
    gregDate.SetDay(2);
    gregDate.SetYear(2020);

    std::cout << gregDate.getAbsoluteDate() << std::endl;
    return 0;
}