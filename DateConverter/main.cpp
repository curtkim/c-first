#include <iostream>
#include "Gregorian.h"
#include "Julian.h"

int main()
{
    GregorianDate gregDate;
    gregDate.SetMonth(12);
    gregDate.SetDay(2);
    gregDate.SetYear(2020);

    JulianDate julDate;

    julDate.CalcJulianDate(gregDate.getAbsoluteDate());

    std::cout << gregDate.getAbsoluteDate() << std::endl;
    std::cout << julDate.getYear() << julDate.getMonth() << julDate.getDay() << std::endl;
    
    return 0;
}