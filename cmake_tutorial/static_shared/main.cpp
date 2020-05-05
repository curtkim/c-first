#include <stdio.h>
#include <stdlib.h>

#include "MathFunctions.h"

int main (int argc, char *argv[])
{
  double inputValue = 4.0;  
  double outputValue = mathfunctions::sqrt(inputValue);

  fprintf(stdout,"The square root of %g is %g\n",
          inputValue, outputValue);
  return 0;
}