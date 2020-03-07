#include <algorithm>

#include "vector_inner.h"

// ----------------
// Regular C++ code
// ----------------

// multiply all entries by 2.0
// input:  std::vector ([...]) (read only)
// output: std::vector ([...]) (new copy)
std::vector<double> modify(const std::vector<double>& input)
{
    std::vector<double> output;

    std::transform(
            input.begin(),
            input.end(),
            std::back_inserter(output),
            [](double x) -> double { return 2.*x; }
    );

    // N.B. this is equivalent to (but there are also other ways to do the same)
    //
    // std::vector<double> output(input.size());
    //
    // for ( size_t i = 0 ; i < input.size() ; ++i )
    //   output[i] = 2. * input[i];

    return output;
}

