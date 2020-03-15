#include <algorithm>
#include <chrono>
#include <iostream>
#include <vector>
#include <set>
#include <random>

constexpr int RUN_TIMES = 5;

template <typename TFunc> void RunAndMeasure(const char* title, TFunc func)
{
    std::set<double> times;
    std::vector results(RUN_TIMES, func()); // invoke it for the first time...
    for (int i = 0; i < RUN_TIMES; ++i)
    {
        const auto start = std::chrono::steady_clock::now();
        const auto ret = func();
        const auto end = std::chrono::steady_clock::now();
        results[i] = ret;
        times.insert(std::chrono::duration<double, std::milli>(end - start).count());
    }

    std::cout << title << ":\t " << *times.begin() << "ms (max was " << *times.rbegin() << ") " << results[0] << '\n';
}

float GenRandomFloat(float lower, float upper)
{
    // usage of thread local random engines allows running the generator in concurrent mode
    thread_local static std::default_random_engine rd;
    std::uniform_real_distribution<float> dist(lower, upper);
    return dist(rd);
}

void TestDoubleValue(const size_t vecSize)
{
    std::vector<double> vec(vecSize, 0.5);
    std::generate(vec.begin(), vec.end(), []() { return GenRandomFloat(-1.0f, 1.0f); });
    std::vector out(vec);

    std::cout << "v*2:\n";

    RunAndMeasure("std::transform   ", [&vec, &out] {
        std::transform(vec.begin(), vec.end(), out.begin(),
                       [](double v) {
                           return v * 2.0;
                       }
        );
        return out[0];
    });

    RunAndMeasure("omp parallel for", [&vec, &out] {
#pragma omp parallel for
        for (int i = 0; i < static_cast<int>(vec.size()); ++i) //  'i': index variable in OpenMP 'for' statement must have signed integral type
            out[i] = vec[i]*2.0;

        return out[0];
    });

    //RunAndMeasure("using raw loop  ", [&vec, &out] {
    //	for (int i = 0; i < static_cast<int>(vec.size()); ++i) //  'i': index variable in OpenMP 'for' statement must have signed integral type
    //		out[i] = vec[i] * 2.0;

    //	return out.size();
    //});
}


int main(int argc, char* argv[])
{
#ifdef _DEBUG
    const size_t vecSize = argc > 1 ? atoi(argv[1]) : 10000;
#else
    const size_t vecSize = argc > 1 ? atoi(argv[1]) : 6000000;
#endif
    std::cout << vecSize << '\n';
    std::cout << "Running each test " << RUN_TIMES << " times\n";

    int step = argc > 2 ? atoi(argv[2]) : 0;

    if (step == 0 || step == 1)
        TestDoubleValue(vecSize);

    return 0;
}