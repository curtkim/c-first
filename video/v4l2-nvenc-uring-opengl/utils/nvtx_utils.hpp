#pragma once

template<typename Callable>
void nvtxRange(const char* range_name, Callable body) {
    nvtxRangePush(range_name);
    body();
    nvtxRangePop();
}
