#include <iostream>
#include <fstream>
#include <string>
#include "sample.pb.h"

int main() {
    SearchRequest req;
    req.set_query("Alex");
    req.set_page_number(1);
    req.set_result_per_page(10);

    std::cout << req.query() << "\n";

    std::ofstream ofs("search_query.data");
    req.SerializeToOstream(&ofs);

    return 0;
}