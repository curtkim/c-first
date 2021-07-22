#include <iostream>
#include <fstream>
#include <string>
#include "sample.pb.h"

int main() {

    SearchRequest req;
    std::ifstream ifs("search_query.data");
    if( !req.ParseFromIstream(&ifs)) {
        std::cout << "failed" << std::endl;
    }

    std::cout << req.query() << std::endl;

    if( req.has_user())
        std::cout << req.user().name() << std::endl;
    else
        std::cout << "no user\n";

    return 0;
}