#include "guardshift.hpp"
#include <iostream>

int main() {
    guardshift::Index index({95, 60}, {{{0, 100, 1000}, {1, 0, 70}}});
    for (const auto& edge : index.apply({{0, 105}, {1, 65}})) {
        std::cout << "guard " << edge.guard << (edge.active ? " entered\n" : " exited\n");
    }
}
