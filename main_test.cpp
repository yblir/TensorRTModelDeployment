#include <iostream>
#include <typeinfo>
#include "utils/general.h"

#include <vector>
#include <filesystem>

int main() {
    Timer timer = Timer();
    timer.timeStart();
    auto t1 = Timer::curTimePoint();
    for (int i = 0; i < 10; ++i) {
        std::cout << "Hello, World!" << std::endl;
    }

    std::cout << timer.timeCount() << " ms" << std::endl;
    for (int i = 0; i < 20; ++i) {
        std::cout << "Hello, World!!!!!!" << std::endl;
    }
    std::cout << timer.timeCount(t1) << " ms" << std::endl;

    return 0;
}
