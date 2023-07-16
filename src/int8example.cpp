#include "engine.h"

int main() {
    if (!Util::doesFileExist("/home/cyrus/work/data/ffhq/67000/")) {
        std::cout << "No bueno!" << std::endl;
    }
}