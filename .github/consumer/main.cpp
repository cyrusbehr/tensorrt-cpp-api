#include <cstdio>

#include <tensorrt_cpp_api/version.h>

int main() {
    std::printf("consumer linked trtcpp %s\n", trtcpp::versionString().c_str());
    return trtcpp::libraryVersion().major == 7 ? 0 : 1;
}
