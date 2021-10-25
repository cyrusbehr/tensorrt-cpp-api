#include "engine.h"

int main() {
    Options options;
    Engine engine(options);

    const std::string onnxModelpath = "../arcfaceresnet100-8.onnx";

    bool succ = engine.build(onnxModelpath);
    if (!succ) {
        throw std::runtime_error("Unable to build TRT engine.");
    }

    return 0;
}
