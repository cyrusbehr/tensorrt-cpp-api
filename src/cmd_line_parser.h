#pragma once
#include "engine.h"
#include <iostream>

struct CommandLineArguments {
    std::string onnxModelPath = "";
    std::string trtModelPath = "";
};

inline void showHelp(char *argv[]) {
    std::cout << "Usage: " << argv[0] << " [OPTIONS]" << std::endl << std::endl;

    std::cout << "Options:" << std::endl;
    std::cout << "--onnx_model <string>             Path to the ONNX model. "
                 "(Either onnx_model or trt_model must be provided)"
              << std::endl;
    std::cout << "--trt_model <string>              Path to the TensorRT model. "
                 "(Either onnx_model or trt_model must be provided)"
              << std::endl;

    std::cout << "Example usage:" << std::endl;
    std::cout << argv[0] << " --onnx_model model.onnx" << std::endl;
};

inline bool tryGetNextArgument(int argc, char *argv[], int &currentIndex, std::string &value, std::string flag, bool printErrors = true) {
    if (currentIndex + 1 >= argc) {
        if (printErrors)
            std::cout << "Error: No arguments provided for flag '" << flag << "'" << std::endl;
        return false;
    }

    std::string nextArgument = argv[currentIndex + 1];
    if (nextArgument.substr(0, 2) == "--") {
        if (printErrors)
            std::cout << "Error: No arguments provided for flag '" << flag << "'" << std::endl;
        return false;
    }

    value = argv[++currentIndex];
    return true;
};

inline bool parseArguments(int argc, char *argv[], CommandLineArguments &arguments) {
    if (argc == 1) {
        showHelp(argv);
        return false;
    }

    for (int i = 1; i < argc; i++) {
        std::string argument = argv[i];

        if (argument.substr(0, 2) == "--") {
            std::string flag = argument.substr(2);
            std::string nextArgument;

            if (flag == "onnx_model") {
                if (!tryGetNextArgument(argc, argv, i, nextArgument, flag))
                    return false;

                if (!Util::doesFileExist(nextArgument)) {
                    std::cout << "Error: Unable to find model at path '" << nextArgument << "' for flag '" << flag << "'" << std::endl;
                    return false;
                }

                arguments.onnxModelPath = nextArgument;
            }

            else if (flag == "trt_model") {
                if (!tryGetNextArgument(argc, argv, i, nextArgument, flag))
                    return false;

                if (!Util::doesFileExist(nextArgument)) {
                    std::cout << "Error: Unable to find model at path '" << nextArgument << "' for flag '" << flag << "'" << std::endl;
                    return false;
                }

                arguments.trtModelPath = nextArgument;
            }

            else {
                std::cout << "Error: Unknown flag '" << flag << "'" << std::endl;
                showHelp(argv);
                return false;
            }
        } else {
            std::cout << "Error: Unknown argument '" << argument << "'" << std::endl;
            showHelp(argv);
            return false;
        }
    }

    if (arguments.onnxModelPath.empty() && arguments.trtModelPath.empty()) {
        std::cout << "Error: Must specify either 'onnx_model' or 'trt_model'" << std::endl;
        return false;
    }

    return true;
}
