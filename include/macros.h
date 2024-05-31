#pragma once

#include <spdlog/spdlog.h>

#define CHECK(condition)                                                                                                                   \
    do {                                                                                                                                   \
        if (!(condition)) {                                                                                                                \
            spdlog::error("Assertion failed: ({}), function {}, file {}, line {}.", #condition, __FUNCTION__, __FILE__, __LINE__);         \
            abort();                                                                                                                       \
        }                                                                                                                                  \
    } while (false);
