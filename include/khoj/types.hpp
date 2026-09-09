#pragma once

#include <cstdint>

namespace khoj {

enum class Metric {
    L2,
    InnerProduct
};

struct SearchResult {
    std::uint64_t label;
    float distance;
};

}
