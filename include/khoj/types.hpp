#pragma once

#include <cstdint>

namespace khoj {

using VectorId = std::uint64_t;

enum class Metric {
    L2,
    InnerProduct
};

struct SearchResult {
    VectorId id = 0;
    float distance = 0.0f;
};

}
