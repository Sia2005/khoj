#pragma once

#include <cstddef>

namespace khoj {
namespace detail {

constexpr std::size_t lane_count = 8;

inline float squared_l2(const float* __restrict left,
                        const float* __restrict right,
                        std::size_t dimension) {
    float lanes[lane_count] = {};
    std::size_t index = 0;

    for (; index + lane_count <= dimension; index += lane_count) {
        for (std::size_t lane = 0; lane < lane_count; ++lane) {
            const float difference = left[index + lane] - right[index + lane];
            lanes[lane] += difference * difference;
        }
    }

    float total = 0.0f;
    for (std::size_t lane = 0; lane < lane_count; ++lane) {
        total += lanes[lane];
    }

    for (; index < dimension; ++index) {
        const float difference = left[index] - right[index];
        total += difference * difference;
    }

    return total;
}

inline float inner_product(const float* __restrict left,
                           const float* __restrict right,
                           std::size_t dimension) {
    float lanes[lane_count] = {};
    std::size_t index = 0;

    for (; index + lane_count <= dimension; index += lane_count) {
        for (std::size_t lane = 0; lane < lane_count; ++lane) {
            lanes[lane] += left[index + lane] * right[index + lane];
        }
    }

    float total = 0.0f;
    for (std::size_t lane = 0; lane < lane_count; ++lane) {
        total += lanes[lane];
    }

    for (; index < dimension; ++index) {
        total += left[index] * right[index];
    }

    return total;
}

}
}
