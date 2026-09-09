#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace khoj::bench {

struct FloatVectors {
    std::size_t count = 0;
    std::size_t dimension = 0;
    std::vector<float> data;

    const float* at(std::size_t index) const { return data.data() + index * dimension; }
};

struct IntVectors {
    std::size_t count = 0;
    std::size_t dimension = 0;
    std::vector<std::int32_t> data;

    const std::int32_t* at(std::size_t index) const { return data.data() + index * dimension; }
};

FloatVectors read_fvecs(const std::string& path, std::size_t max_count);
IntVectors read_ivecs(const std::string& path, std::size_t max_count);

}
