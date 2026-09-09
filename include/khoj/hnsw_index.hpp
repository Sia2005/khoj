#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "khoj/types.hpp"

namespace khoj {

struct HnswParams {
    std::size_t dimension = 0;
    std::size_t max_neighbors = 16;
    std::size_t ef_construction = 200;
    std::uint64_t seed = 0;
};

class HnswIndex {
public:
    explicit HnswIndex(const HnswParams& params);
    ~HnswIndex();

    HnswIndex(const HnswIndex&) = delete;
    HnswIndex& operator=(const HnswIndex&) = delete;
    HnswIndex(HnswIndex&& other) noexcept;
    HnswIndex& operator=(HnswIndex&& other) noexcept;

    void build(const std::vector<float>& contiguous_vectors);
    VectorId add(const std::vector<float>& embedding);
    std::vector<SearchResult> search(const std::vector<float>& query,
                                     std::size_t k,
                                     std::size_t ef_search) const;

    void save(const std::string& path) const;
    static HnswIndex load(const std::string& path);

    std::size_t size() const;
    const HnswParams& params() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}
