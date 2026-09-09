#pragma once

#include <cstddef>
#include <vector>

#include "khoj/types.hpp"

namespace khoj {

class FlatIndex {
public:
    FlatIndex(std::size_t dimension, Metric metric);

    void reserve(std::size_t count);
    VectorId add(const std::vector<float>& embedding);
    void add_batch(const std::vector<float>& contiguous_vectors);

    std::vector<SearchResult> search(const std::vector<float>& query, std::size_t k) const;
    std::vector<std::vector<SearchResult>> search_batch(const std::vector<float>& queries,
                                                        std::size_t k) const;

    std::size_t size() const;
    std::size_t dimension() const;
    Metric metric() const;
    const std::vector<float>& data() const;

private:
    std::size_t dimension_;
    Metric metric_;
    std::vector<float> data_;
};

}
