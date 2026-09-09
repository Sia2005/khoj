#include "khoj/flat_index.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

namespace khoj {

namespace {

constexpr std::size_t lane_count = 8;

float squared_l2(const float* __restrict left,
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

float inner_product(const float* __restrict left,
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

bool ranks_before(const SearchResult& left, const SearchResult& right, Metric metric) {
    if (left.distance != right.distance) {
        return metric == Metric::L2 ? left.distance < right.distance
                                    : left.distance > right.distance;
    }
    return left.label < right.label;
}

std::string size_mismatch_message(const std::string& operation,
                                  std::size_t received,
                                  std::size_t expected) {
    return "khoj::FlatIndex::" + operation + " received " + std::to_string(received) +
           " values, expected " + std::to_string(expected);
}

}

FlatIndex::FlatIndex(std::size_t dimension, Metric metric)
    : dimension_(dimension), metric_(metric) {
    if (dimension_ == 0) {
        throw std::invalid_argument("khoj::FlatIndex requires a non-zero dimension");
    }
}

void FlatIndex::reserve(std::size_t count) {
    data_.reserve(count * dimension_);
}

std::uint64_t FlatIndex::add(const std::vector<float>& embedding) {
    if (embedding.size() != dimension_) {
        throw std::invalid_argument(size_mismatch_message("add", embedding.size(), dimension_));
    }

    const std::uint64_t id = static_cast<std::uint64_t>(size());
    data_.insert(data_.end(), embedding.begin(), embedding.end());
    return id;
}

void FlatIndex::add_batch(const std::vector<float>& contiguous_vectors) {
    if (contiguous_vectors.size() % dimension_ != 0) {
        throw std::invalid_argument(
            size_mismatch_message("add_batch", contiguous_vectors.size(), dimension_));
    }

    data_.insert(data_.end(), contiguous_vectors.begin(), contiguous_vectors.end());
}

std::vector<SearchResult> FlatIndex::search(const std::vector<float>& query, std::size_t k) const {
    if (query.size() != dimension_) {
        throw std::invalid_argument(size_mismatch_message("search", query.size(), dimension_));
    }

    const std::size_t count = size();
    const std::size_t result_count = std::min(k, count);

    std::vector<SearchResult> results;
    if (result_count == 0) {
        return results;
    }
    results.reserve(result_count);

    const auto ordering = [this](const SearchResult& left, const SearchResult& right) {
        return ranks_before(left, right, metric_);
    };

    const float* const query_data = query.data();
    const float* const base = data_.data();

    for (std::size_t position = 0; position < count; ++position) {
        const float* const candidate = base + position * dimension_;
        const float score = metric_ == Metric::L2
                                ? squared_l2(query_data, candidate, dimension_)
                                : inner_product(query_data, candidate, dimension_);
        const SearchResult result{static_cast<std::uint64_t>(position), score};

        if (results.size() < result_count) {
            results.push_back(result);
            std::push_heap(results.begin(), results.end(), ordering);
        } else if (ranks_before(result, results.front(), metric_)) {
            std::pop_heap(results.begin(), results.end(), ordering);
            results.back() = result;
            std::push_heap(results.begin(), results.end(), ordering);
        }
    }

    std::sort_heap(results.begin(), results.end(), ordering);

    if (metric_ == Metric::L2) {
        for (SearchResult& result : results) {
            result.distance = std::sqrt(result.distance);
        }
    }

    return results;
}

std::vector<std::vector<SearchResult>> FlatIndex::search_batch(const std::vector<float>& queries,
                                                               std::size_t k) const {
    if (queries.size() % dimension_ != 0) {
        throw std::invalid_argument(
            size_mismatch_message("search_batch", queries.size(), dimension_));
    }

    const std::size_t query_count = queries.size() / dimension_;
    std::vector<std::vector<SearchResult>> results;
    results.reserve(query_count);

    std::vector<float> query(dimension_);
    for (std::size_t position = 0; position < query_count; ++position) {
        const auto begin = queries.begin() + static_cast<std::ptrdiff_t>(position * dimension_);
        query.assign(begin, begin + static_cast<std::ptrdiff_t>(dimension_));
        results.push_back(search(query, k));
    }

    return results;
}

std::size_t FlatIndex::size() const {
    return data_.size() / dimension_;
}

std::size_t FlatIndex::dimension() const {
    return dimension_;
}

Metric FlatIndex::metric() const {
    return metric_;
}

const std::vector<float>& FlatIndex::data() const {
    return data_;
}

}
