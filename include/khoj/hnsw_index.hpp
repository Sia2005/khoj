#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "khoj/types.hpp"
#include "khoj/visited_list_pool.hpp"

namespace khoj {

struct HnswParams {
    std::size_t dimension = 0;
    std::size_t max_neighbors = 16;
    std::size_t max_neighbors_layer0 = 32;
    std::size_t ef_construction = 200;
    double level_multiplier = 0.0;
    std::uint64_t seed = 42;
    Metric metric = Metric::L2;
};

class HnswIndex {
public:
    explicit HnswIndex(const HnswParams& params);

    void reserve(std::size_t expected_elements);
    void add(std::uint64_t label, const float* vector);

    std::vector<SearchResult> search(const float* query, std::size_t k, std::size_t ef_search) const;

    std::size_t size() const { return element_count_; }
    std::size_t dimension() const { return params_.dimension; }
    std::size_t max_level() const { return max_level_; }
    const HnswParams& params() const { return params_; }

    void save(const std::string& path) const;
    static HnswIndex load(const std::string& path);

private:
    using InternalId = std::uint32_t;

    struct Candidate {
        float distance;
        InternalId id;
    };

    float distance_to_stored(const float* query, InternalId target) const;
    float distance_between(InternalId left, InternalId right) const;
    const float* vector_at(InternalId id) const;

    std::size_t draw_level();

    InternalId greedy_descend(const float* query, InternalId entry, std::size_t layer) const;

    std::vector<Candidate> search_layer(const float* query,
                                        const std::vector<InternalId>& entry_points,
                                        std::size_t ef,
                                        std::size_t layer) const;

    std::vector<InternalId> select_neighbors_heuristic(const float* base,
                                                       std::vector<Candidate> candidates,
                                                       std::size_t limit) const;

    std::vector<InternalId> select_neighbors_heuristic_for(InternalId base,
                                                           std::vector<Candidate> candidates,
                                                           std::size_t limit) const;

    InternalId* neighbors_mutable(InternalId id, std::size_t layer);
    const InternalId* neighbors(InternalId id, std::size_t layer) const;
    std::uint32_t degree(InternalId id, std::size_t layer) const;
    void set_degree(InternalId id, std::size_t layer, std::uint32_t value);
    std::size_t capacity_at(std::size_t layer) const;

    void connect(InternalId from, const std::vector<InternalId>& targets, std::size_t layer);
    void prune_if_overfull(InternalId node, std::size_t layer);

    HnswParams params_;
    double resolved_level_multiplier_ = 0.0;

    std::size_t element_count_ = 0;
    std::size_t max_level_ = 0;
    InternalId entry_point_ = 0;
    bool has_entry_point_ = false;

    std::vector<float> vectors_;
    std::vector<std::uint64_t> labels_;
    std::vector<std::uint32_t> levels_;

    std::vector<InternalId> links_layer0_;
    std::vector<std::uint32_t> degrees_layer0_;

    std::vector<std::vector<InternalId>> links_upper_;
    std::vector<std::vector<std::uint32_t>> degrees_upper_;

    std::uint64_t rng_state_ = 0;

    mutable VisitedListPool visited_pool_;
};

}
