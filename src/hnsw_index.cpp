#include "khoj/hnsw_index.hpp"

#include "distance_kernels.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <queue>
#include <stdexcept>

namespace khoj {
namespace {

constexpr std::uint32_t kMagic = 0x4b484f4a;
constexpr double kMinUniform = 1e-12;

}

HnswIndex::HnswIndex(const HnswParams& params) : params_(params) {
    if (params_.dimension == 0) {
        throw std::invalid_argument("dimension must be positive");
    }
    if (params_.max_neighbors == 0 || params_.max_neighbors_layer0 == 0) {
        throw std::invalid_argument("neighbor limits must be positive");
    }
    if (params_.ef_construction < params_.max_neighbors) {
        throw std::invalid_argument("ef_construction must be at least max_neighbors");
    }

    resolved_level_multiplier_ = params_.level_multiplier > 0.0
        ? params_.level_multiplier
        : 1.0 / std::log(static_cast<double>(params_.max_neighbors));

    rng_state_ = params_.seed == 0 ? 0x9e3779b97f4a7c15ull : params_.seed;
}

void HnswIndex::reserve(std::size_t expected_elements) {
    vectors_.reserve(expected_elements * params_.dimension);
    labels_.reserve(expected_elements);
    levels_.reserve(expected_elements);
    links_layer0_.reserve(expected_elements * params_.max_neighbors_layer0);
    degrees_layer0_.reserve(expected_elements);
    links_upper_.reserve(expected_elements);
    degrees_upper_.reserve(expected_elements);
}

const float* HnswIndex::vector_at(InternalId id) const {
    return vectors_.data() + static_cast<std::size_t>(id) * params_.dimension;
}

float HnswIndex::distance_to_stored(const float* query, InternalId target) const {
    const float* stored = vector_at(target);
    return params_.metric == Metric::L2
        ? detail::squared_l2(query, stored, params_.dimension)
        : -detail::inner_product(query, stored, params_.dimension);
}

float HnswIndex::distance_between(InternalId left, InternalId right) const {
    return distance_to_stored(vector_at(left), right);
}

std::size_t HnswIndex::draw_level() {
    rng_state_ ^= rng_state_ << 13;
    rng_state_ ^= rng_state_ >> 7;
    rng_state_ ^= rng_state_ << 17;

    const double uniform = static_cast<double>(rng_state_ >> 11) / 9007199254740992.0;
    const double guarded = uniform < kMinUniform ? kMinUniform : uniform;
    const double level = -std::log(guarded) * resolved_level_multiplier_;

    return static_cast<std::size_t>(level);
}

std::size_t HnswIndex::capacity_at(std::size_t layer) const {
    return layer == 0 ? params_.max_neighbors_layer0 : params_.max_neighbors;
}

HnswIndex::InternalId* HnswIndex::neighbors_mutable(InternalId id, std::size_t layer) {
    if (layer == 0) {
        return links_layer0_.data() + static_cast<std::size_t>(id) * params_.max_neighbors_layer0;
    }
    return links_upper_[id].data() + (layer - 1) * params_.max_neighbors;
}

const HnswIndex::InternalId* HnswIndex::neighbors(InternalId id, std::size_t layer) const {
    if (layer == 0) {
        return links_layer0_.data() + static_cast<std::size_t>(id) * params_.max_neighbors_layer0;
    }
    return links_upper_[id].data() + (layer - 1) * params_.max_neighbors;
}

std::uint32_t HnswIndex::degree(InternalId id, std::size_t layer) const {
    return layer == 0 ? degrees_layer0_[id] : degrees_upper_[id][layer - 1];
}

void HnswIndex::set_degree(InternalId id, std::size_t layer, std::uint32_t value) {
    if (layer == 0) {
        degrees_layer0_[id] = value;
    } else {
        degrees_upper_[id][layer - 1] = value;
    }
}

std::vector<HnswIndex::Candidate> HnswIndex::search_layer(const float* query,
                                                          const std::vector<InternalId>& entry_points,
                                                          std::size_t ef,
                                                          std::size_t layer,
                                                          const LabelFilter* allowed) const {
    const auto admits = [this, allowed](InternalId id) { return allowed == nullptr || (*allowed)(labels_[id]); };
    const auto nearer = [](const Candidate& a, const Candidate& b) { return a.distance > b.distance; };
    const auto farther = [](const Candidate& a, const Candidate& b) { return a.distance < b.distance; };

    std::priority_queue<Candidate, std::vector<Candidate>, decltype(nearer)> candidates(nearer);
    std::priority_queue<Candidate, std::vector<Candidate>, decltype(farther)> results(farther);
    const VisitedListPool::Lease visited = visited_pool_.acquire(element_count_);

    for (const InternalId entry : entry_points) {
        const float distance = distance_to_stored(query, entry);
        candidates.push({distance, entry});
        if (admits(entry)) {
            results.push({distance, entry});
        }
        visited->mark(entry);
    }

    while (results.size() > ef) {
        results.pop();
    }

    while (!candidates.empty()) {
        const Candidate current = candidates.top();
        if (results.size() >= ef && current.distance > results.top().distance) {
            break;
        }
        candidates.pop();

        const InternalId* adjacency = neighbors(current.id, layer);
        const std::uint32_t adjacency_size = degree(current.id, layer);

        for (std::uint32_t i = 0; i < adjacency_size; ++i) {
            const InternalId neighbor = adjacency[i];
            if (!visited->mark(neighbor)) {
                continue;
            }
            const float distance = distance_to_stored(query, neighbor);
            if (results.size() < ef || distance < results.top().distance) {
                candidates.push({distance, neighbor});
                if (admits(neighbor)) {
                    results.push({distance, neighbor});
                    if (results.size() > ef) {
                        results.pop();
                    }
                }
            }
        }
    }

    std::vector<Candidate> ordered;
    ordered.reserve(results.size());
    while (!results.empty()) {
        ordered.push_back(results.top());
        results.pop();
    }
    std::reverse(ordered.begin(), ordered.end());
    return ordered;
}

HnswIndex::InternalId HnswIndex::greedy_descend(const float* query, InternalId entry, std::size_t layer) const {
    InternalId current = entry;
    float current_distance = distance_to_stored(query, current);
    bool improved = true;

    while (improved) {
        improved = false;
        const InternalId* adjacency = neighbors(current, layer);
        const std::uint32_t adjacency_size = degree(current, layer);

        for (std::uint32_t i = 0; i < adjacency_size; ++i) {
            const InternalId neighbor = adjacency[i];
            const float distance = distance_to_stored(query, neighbor);
            if (distance < current_distance) {
                current_distance = distance;
                current = neighbor;
                improved = true;
            }
        }
    }

    return current;
}

std::vector<HnswIndex::InternalId> HnswIndex::select_neighbors_heuristic(const float* base,
                                                                         std::vector<Candidate> candidates,
                                                                         std::size_t limit) const {
    std::sort(candidates.begin(), candidates.end(),
              [](const Candidate& a, const Candidate& b) { return a.distance < b.distance; });

    std::vector<InternalId> selected;
    selected.reserve(limit);
    std::vector<Candidate> discarded;

    for (const Candidate& candidate : candidates) {
        if (selected.size() >= limit) {
            break;
        }

        bool keep = true;
        for (const InternalId chosen : selected) {
            if (distance_between(candidate.id, chosen) < candidate.distance) {
                keep = false;
                break;
            }
        }

        if (keep) {
            selected.push_back(candidate.id);
        } else {
            discarded.push_back(candidate);
        }
    }

    for (const Candidate& candidate : discarded) {
        if (selected.size() >= limit) {
            break;
        }
        selected.push_back(candidate.id);
    }

    (void)base;
    return selected;
}

std::vector<HnswIndex::InternalId> HnswIndex::select_neighbors_heuristic_for(InternalId base,
                                                                             std::vector<Candidate> candidates,
                                                                             std::size_t limit) const {
    return select_neighbors_heuristic(vector_at(base), std::move(candidates), limit);
}

void HnswIndex::connect(InternalId from, const std::vector<InternalId>& targets, std::size_t layer) {
    InternalId* adjacency = neighbors_mutable(from, layer);
    const std::size_t capacity = capacity_at(layer);
    const std::size_t count = std::min(targets.size(), capacity);

    for (std::size_t i = 0; i < count; ++i) {
        adjacency[i] = targets[i];
    }
    set_degree(from, layer, static_cast<std::uint32_t>(count));
}

void HnswIndex::prune_if_overfull(InternalId node, std::size_t layer) {
    const std::size_t capacity = capacity_at(layer);
    if (degree(node, layer) <= capacity) {
        return;
    }

    const InternalId* adjacency = neighbors(node, layer);
    std::vector<Candidate> existing;
    existing.reserve(degree(node, layer));

    for (std::uint32_t i = 0; i < degree(node, layer); ++i) {
        existing.push_back({distance_between(node, adjacency[i]), adjacency[i]});
    }

    const std::vector<InternalId> kept = select_neighbors_heuristic_for(node, std::move(existing), capacity);
    connect(node, kept, layer);
}

void HnswIndex::add(std::uint64_t label, const float* vector) {
    const InternalId id = static_cast<InternalId>(element_count_);
    const std::size_t level = draw_level();

    vectors_.insert(vectors_.end(), vector, vector + params_.dimension);
    labels_.push_back(label);
    levels_.push_back(static_cast<std::uint32_t>(level));

    links_layer0_.resize(links_layer0_.size() + params_.max_neighbors_layer0, 0);
    degrees_layer0_.push_back(0);

    links_upper_.emplace_back(level * params_.max_neighbors, 0);
    degrees_upper_.emplace_back(level, 0);

    ++element_count_;

    if (!has_entry_point_) {
        entry_point_ = id;
        max_level_ = level;
        has_entry_point_ = true;
        return;
    }

    InternalId current_entry = entry_point_;

    for (std::size_t layer = max_level_; layer > level; --layer) {
        current_entry = greedy_descend(vector, current_entry, layer);
    }

    const std::size_t start_layer = std::min(level, max_level_);
    for (std::size_t layer = start_layer + 1; layer-- > 0;) {
        const std::size_t capacity = capacity_at(layer);

        std::vector<Candidate> pool = search_layer(vector, {current_entry}, params_.ef_construction, layer);
        if (pool.empty()) {
            continue;
        }

        const std::vector<InternalId> selected = select_neighbors_heuristic(vector, pool, capacity);
        connect(id, selected, layer);

        for (const InternalId neighbor : selected) {
            const std::uint32_t existing_degree = degree(neighbor, layer);
            const std::size_t neighbor_capacity = capacity_at(layer);

            if (existing_degree < neighbor_capacity) {
                neighbors_mutable(neighbor, layer)[existing_degree] = id;
                set_degree(neighbor, layer, existing_degree + 1);
            } else {
                const InternalId* adjacency = neighbors(neighbor, layer);
                std::vector<Candidate> combined;
                combined.reserve(existing_degree + 1);
                for (std::uint32_t i = 0; i < existing_degree; ++i) {
                    combined.push_back({distance_between(neighbor, adjacency[i]), adjacency[i]});
                }
                combined.push_back({distance_between(neighbor, id), id});

                const std::vector<InternalId> kept =
                    select_neighbors_heuristic_for(neighbor, std::move(combined), neighbor_capacity);
                connect(neighbor, kept, layer);
            }
        }

        current_entry = pool.front().id;
    }

    if (level > max_level_) {
        max_level_ = level;
        entry_point_ = id;
    }
}

std::vector<SearchResult> HnswIndex::search(const float* query,
                                            std::size_t k,
                                            std::size_t ef_search,
                                            const LabelFilter& allowed) const {
    if (element_count_ == 0 || k == 0) {
        return {};
    }

    const std::size_t ef = std::max(ef_search, k);
    InternalId current_entry = entry_point_;

    for (std::size_t layer = max_level_; layer > 0; --layer) {
        current_entry = greedy_descend(query, current_entry, layer);
    }

    const LabelFilter* const filter = allowed ? &allowed : nullptr;
    const std::vector<Candidate> found = search_layer(query, {current_entry}, ef, 0, filter);

    std::vector<SearchResult> results;
    const std::size_t count = std::min(k, found.size());
    results.reserve(count);

    for (std::size_t i = 0; i < count; ++i) {
        results.push_back({labels_[found[i].id], found[i].distance});
    }
    return results;
}

void HnswIndex::save(const std::string& path) const {
    std::ofstream out(path, std::ios::binary);
    if (!out) {
        throw std::runtime_error("cannot open index file for writing: " + path);
    }

    const auto write_scalar = [&out](auto value) {
        out.write(reinterpret_cast<const char*>(&value), sizeof(value));
    };

    write_scalar(kMagic);
    write_scalar(static_cast<std::uint64_t>(params_.dimension));
    write_scalar(static_cast<std::uint64_t>(params_.max_neighbors));
    write_scalar(static_cast<std::uint64_t>(params_.max_neighbors_layer0));
    write_scalar(static_cast<std::uint64_t>(params_.ef_construction));
    write_scalar(params_.level_multiplier);
    write_scalar(params_.seed);
    write_scalar(static_cast<std::uint32_t>(params_.metric));
    write_scalar(static_cast<std::uint64_t>(element_count_));
    write_scalar(static_cast<std::uint64_t>(max_level_));
    write_scalar(entry_point_);

    out.write(reinterpret_cast<const char*>(vectors_.data()), vectors_.size() * sizeof(float));
    out.write(reinterpret_cast<const char*>(labels_.data()), labels_.size() * sizeof(std::uint64_t));
    out.write(reinterpret_cast<const char*>(levels_.data()), levels_.size() * sizeof(std::uint32_t));
    out.write(reinterpret_cast<const char*>(links_layer0_.data()), links_layer0_.size() * sizeof(InternalId));
    out.write(reinterpret_cast<const char*>(degrees_layer0_.data()), degrees_layer0_.size() * sizeof(std::uint32_t));

    for (std::size_t i = 0; i < element_count_; ++i) {
        const std::uint64_t upper_size = links_upper_[i].size();
        write_scalar(upper_size);
        if (upper_size > 0) {
            out.write(reinterpret_cast<const char*>(links_upper_[i].data()), upper_size * sizeof(InternalId));
            out.write(reinterpret_cast<const char*>(degrees_upper_[i].data()),
                      degrees_upper_[i].size() * sizeof(std::uint32_t));
        }
    }
}

HnswIndex HnswIndex::load(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("cannot open index file for reading: " + path);
    }

    const auto read_scalar = [&in](auto& value) {
        in.read(reinterpret_cast<char*>(&value), sizeof(value));
    };

    std::uint32_t magic = 0;
    read_scalar(magic);
    if (magic != kMagic) {
        throw std::runtime_error("not a khoj index file: " + path);
    }

    HnswParams params;
    std::uint64_t dimension = 0;
    std::uint64_t max_neighbors = 0;
    std::uint64_t max_neighbors_layer0 = 0;
    std::uint64_t ef_construction = 0;
    std::uint32_t metric = 0;

    read_scalar(dimension);
    read_scalar(max_neighbors);
    read_scalar(max_neighbors_layer0);
    read_scalar(ef_construction);
    read_scalar(params.level_multiplier);
    read_scalar(params.seed);
    read_scalar(metric);

    params.dimension = static_cast<std::size_t>(dimension);
    params.max_neighbors = static_cast<std::size_t>(max_neighbors);
    params.max_neighbors_layer0 = static_cast<std::size_t>(max_neighbors_layer0);
    params.ef_construction = static_cast<std::size_t>(ef_construction);
    params.metric = static_cast<Metric>(metric);

    HnswIndex index(params);

    std::uint64_t element_count = 0;
    std::uint64_t max_level = 0;
    read_scalar(element_count);
    read_scalar(max_level);
    read_scalar(index.entry_point_);

    index.element_count_ = static_cast<std::size_t>(element_count);
    index.max_level_ = static_cast<std::size_t>(max_level);
    index.has_entry_point_ = index.element_count_ > 0;

    index.vectors_.resize(index.element_count_ * params.dimension);
    index.labels_.resize(index.element_count_);
    index.levels_.resize(index.element_count_);
    index.links_layer0_.resize(index.element_count_ * params.max_neighbors_layer0);
    index.degrees_layer0_.resize(index.element_count_);

    in.read(reinterpret_cast<char*>(index.vectors_.data()), index.vectors_.size() * sizeof(float));
    in.read(reinterpret_cast<char*>(index.labels_.data()), index.labels_.size() * sizeof(std::uint64_t));
    in.read(reinterpret_cast<char*>(index.levels_.data()), index.levels_.size() * sizeof(std::uint32_t));
    in.read(reinterpret_cast<char*>(index.links_layer0_.data()), index.links_layer0_.size() * sizeof(InternalId));
    in.read(reinterpret_cast<char*>(index.degrees_layer0_.data()),
            index.degrees_layer0_.size() * sizeof(std::uint32_t));

    index.links_upper_.resize(index.element_count_);
    index.degrees_upper_.resize(index.element_count_);

    for (std::size_t i = 0; i < index.element_count_; ++i) {
        std::uint64_t upper_size = 0;
        read_scalar(upper_size);
        index.links_upper_[i].resize(upper_size);
        index.degrees_upper_[i].resize(upper_size / params.max_neighbors);

        if (upper_size > 0) {
            in.read(reinterpret_cast<char*>(index.links_upper_[i].data()), upper_size * sizeof(InternalId));
            in.read(reinterpret_cast<char*>(index.degrees_upper_[i].data()),
                    index.degrees_upper_[i].size() * sizeof(std::uint32_t));
        }
    }

    return index;
}

}
