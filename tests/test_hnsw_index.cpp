#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <random>
#include <thread>
#include <unordered_set>
#include <vector>

#include "khoj/flat_index.hpp"
#include "khoj/hnsw_index.hpp"

using Catch::Matchers::WithinRel;
using khoj::FlatIndex;
using khoj::HnswIndex;
using khoj::HnswParams;
using khoj::LabelFilter;
using khoj::Metric;
using khoj::SearchResult;

namespace {

struct RecallSummary {
    double mean;
    double worst;
};

struct SweepPoint {
    std::size_t ef_search;
    double minimum_recall;
};

std::vector<float> gaussian_cloud(std::size_t count,
                                  std::size_t dimension,
                                  std::uint64_t seed,
                                  float center,
                                  float spread) {
    std::mt19937_64 engine(seed);
    std::normal_distribution<float> normal(center, spread);

    std::vector<float> values(count * dimension);
    for (float& value : values) {
        value = normal(engine);
    }
    return values;
}

HnswIndex index_over(const std::vector<float>& data, std::size_t dimension) {
    HnswParams params;
    params.dimension = dimension;

    HnswIndex index(params);
    const std::size_t count = data.size() / dimension;
    index.reserve(count);

    for (std::size_t position = 0; position < count; ++position) {
        index.add(static_cast<std::uint64_t>(position), data.data() + position * dimension);
    }
    return index;
}

double recall_of(const std::vector<SearchResult>& expected, const std::vector<SearchResult>& actual) {
    if (expected.empty()) {
        return 1.0;
    }

    std::size_t hits = 0;
    for (const SearchResult& target : expected) {
        const bool retrieved =
            std::any_of(actual.begin(), actual.end(), [&target](const SearchResult& candidate) {
                return candidate.label == target.label;
            });
        if (retrieved) {
            ++hits;
        }
    }
    return static_cast<double>(hits) / static_cast<double>(expected.size());
}

RecallSummary recall_over(const std::vector<std::vector<SearchResult>>& truth,
                          const HnswIndex& index,
                          const std::vector<float>& queries,
                          std::size_t k,
                          std::size_t ef_search) {
    const std::size_t dimension = index.dimension();
    const std::size_t query_count = queries.size() / dimension;

    double total = 0.0;
    double worst = 1.0;

    for (std::size_t position = 0; position < query_count; ++position) {
        const float* const query = queries.data() + position * dimension;
        const double recall = recall_of(truth[position], index.search(query, k, ef_search));
        total += recall;
        worst = std::min(worst, recall);
    }

    return RecallSummary{total / static_cast<double>(query_count), worst};
}

std::vector<float> concatenated(const std::vector<float>& first, const std::vector<float>& second) {
    std::vector<float> joined(first);
    joined.insert(joined.end(), second.begin(), second.end());
    return joined;
}

std::unordered_set<std::uint64_t> random_subset(std::size_t count, double fraction, std::uint64_t seed) {
    std::mt19937_64 engine(seed);
    std::bernoulli_distribution chosen(fraction);

    std::unordered_set<std::uint64_t> subset;
    for (std::size_t position = 0; position < count; ++position) {
        if (chosen(engine)) {
            subset.insert(static_cast<std::uint64_t>(position));
        }
    }
    return subset;
}

LabelFilter membership_in(const std::unordered_set<std::uint64_t>& allowed) {
    return [&allowed](std::uint64_t label) { return allowed.count(label) != 0; };
}

std::vector<SearchResult> restricted_brute_force(const std::vector<float>& data,
                                                 std::size_t dimension,
                                                 const float* query,
                                                 std::size_t k,
                                                 const std::unordered_set<std::uint64_t>& allowed) {
    std::vector<SearchResult> scored;
    for (const std::uint64_t label : allowed) {
        const float* const stored = data.data() + label * dimension;
        float distance = 0.0f;
        for (std::size_t axis = 0; axis < dimension; ++axis) {
            const float difference = query[axis] - stored[axis];
            distance += difference * difference;
        }
        scored.push_back({label, distance});
    }

    const std::size_t kept = std::min(k, scored.size());
    std::partial_sort(scored.begin(), scored.begin() + static_cast<std::ptrdiff_t>(kept), scored.end(),
                      [](const SearchResult& left, const SearchResult& right) {
                          return left.distance < right.distance;
                      });
    scored.resize(kept);
    return scored;
}

bool identical(const std::vector<SearchResult>& left, const std::vector<SearchResult>& right) {
    return std::equal(left.begin(), left.end(), right.begin(), right.end(),
                      [](const SearchResult& a, const SearchResult& b) {
                          return a.label == b.label && a.distance == b.distance;
                      });
}

}

TEST_CASE("default hnsw params are populated", "[hnsw][params]") {
    const khoj::HnswParams params;

    REQUIRE(params.dimension == 0);
    REQUIRE(params.max_neighbors == 16);
    REQUIRE(params.max_neighbors_layer0 == 32);
    REQUIRE(params.ef_construction == 200);
    REQUIRE(params.level_multiplier == 0.0);
    REQUIRE(params.seed == 42);
    REQUIRE(params.metric == khoj::Metric::L2);
}

TEST_CASE("construction rejects degenerate parameters", "[hnsw][params]") {
    khoj::HnswParams params;
    params.dimension = 0;
    REQUIRE_THROWS(khoj::HnswIndex{params});

    params.dimension = 8;
    params.max_neighbors = 0;
    REQUIRE_THROWS(khoj::HnswIndex{params});

    params.max_neighbors = 16;
    params.ef_construction = 4;
    REQUIRE_THROWS(khoj::HnswIndex{params});
}

TEST_CASE("recall against the flat index rises with ef_search", "[hnsw][recall]") {
    constexpr std::size_t dimension = 32;
    constexpr std::size_t count = 5000;
    constexpr std::size_t query_count = 100;
    constexpr std::size_t k = 10;
    constexpr double sweep_regression_tolerance = 0.02;

    const std::vector<float> data = gaussian_cloud(count, dimension, 20240617, 0.0f, 1.0f);
    const std::vector<float> queries = gaussian_cloud(query_count, dimension, 913, 0.0f, 1.0f);

    FlatIndex reference(dimension, Metric::L2);
    reference.add_batch(data);
    const std::vector<std::vector<SearchResult>> truth = reference.search_batch(queries, k);

    const HnswIndex index = index_over(data, dimension);
    REQUIRE(index.size() == count);
    REQUIRE(index.max_level() > 0);

    const std::vector<SweepPoint> sweep = {{10, 0.60}, {50, 0.95}, {100, 0.97}, {200, 0.98}};

    double previous_recall = 0.0;
    for (const SweepPoint& point : sweep) {
        const RecallSummary summary = recall_over(truth, index, queries, k, point.ef_search);

        INFO("ef_search=" << point.ef_search << " mean recall@10=" << summary.mean
                          << " worst recall@10=" << summary.worst);
        REQUIRE(summary.mean >= point.minimum_recall);
        REQUIRE(summary.mean >= previous_recall - sweep_regression_tolerance);

        previous_recall = summary.mean;
    }
}

TEST_CASE("an exhaustive ef_search reproduces the flat index exactly", "[hnsw][recall]") {
    constexpr std::size_t dimension = 16;
    constexpr std::size_t count = 400;
    constexpr std::size_t query_count = 20;
    constexpr std::size_t k = 5;

    const std::vector<float> data = gaussian_cloud(count, dimension, 4242, 0.0f, 1.0f);
    const std::vector<float> queries = gaussian_cloud(query_count, dimension, 24, 0.0f, 1.0f);

    FlatIndex reference(dimension, Metric::L2);
    reference.add_batch(data);

    const HnswIndex index = index_over(data, dimension);

    for (std::size_t position = 0; position < query_count; ++position) {
        const auto begin = queries.begin() + static_cast<std::ptrdiff_t>(position * dimension);
        const std::vector<float> query(begin, begin + dimension);

        const std::vector<SearchResult> expected = reference.search(query, k);
        const std::vector<SearchResult> actual = index.search(query.data(), k, count);

        INFO("query " << position);
        REQUIRE(actual.size() == expected.size());

        for (std::size_t rank = 0; rank < actual.size(); ++rank) {
            REQUIRE(actual[rank].label == expected[rank].label);
            REQUIRE_THAT(actual[rank].distance,
                         WithinRel(expected[rank].distance * expected[rank].distance, 1e-4f));
            if (rank > 0) {
                REQUIRE(actual[rank].distance >= actual[rank - 1].distance);
            }
        }
    }
}

TEST_CASE("the graph bridges two separated clusters", "[hnsw][recall][bridge]") {
    constexpr std::size_t dimension = 32;
    constexpr std::size_t per_cluster = 1000;
    constexpr std::size_t k = 10;
    constexpr float separation = 50.0f;
    constexpr float spread = 0.5f;

    const std::vector<float> data = concatenated(
        gaussian_cloud(per_cluster, dimension, 55, -separation, spread),
        gaussian_cloud(per_cluster, dimension, 56, separation, spread));

    FlatIndex reference(dimension, Metric::L2);
    reference.add_batch(data);

    const HnswIndex index = index_over(data, dimension);
    REQUIRE(index.size() == 2 * per_cluster);

    SECTION("a query between the clusters retrieves neighbours from both") {
        const std::vector<float> midpoint(dimension, 0.0f);
        const std::vector<SearchResult> expected = reference.search(midpoint, k);

        const std::size_t from_near_cluster =
            static_cast<std::size_t>(std::count_if(expected.begin(), expected.end(),
                                                   [](const SearchResult& result) {
                                                       return result.label < per_cluster;
                                                   }));
        REQUIRE(from_near_cluster > 0);
        REQUIRE(from_near_cluster < k);

        const std::vector<SearchResult> actual = index.search(midpoint.data(), k, 100);
        INFO("recall@10 = " << recall_of(expected, actual));
        REQUIRE(actual.front().label == expected.front().label);
        REQUIRE(recall_of(expected, actual) >= 0.9);
    }

    SECTION("queries drawn from either cluster match the flat index") {
        constexpr std::size_t per_cluster_queries = 25;
        constexpr std::size_t ef_search = 50;

        const std::vector<float> queries = concatenated(
            gaussian_cloud(per_cluster_queries, dimension, 771, -separation, spread),
            gaussian_cloud(per_cluster_queries, dimension, 772, separation, spread));

        const std::vector<std::vector<SearchResult>> truth = reference.search_batch(queries, k);
        const RecallSummary summary = recall_over(truth, index, queries, k, ef_search);

        INFO("mean recall@10=" << summary.mean << " worst recall@10=" << summary.worst);
        REQUIRE(summary.mean >= 0.98);
        REQUIRE(summary.worst >= 0.8);
    }

    SECTION("an indexed vector from the far cluster is returned exactly") {
        constexpr std::size_t ef_search = 10;

        for (std::size_t offset = 0; offset < per_cluster; offset += 100) {
            const std::size_t target = per_cluster + offset;
            const std::vector<SearchResult> found =
                index.search(data.data() + target * dimension, 1, ef_search);

            INFO("target " << target);
            REQUIRE(found.size() == 1);
            REQUIRE(found.front().label == target);
            REQUIRE(found.front().distance == 0.0f);
        }
    }
}

TEST_CASE("a filtered search returns only allowed labels at high recall", "[hnsw][filter][recall]") {
    constexpr std::size_t dimension = 32;
    constexpr std::size_t count = 5000;
    constexpr std::size_t query_count = 100;
    constexpr std::size_t k = 10;
    constexpr std::size_t ef_search = 100;

    const std::vector<float> data = gaussian_cloud(count, dimension, 20240617, 0.0f, 1.0f);
    const std::vector<float> queries = gaussian_cloud(query_count, dimension, 913, 0.0f, 1.0f);
    const HnswIndex index = index_over(data, dimension);

    const double fraction = GENERATE(0.01, 0.1, 0.5, 0.9);
    const std::unordered_set<std::uint64_t> allowed = random_subset(count, fraction, 77);
    const LabelFilter filter = membership_in(allowed);

    double total_recall = 0.0;
    for (std::size_t position = 0; position < query_count; ++position) {
        const float* const query = queries.data() + position * dimension;
        const std::vector<SearchResult> expected = restricted_brute_force(data, dimension, query, k, allowed);
        const std::vector<SearchResult> actual = index.search(query, k, ef_search, filter);

        INFO("fraction " << fraction << " query " << position);
        REQUIRE(actual.size() == expected.size());
        for (const SearchResult& result : actual) {
            REQUIRE(allowed.count(result.label) == 1);
        }
        total_recall += recall_of(expected, actual);
    }

    const double mean_recall = total_recall / static_cast<double>(query_count);
    INFO("fraction " << fraction << " mean recall@10=" << mean_recall);
    REQUIRE(mean_recall >= 0.95);
}

TEST_CASE("a filter that admits every label changes nothing", "[hnsw][filter]") {
    constexpr std::size_t dimension = 16;
    constexpr std::size_t count = 2000;
    constexpr std::size_t query_count = 50;
    constexpr std::size_t k = 10;
    constexpr std::size_t ef_search = 40;

    const std::vector<float> data = gaussian_cloud(count, dimension, 31, 0.0f, 1.0f);
    const std::vector<float> queries = gaussian_cloud(query_count, dimension, 32, 0.0f, 1.0f);
    const HnswIndex index = index_over(data, dimension);
    const LabelFilter admit_all = [](std::uint64_t) { return true; };

    for (std::size_t position = 0; position < query_count; ++position) {
        const float* const query = queries.data() + position * dimension;
        INFO("query " << position);
        REQUIRE(identical(index.search(query, k, ef_search), index.search(query, k, ef_search, admit_all)));
    }
}

TEST_CASE("a filter narrower than k returns every allowed label", "[hnsw][filter]") {
    constexpr std::size_t dimension = 16;
    constexpr std::size_t count = 2000;
    constexpr std::size_t k = 10;
    constexpr std::size_t ef_search = 20;

    const std::vector<float> data = gaussian_cloud(count, dimension, 41, 0.0f, 1.0f);
    const std::vector<float> query = gaussian_cloud(1, dimension, 42, 0.0f, 1.0f);
    const HnswIndex index = index_over(data, dimension);

    SECTION("three allowed labels come back nearest first") {
        const std::unordered_set<std::uint64_t> allowed = {7, 1234, 1999};
        const std::vector<SearchResult> expected =
            restricted_brute_force(data, dimension, query.data(), k, allowed);
        const std::vector<SearchResult> actual = index.search(query.data(), k, ef_search, membership_in(allowed));

        REQUIRE(actual.size() == allowed.size());
        for (std::size_t rank = 0; rank < actual.size(); ++rank) {
            REQUIRE(actual[rank].label == expected[rank].label);
        }
    }

    SECTION("a filter that admits nothing returns nothing") {
        const LabelFilter admit_none = [](std::uint64_t) { return false; };
        REQUIRE(index.search(query.data(), k, ef_search, admit_none).empty());
    }
}

TEST_CASE("parallel filtered queries reproduce the single-threaded results", "[hnsw][filter][concurrency]") {
    constexpr std::size_t dimension = 16;
    constexpr std::size_t count = 3000;
    constexpr std::size_t query_count = 200;
    constexpr std::size_t k = 10;
    constexpr std::size_t ef_search = 50;
    constexpr std::size_t thread_count = 4;

    const std::vector<float> data = gaussian_cloud(count, dimension, 51, 0.0f, 1.0f);
    const std::vector<float> queries = gaussian_cloud(query_count, dimension, 52, 0.0f, 1.0f);
    const HnswIndex index = index_over(data, dimension);
    const std::unordered_set<std::uint64_t> allowed = random_subset(count, 0.1, 53);
    const LabelFilter filter = membership_in(allowed);

    std::vector<std::vector<SearchResult>> sequential(query_count);
    for (std::size_t position = 0; position < query_count; ++position) {
        sequential[position] = index.search(queries.data() + position * dimension, k, ef_search, filter);
    }

    std::vector<std::vector<SearchResult>> parallel(query_count);
    std::vector<std::thread> workers;
    for (std::size_t worker = 0; worker < thread_count; ++worker) {
        workers.emplace_back([&, worker] {
            for (std::size_t position = worker; position < query_count; position += thread_count) {
                parallel[position] = index.search(queries.data() + position * dimension, k, ef_search, filter);
            }
        });
    }
    for (std::thread& thread : workers) {
        thread.join();
    }

    for (std::size_t position = 0; position < query_count; ++position) {
        INFO("query " << position);
        REQUIRE(identical(sequential[position], parallel[position]));
    }
}
