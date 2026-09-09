#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <random>
#include <stdexcept>
#include <thread>
#include <vector>

#include "khoj/flat_index.hpp"

using Catch::Matchers::WithinAbs;
using khoj::FlatIndex;
using khoj::Metric;
using khoj::SearchResult;
using khoj::VectorId;

namespace {

std::vector<float> random_values(std::size_t total, std::uint64_t seed) {
    std::mt19937_64 engine(seed);
    std::uniform_real_distribution<float> distribution(-1.0f, 1.0f);

    std::vector<float> values(total);
    for (float& value : values) {
        value = distribution(engine);
    }
    return values;
}

std::vector<SearchResult> naive_search(const std::vector<float>& data,
                                       const std::vector<float>& query,
                                       std::size_t dimension,
                                       Metric metric,
                                       std::size_t k) {
    const std::size_t count = data.size() / dimension;

    std::vector<SearchResult> scored;
    scored.reserve(count);

    for (std::size_t position = 0; position < count; ++position) {
        float total = 0.0f;
        for (std::size_t index = 0; index < dimension; ++index) {
            const float stored = data[position * dimension + index];
            if (metric == Metric::L2) {
                const float difference = stored - query[index];
                total += difference * difference;
            } else {
                total += stored * query[index];
            }
        }
        const float score = metric == Metric::L2 ? std::sqrt(total) : total;
        scored.push_back(SearchResult{static_cast<VectorId>(position), score});
    }

    std::stable_sort(scored.begin(), scored.end(),
                     [metric](const SearchResult& left, const SearchResult& right) {
                         if (left.distance != right.distance) {
                             return metric == Metric::L2 ? left.distance < right.distance
                                                         : left.distance > right.distance;
                         }
                         return left.id < right.id;
                     });

    scored.resize(std::min(k, scored.size()));
    return scored;
}

void require_matches_reference(const std::vector<SearchResult>& actual,
                               const std::vector<SearchResult>& expected) {
    REQUIRE(actual.size() == expected.size());
    for (std::size_t position = 0; position < actual.size(); ++position) {
        REQUIRE(actual[position].id == expected[position].id);
        REQUIRE_THAT(actual[position].distance, WithinAbs(expected[position].distance, 1e-4f));
    }
}

}

TEST_CASE("l2 search returns exact distances in ascending order", "[flat][l2]") {
    FlatIndex index(2, Metric::L2);
    index.add({0.0f, 0.0f});
    index.add({3.0f, 4.0f});
    index.add({1.0f, 1.0f});

    const std::vector<SearchResult> results = index.search({0.0f, 0.0f}, 3);

    REQUIRE(results.size() == 3);
    REQUIRE(results[0].id == 0);
    REQUIRE_THAT(results[0].distance, WithinAbs(0.0f, 1e-6f));
    REQUIRE(results[1].id == 2);
    REQUIRE_THAT(results[1].distance, WithinAbs(std::sqrt(2.0f), 1e-6f));
    REQUIRE(results[2].id == 1);
    REQUIRE_THAT(results[2].distance, WithinAbs(5.0f, 1e-6f));
}

TEST_CASE("inner product search returns the largest score first", "[flat][ip]") {
    FlatIndex index(3, Metric::InnerProduct);
    index.add({1.0f, 0.0f, 0.0f});
    index.add({0.0f, 1.0f, 0.0f});
    index.add({2.0f, 0.0f, 0.0f});

    const std::vector<SearchResult> results = index.search({1.0f, 0.0f, 0.0f}, 3);

    REQUIRE(results.size() == 3);
    REQUIRE(results[0].id == 2);
    REQUIRE_THAT(results[0].distance, WithinAbs(2.0f, 1e-6f));
    REQUIRE(results[1].id == 0);
    REQUIRE_THAT(results[1].distance, WithinAbs(1.0f, 1e-6f));
    REQUIRE(results[2].id == 1);
    REQUIRE_THAT(results[2].distance, WithinAbs(0.0f, 1e-6f));
}

TEST_CASE("add assigns sequential ids and tracks size", "[flat][add]") {
    FlatIndex index(4, Metric::L2);

    REQUIRE(index.size() == 0);
    REQUIRE(index.add({1.0f, 2.0f, 3.0f, 4.0f}) == 0);
    REQUIRE(index.add({5.0f, 6.0f, 7.0f, 8.0f}) == 1);
    REQUIRE(index.size() == 2);
    REQUIRE(index.dimension() == 4);
    REQUIRE(index.data().size() == 8);
}

TEST_CASE("add_batch stores the same buffer as repeated add", "[flat][add]") {
    const std::vector<float> values = random_values(6 * 5, 7);

    FlatIndex batched(5, Metric::L2);
    batched.add_batch(values);

    FlatIndex individual(5, Metric::L2);
    for (std::size_t position = 0; position < 6; ++position) {
        const auto begin = values.begin() + static_cast<std::ptrdiff_t>(position * 5);
        individual.add(std::vector<float>(begin, begin + 5));
    }

    REQUIRE(batched.size() == 6);
    REQUIRE(batched.data() == individual.data());
}

TEST_CASE("search clamps k to the number of stored vectors", "[flat][search]") {
    FlatIndex index(2, Metric::L2);
    index.add({1.0f, 1.0f});
    index.add({2.0f, 2.0f});

    REQUIRE(index.search({0.0f, 0.0f}, 10).size() == 2);
    REQUIRE(index.search({0.0f, 0.0f}, 0).empty());
}

TEST_CASE("search on an empty index returns nothing", "[flat][search]") {
    const FlatIndex index(8, Metric::L2);

    REQUIRE(index.search(std::vector<float>(8, 0.0f), 5).empty());
}

TEST_CASE("dimension mismatches are rejected", "[flat][errors]") {
    FlatIndex index(3, Metric::L2);

    REQUIRE_THROWS_AS(index.add({1.0f, 2.0f}), std::invalid_argument);
    REQUIRE_THROWS_AS(index.add_batch({1.0f, 2.0f, 3.0f, 4.0f}), std::invalid_argument);
    REQUIRE_THROWS_AS(index.search({1.0f, 2.0f}, 1), std::invalid_argument);
    REQUIRE_THROWS_AS(FlatIndex(0, Metric::L2), std::invalid_argument);
}

TEST_CASE("equal distances are broken by ascending id", "[flat][search]") {
    FlatIndex index(4, Metric::L2);
    index.add({1.0f, 1.0f, 1.0f, 1.0f});
    index.add({1.0f, 1.0f, 1.0f, 1.0f});
    index.add({1.0f, 1.0f, 1.0f, 1.0f});

    const std::vector<SearchResult> results = index.search({1.0f, 1.0f, 1.0f, 1.0f}, 2);

    REQUIRE(results.size() == 2);
    REQUIRE(results[0].id == 0);
    REQUIRE(results[1].id == 1);
}

TEST_CASE("dimensions that straddle the unrolled tail stay exact", "[flat][simd]") {
    const std::size_t count = 37;

    for (const std::size_t dimension : {1u, 7u, 8u, 9u, 13u, 16u, 31u}) {
        const std::vector<float> data = random_values(count * dimension, 11 + dimension);
        const std::vector<float> query = random_values(dimension, 99 + dimension);

        for (const Metric metric : {Metric::L2, Metric::InnerProduct}) {
            FlatIndex index(dimension, metric);
            index.add_batch(data);

            require_matches_reference(index.search(query, 5),
                                      naive_search(data, query, dimension, metric, 5));
        }
    }
}

TEST_CASE("unrolled kernels agree with a scalar reference", "[flat][simd]") {
    constexpr std::size_t dimension = 128;
    constexpr std::size_t count = 500;

    const std::vector<float> data = random_values(count * dimension, 2024);
    const std::vector<float> query = random_values(dimension, 4242);

    for (const Metric metric : {Metric::L2, Metric::InnerProduct}) {
        FlatIndex index(dimension, metric);
        index.reserve(count);
        index.add_batch(data);

        REQUIRE(index.size() == count);
        require_matches_reference(index.search(query, 10),
                                  naive_search(data, query, dimension, metric, 10));
    }
}

TEST_CASE("search_batch matches per query search", "[flat][search]") {
    constexpr std::size_t dimension = 16;
    constexpr std::size_t count = 60;
    constexpr std::size_t query_count = 5;

    const std::vector<float> data = random_values(count * dimension, 31337);
    const std::vector<float> queries = random_values(query_count * dimension, 1234);

    FlatIndex index(dimension, Metric::L2);
    index.add_batch(data);

    const std::vector<std::vector<SearchResult>> batched = index.search_batch(queries, 4);
    REQUIRE(batched.size() == query_count);

    for (std::size_t position = 0; position < query_count; ++position) {
        const auto begin = queries.begin() + static_cast<std::ptrdiff_t>(position * dimension);
        const std::vector<float> query(begin, begin + dimension);
        require_matches_reference(batched[position], index.search(query, 4));
    }
}

TEST_CASE("parallel queries match the single threaded run", "[flat][concurrency]") {
    constexpr std::size_t dimension = 32;
    constexpr std::size_t count = 400;
    constexpr std::size_t query_count = 64;
    constexpr std::size_t k = 10;

    const std::vector<float> data = random_values(count * dimension, 555);
    const std::vector<float> queries = random_values(query_count * dimension, 777);

    FlatIndex index(dimension, Metric::L2);
    index.add_batch(data);

    const std::vector<std::vector<SearchResult>> expected = index.search_batch(queries, k);

    std::vector<std::vector<SearchResult>> actual(query_count);
    const std::size_t thread_count = 4;
    std::vector<std::thread> workers;
    workers.reserve(thread_count);

    for (std::size_t worker = 0; worker < thread_count; ++worker) {
        workers.emplace_back([&, worker]() {
            for (std::size_t position = worker; position < query_count; position += thread_count) {
                const auto begin =
                    queries.begin() + static_cast<std::ptrdiff_t>(position * dimension);
                const std::vector<float> query(begin, begin + dimension);
                actual[position] = index.search(query, k);
            }
        });
    }

    for (std::thread& worker : workers) {
        worker.join();
    }

    for (std::size_t position = 0; position < query_count; ++position) {
        require_matches_reference(actual[position], expected[position]);
    }
}
