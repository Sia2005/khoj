#include <catch2/catch_test_macros.hpp>

#include "khoj/hnsw_index.hpp"

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
