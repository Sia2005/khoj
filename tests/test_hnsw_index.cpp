#include <catch2/catch_test_macros.hpp>

#include "khoj/hnsw_index.hpp"

TEST_CASE("default hnsw params are populated", "[hnsw][params]") {
    const khoj::HnswParams params;

    REQUIRE(params.dimension == 0);
    REQUIRE(params.max_neighbors == 16);
    REQUIRE(params.ef_construction == 200);
    REQUIRE(params.seed == 0);
}

TEST_CASE("search result defaults to an empty match", "[hnsw][search]") {
    const khoj::SearchResult result;

    REQUIRE(result.id == 0);
    REQUIRE(result.distance == 0.0f);
}
