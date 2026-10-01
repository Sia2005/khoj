#include <catch2/catch_test_macros.hpp>

#include <cstddef>
#include <cstdint>

#include "khoj/visited_list_pool.hpp"

using khoj::VisitedList;
using khoj::VisitedListPool;

TEST_CASE("a visited list reports each id once per query", "[visited]") {
    VisitedList list;
    list.begin_query(4);

    REQUIRE(list.mark(2));
    REQUIRE_FALSE(list.mark(2));
    REQUIRE(list.mark(3));
}

TEST_CASE("beginning a query forgets every earlier mark", "[visited]") {
    VisitedList list;
    list.begin_query(4);
    list.mark(0);
    list.mark(1);

    list.begin_query(4);

    REQUIRE(list.mark(0));
    REQUIRE(list.mark(1));
}

TEST_CASE("marks stay isolated across a full wrap of the stamp counter", "[visited]") {
    constexpr std::size_t stamp_period = 65536;
    VisitedList list;

    list.begin_query(2);
    list.mark(0);

    for (std::size_t query = 1; query <= 2 * stamp_period; ++query) {
        list.begin_query(2);
        REQUIRE(list.mark(query % 2 == 0 ? 0 : 1));
    }

    list.begin_query(2);
    REQUIRE(list.mark(0));
    REQUIRE(list.mark(1));
}

TEST_CASE("a visited list grows to cover ids added after it was created", "[visited]") {
    VisitedList list;
    list.begin_query(2);
    list.mark(1);

    list.begin_query(10);

    REQUIRE(list.mark(1));
    REQUIRE(list.mark(9));
    REQUIRE_FALSE(list.mark(9));
}

TEST_CASE("the pool hands a released list to the next lease", "[visited]") {
    VisitedListPool pool;
    const VisitedList* first_address = nullptr;
    {
        const VisitedListPool::Lease lease = pool.acquire(8);
        first_address = &*lease;
        lease->mark(5);
    }

    const VisitedListPool::Lease reused = pool.acquire(8);

    REQUIRE(&*reused == first_address);
    REQUIRE(reused->mark(5));
}

TEST_CASE("concurrent leases never share a list", "[visited]") {
    VisitedListPool pool;

    const VisitedListPool::Lease first = pool.acquire(8);
    const VisitedListPool::Lease second = pool.acquire(8);

    REQUIRE(&*first != &*second);
}
