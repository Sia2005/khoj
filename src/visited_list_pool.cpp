#include "khoj/visited_list_pool.hpp"

#include <algorithm>
#include <utility>

namespace khoj {

void VisitedList::begin_query(std::size_t element_count) {
    if (stamps_.size() < element_count) {
        stamps_.resize(element_count, 0);
    }
    ++current_stamp_;
    if (current_stamp_ == 0) {
        std::fill(stamps_.begin(), stamps_.end(), 0);
        current_stamp_ = 1;
    }
}

VisitedListPool::Lease::Lease(VisitedListPool& pool, std::unique_ptr<VisitedList> list)
    : pool_(&pool), list_(std::move(list)) {}

VisitedListPool::Lease::Lease(Lease&& other) noexcept
    : pool_(other.pool_), list_(std::move(other.list_)) {}

VisitedListPool::Lease::~Lease() {
    if (list_) {
        pool_->release(std::move(list_));
    }
}

VisitedListPool::Lease VisitedListPool::acquire(std::size_t element_count) {
    std::unique_ptr<VisitedList> list;
    {
        const std::lock_guard<std::mutex> guard(mutex_);
        if (!free_lists_.empty()) {
            list = std::move(free_lists_.back());
            free_lists_.pop_back();
        }
    }
    if (!list) {
        list = std::make_unique<VisitedList>();
    }
    list->begin_query(element_count);
    return Lease(*this, std::move(list));
}

void VisitedListPool::release(std::unique_ptr<VisitedList> list) {
    const std::lock_guard<std::mutex> guard(mutex_);
    free_lists_.push_back(std::move(list));
}

}
