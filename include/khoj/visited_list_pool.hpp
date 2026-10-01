#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <vector>

namespace khoj {

class VisitedList {
public:
    void begin_query(std::size_t element_count);

    bool mark(std::uint32_t id) {
        if (stamps_[id] == current_stamp_) {
            return false;
        }
        stamps_[id] = current_stamp_;
        return true;
    }

private:
    std::vector<std::uint16_t> stamps_;
    std::uint16_t current_stamp_ = 0;
};

class VisitedListPool {
public:
    class Lease {
    public:
        Lease(VisitedListPool& pool, std::unique_ptr<VisitedList> list);
        Lease(Lease&& other) noexcept;
        Lease(const Lease&) = delete;
        Lease& operator=(const Lease&) = delete;
        Lease& operator=(Lease&&) = delete;
        ~Lease();

        VisitedList& operator*() const { return *list_; }
        VisitedList* operator->() const { return list_.get(); }

    private:
        VisitedListPool* pool_;
        std::unique_ptr<VisitedList> list_;
    };

    VisitedListPool() = default;
    VisitedListPool(const VisitedListPool&) {}
    VisitedListPool& operator=(const VisitedListPool&) { return *this; }

    Lease acquire(std::size_t element_count);

private:
    void release(std::unique_ptr<VisitedList> list);

    std::mutex mutex_;
    std::vector<std::unique_ptr<VisitedList>> free_lists_;
};

}
