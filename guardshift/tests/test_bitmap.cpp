#include "guardshift.hpp"

#include <cstdlib>
#include <iostream>
#include <limits>
#include <new>
#include <stdexcept>
#include <vector>

namespace {
int remaining_allocations = -1;
// Keep the implementation of replacement delete opaque to GCC's inliner.
// It otherwise warns about freeing storage allocated by replacement new.
#if defined(__GNUC__)
__attribute__((noinline))
#endif
void release_storage(void* p) noexcept { std::free(p); }
}

void* operator new(std::size_t bytes) {
    if (remaining_allocations == 0) throw std::bad_alloc();
    if (remaining_allocations > 0) --remaining_allocations;
    if (void* p = std::malloc(bytes ? bytes : 1)) return p;
    throw std::bad_alloc();
}
void operator delete(void* p) noexcept { release_storage(p); }
void operator delete(void* p, std::size_t) noexcept { release_storage(p); }

namespace {
using namespace guardshift;
using Guards = std::vector<std::vector<Term>>;
using Patch = std::vector<Update>;
constexpr auto lo = std::numeric_limits<Value>::min();
constexpr auto hi = std::numeric_limits<Value>::max();

bool evaluate(const std::vector<Term>& terms, const std::vector<Value>& values) {
    for (const auto& term : terms) {
        if (values[term.signal] < term.low || values[term.signal] > term.high) return false;
    }
    return true;
}

Guards make_guards(Id count, Id fields) {
    Guards guards(count);
    for (Id i = 0; i < count; ++i) {
        if (i % 19 == 0) continue; // An empty conjunction is always true.
        guards[i].push_back({0, 0, hi});
        if (fields >= 2 && i % 3 == 0) guards[i].push_back({1, lo, 0});
        if (fields >= 3 && i % 5 == 0) guards[i].push_back({2, 2, 5});
        if (fields >= 4 && i % 7 == 0) guards[i].push_back({3, lo, lo});
    }
    // Set bits on both sides of several 64-bit word boundaries.
    for (Id g = 63; g + 1 < count; g += 64) {
        guards[g] = {{0, hi, hi}};
        guards[g + 1] = {{0, lo, lo}};
    }
    guards[1] = {{0, lo, hi}};
    guards.back() = {{0, 5, hi}};
    return guards;
}

void exercise(Id count, Id fields) {
    const Guards guards = make_guards(count, fields);
    std::vector<Value> values(fields, lo);
    Index index(values, guards);
    auto compare = [&](const Patch& patch, const std::vector<Change>& changes) {
        auto next = values;
        for (auto u : patch) next.at(u.signal) = u.value;
        std::size_t pos = 0;
        for (Id g = 0; g < count; ++g) {
            const auto active = evaluate(guards[g], next);
            if (index.active(g) != active) throw std::runtime_error("incorrect active bit");
            if (active == evaluate(guards[g], values)) continue;
            if (pos == changes.size() || changes[pos].guard != g || changes[pos].active != active)
                throw std::runtime_error("incorrect changed bit");
            ++pos;
        }
        if (pos != changes.size()) throw std::runtime_error("extra changed bit");
        values.swap(next);
    };
    auto apply = [&](const Patch& patch) { compare(patch, index.apply(patch)); };
    apply({});

    // Fail each successive result allocation, including growth after earlier
    // bitmap words have been filled. Retry from exactly the same input state.
    const Patch flip{{0, 0}, {0, hi}};
    int failures = 0;
    for (int budget = 0; ; ++budget) {
        std::vector<Change> changes;
        remaining_allocations = budget;
        try {
            changes = index.apply(flip);
        } catch (const std::bad_alloc&) {
            remaining_allocations = -1;
            ++failures;
            apply({});
            apply({{0, lo + 1}});
            apply({{0, lo}});
            continue;
        }
        remaining_allocations = -1;
        compare(flip, changes);
        break;
    }
    if (failures == 0) throw std::runtime_error("allocation failure was not exercised");

    try {
        index.apply({{0, lo}, {fields, hi}});
        throw std::runtime_error("invalid field accepted");
    } catch (const std::out_of_range&) {}
    compare({}, {});
    try {
        (void)index.active(count);
        throw std::runtime_error("invalid guard accepted");
    } catch (const std::out_of_range&) {}

    const std::vector<Value> points{lo, lo + 1, -1, 0, 1, 2, 3, 5, 6, hi - 1, hi};
    for (auto x : points) {
        for (auto y : points) {
            Patch patch;
            for (Id f = 0; f < fields; ++f) {
                patch.push_back({f, hi});
                patch.push_back({f, f % 2 == 0 ? x : y});
            }
            apply(patch);
        }
    }
}
} // namespace

int main() {
    for (Id fields : {1, 2, 3, 4}) {
        for (Id count : {256, 257, 319, 320, 321}) exercise(count, fields);
    }
    std::cout << "Guardshift bitmap tests passed\n";
}
