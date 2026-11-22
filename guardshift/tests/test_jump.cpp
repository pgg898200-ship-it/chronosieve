#include "guardshift.hpp"

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <new>
#include <stdexcept>
#include <utility>
#include <vector>

// apply() preallocates its scratch space. Fail a result-vector allocation to
// exercise the reverse traversal after a patch has already crossed boundaries.
namespace {
int allocations_before_failure = -1;
}

void* operator new(std::size_t bytes) {
    if (allocations_before_failure == 0) throw std::bad_alloc();
    if (allocations_before_failure > 0) --allocations_before_failure;
    if (void* p = std::malloc(bytes ? bytes : 1)) return p;
    throw std::bad_alloc();
}
void operator delete(void* p) noexcept { std::free(p); }
void operator delete(void* p, std::size_t) noexcept { std::free(p); }

namespace {
using namespace guardshift;
using Guards = std::vector<std::vector<Term>>;
using Patch = std::vector<Update>;
constexpr Value lowest = std::numeric_limits<Value>::min();
constexpr Value highest = std::numeric_limits<Value>::max();

bool matches(const std::vector<Term>& guard, const std::vector<Value>& values) {
    for (const Term t : guard) {
        if (values.at(t.signal) < t.low || values.at(t.signal) > t.high) return false;
    }
    return true;
}

struct Replay {
    const Guards& guards;
    std::vector<Value> values;
    Index index;

    Replay(const Guards& guards, std::vector<Value> initial)
        : guards(guards), values(std::move(initial)), index(values, guards) {}

    void check(const Patch& patch, const std::vector<Change>& result) {
        auto next = values;
        for (const Update u : patch) next.at(u.signal) = u.value;
        std::size_t pos = 0;
        for (Id g = 0; g < guards.size(); ++g) {
            const bool active = matches(guards[g], next);
            if (index.active(g) != active) throw std::runtime_error("active state mismatch");
            if (active == matches(guards[g], values)) continue;
            if (pos == result.size() || result[pos].guard != g || result[pos].active != active)
                throw std::runtime_error("incorrect or missing transition");
            ++pos;
        }
        if (pos != result.size()) throw std::runtime_error("unexpected transition");
        values.swap(next);
    }

    void advance(const Patch& patch) { check(patch, index.apply(patch)); }
};

void threshold_edges() {
    for (int n : {127, 128, 129, 257}) {
        Guards guards;
        for (int i = 0; i < n; ++i) guards.push_back({{0, i, highest}});
        Replay replay(guards, {-1});
        replay.advance({{0, n}});
        replay.advance({{0, -1}});
        // Cross precisely 128 then 129 edges when the index permits both.
        replay.advance({{0, 127}});
        replay.advance({{0, -1}});
        replay.advance({{0, 128}});
        replay.advance({{0, -1}});
    }
}

void mixed_intervals() {
    Guards guards{{}, {{0, lowest, highest}}, {{0, lowest, lowest}}, {{0, highest, highest}}};
    for (Value x = -256; x <= 256; ++x) {
        guards.push_back({{0, x, x}});
        guards.push_back({{0, x, x + (x + 256) % 149}, {1, -x - 57, -x}});
        guards.push_back({{0, lowest, x}, {1, -x, highest}});
        guards.push_back({{0, x, highest}, {1, lowest, -x}});
    }
    Replay replay(guards, {lowest, highest});
    const Patch jump{{1, lowest}, {0, 17}, {0, highest}};
    // Move both cursors in opposite directions and fail each successive
    // allocation. Empty and local patches probe the state left by rollback.
    int failed = 0;
    for (int allowance = 0; ; ++allowance) {
        allocations_before_failure = allowance;
        std::vector<Change> result;
        try {
            result = replay.index.apply(jump);
        } catch (const std::bad_alloc&) {
            allocations_before_failure = -1;
            ++failed;
            replay.advance({});
            replay.advance({{0, lowest + 1}, {1, highest - 1}});
            replay.advance({{1, highest}, {0, lowest}});
            continue;
        }
        allocations_before_failure = -1;
        replay.check(jump, result);
        break;
    }
    if (!failed) throw std::runtime_error("allocation failure was not exercised");

    const std::vector<Value> targets{
        lowest, lowest + 1, -257, -256, -193, -32, -1, 0, 1, 31, 128, 256, 257, highest - 1, highest
    };
    for (Value a : targets) {
        for (Value b : targets) {
            replay.advance({{0, a}, {1, b}});
            replay.advance({{1, a}, {0, b}});
        }
    }
}
} // namespace

int main() {
    threshold_edges();
    mixed_intervals();
    std::cout << "Guardshift jump and rollback tests passed\n";
}
