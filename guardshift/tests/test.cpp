#include "guardshift.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {
using namespace guardshift;
using Guards = std::vector<std::vector<Term>>;
constexpr Value min_value = std::numeric_limits<Value>::min();
constexpr Value max_value = std::numeric_limits<Value>::max();

void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

template <class Operation>
void require_throw(Operation operation, const std::string& message) {
    try {
        operation();
    } catch (const std::exception&) {
        return;
    }
    throw std::runtime_error(message);
}

bool matches(const std::vector<Value>& values, const std::vector<Term>& terms) {
    return std::all_of(terms.begin(), terms.end(), [&](const Term& t) {
        return t.low <= values.at(t.signal) && values.at(t.signal) <= t.high;
    });
}

// Full recomputation after each whole patch supplies the independent oracle.
struct Replay {
    std::vector<Value> values;
    Guards guards;
    Index index;

    Replay(std::vector<Value> initial, Guards definitions)
        : values(std::move(initial)), guards(std::move(definitions)), index(values, guards) {
        inspect("initial");
    }

    void inspect(const std::string& where) const {
        for (Id id = 0; id < guards.size(); ++id) {
            require(index.active(id) == matches(values, guards[id]),
                    where + ": wrong state for guard " + std::to_string(id));
        }
    }

    void advance(const std::vector<Update>& patch, const std::string& where) {
        auto next = values;
        for (const auto& u : patch) next.at(u.signal) = u.value;
        const auto changes = index.apply(patch);
        std::size_t pos = 0;
        for (Id id = 0; id < guards.size(); ++id) {
            const bool active = matches(next, guards[id]);
            if (matches(values, guards[id]) == active) continue;
            require(pos < changes.size(), where + ": missing change");
            require(changes[pos].guard == id && changes[pos].active == active,
                    where + ": wrong or unsorted change at " + std::to_string(pos));
            ++pos;
        }
        require(pos == changes.size(), where + ": duplicate or transient change");
        values = std::move(next);
        inspect(where);
    }
};

void edges() {
    Replay empty({}, {{}, {}});
    empty.advance({}, "no signals");
    Replay unused({0, 1}, {});
    unused.advance({{1, 2}}, "unused signal");
    Replay endpoints(
        {min_value, max_value, 0},
        {{{0, min_value, min_value}}, {{0, max_value, max_value}},
         {{0, min_value, max_value}}, {{0, -1, 1}, {1, 4, max_value}},
         {{1, 4, 4}}, {{0, 0, 0}, {2, 0, 0}}, {},
         {{0, -1, 1}, {1, 4, max_value}}, {{0, min_value + 1, max_value - 1}}});
    for (const auto v : {min_value + 1, Value(-1), Value(0), Value(1), Value(2),
                         max_value - 1, max_value, min_value, max_value}) {
        endpoints.advance({{0, v}}, "closed endpoints and full-range jumps");
    }
    endpoints.advance({{0, 0}, {1, 4}, {2, 1}}, "shared and coincident endpoints");
    endpoints.advance({{0, max_value}, {1, min_value}, {0, 0}, {1, 4}},
                      "last write wins, net no change");
    endpoints.advance({}, "empty patch");
    Replay swap({0, 1}, {{{0, 0, 0}, {1, 0, 0}}});
    swap.advance({{1, 0}, {0, 1}}, "ignore transient all-true state");
    swap.advance({{0, 0}, {1, 1}}, "reverse atomic swap");
    swap.advance({{0, 1}, {0, 0}, {1, 0}}, "duplicate writes");
}

void invalid_input() {
    require_throw([] { Index bad({1}, {{{1, 0, 2}}}); }, "unknown signal accepted");
    require_throw([] { Index bad({1}, {{{0, 2, 1}}}); }, "reversed bounds accepted");
    require_throw([] { Index bad({1}, {{{0, 1, 1}, {0, 2, 3}}}); },
                  "duplicate signal accepted");
    Replay replay({0, 0}, {{{0, 1, 1}}, {{1, 1, 1}}, {{0, 1, 1}, {1, 1, 1}}});
    for (const Id invalid : {Id(2), std::numeric_limits<Id>::max()}) {
        require_throw([&] { replay.index.apply({{0, 1}, {invalid, 9}, {1, 1}}); },
                      "invalid patch accepted");
        replay.inspect("failed patch changed state");
        replay.advance({{1, 1}, {0, 1}}, "valid patch after rejection");
        replay.advance({{1, 0}, {0, 0}}, "reset after rejection");
    }
}

void wide_conjunction() {
    constexpr Id count = 65537;
    Guards guards(1);
    std::vector<Update> patch;
    for (Id i = 0; i < count; ++i) {
        guards[0].push_back({i, 1, 1});
        patch.push_back({i, 1});
    }
    Replay wide(std::vector<Value>(count, 0), std::move(guards));
    wide.advance({patch.back()}, "65536 terms still false");
    wide.advance(patch, "all 65537 terms true");
    wide.advance({{0, min_value}, {count - 1, max_value}}, "leave both ends");
}

void exhaustive_two_signal_domain() {
    Guards guards;
    for (Value a = -1; a <= 1; ++a) {
        for (Value b = a; b <= 1; ++b) {
            for (Value c = -1; c <= 1; ++c) {
                for (Value d = c; d <= 1; ++d) {
                    guards.push_back({{0, a, b}, {1, c, d}});
                }
            }
        }
    }
    Replay replay({0, 0}, std::move(guards));
    // Include points beyond both sides of every bounded interval.
    for (Value x = -2; x <= 2; ++x) {
        for (Value y = -2; y <= 2; ++y) {
            for (Value u = -2; u <= 2; ++u) {
                for (Value v = -2; v <= 2; ++v) {
                    replay.advance({{0, x}, {1, y}}, "exhaustive start");
                    replay.advance({{0, u}, {1, v}}, "exhaustive end");
                }
            }
        }
    }
}

Value sample(std::mt19937_64& rng) {
    switch (rng() % 12) {
        case 0: return min_value;
        case 1: return min_value + 1;
        case 2: return max_value;
        case 3: return max_value - 1;
        default: return static_cast<Value>(rng() % 65) - 32;
    }
}

void random_replay(std::uint64_t seed) {
    std::mt19937_64 rng(seed);
    constexpr Id signals = 13;
    constexpr Id guard_count = 101;
    std::vector<Id> order(signals);
    std::iota(order.begin(), order.end(), Id(0));
    std::vector<Value> values(signals);
    for (auto& v : values) v = sample(rng);
    Guards guards(guard_count);
    for (auto& g : guards) {
        std::shuffle(order.begin(), order.end(), rng);
        const auto length = rng() % 7;
        for (std::size_t i = 0; i < length; ++i) {
            const Value a = sample(rng), b = sample(rng);
            g.push_back({order[i], std::min(a, b), std::max(a, b)});
        }
    }
    // Reverse guard registration and term order to expose ordering dependencies.
    auto permuted = guards;
    std::reverse(permuted.begin(), permuted.end());
    for (auto& g : permuted) std::reverse(g.begin(), g.end());
    Replay forward(values, guards), reverse(values, permuted);
    for (int step = 0; step < 400; ++step) {
        const auto where = "seed " + std::to_string(seed) + ", patch " + std::to_string(step);
        std::vector<Update> patch;
        const auto length = rng() % (signals * 2 + 1);
        for (std::size_t j = 0; j < length; ++j) {
            patch.push_back({static_cast<Id>(rng() % signals), sample(rng)});
        }
        forward.advance(patch, where);
        // Distinct-signal writes commute; preserve order among writes to one signal.
        std::stable_sort(patch.begin(), patch.end(),
                         [](const Update& a, const Update& b) { return a.signal > b.signal; });
        reverse.advance(patch, where + " permuted");
    }
}
} // namespace

int main() {
    try {
        edges();
        invalid_input();
        wide_conjunction();
        exhaustive_two_signal_domain();
        for (const auto seed : {0ULL, 239ULL, 0x5EED1234ULL, 0xDEADBEEF1234ULL})
            random_replay(seed);
        std::cout << "Guardshift correctness tests passed\n";
    } catch (const std::exception& err) {
        std::cerr << err.what() << '\n';
        return EXIT_FAILURE;
    }
}
