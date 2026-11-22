#include "kthwake.hpp"

#include <algorithm>
#include <cstdint>
#include <exception>
#include <iostream>
#include <limits>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {
using kthwake::Guard;
using kthwake::Index;
using Tick = std::uint64_t;
using Event = std::pair<std::uint32_t, bool>;
using Arrival = std::pair<Tick, std::uint64_t>;
constexpr Tick max_tick = std::numeric_limits<Tick>::max();
std::uint64_t checked_steps = 0;

void require(bool ok, const std::string& why) {
    if (!ok) throw std::runtime_error(why);
}

// Keep the complete arrival history. Evaluate the written predicate independently
// for each guard. Subtraction avoids wraparound at either end of the clock.
class Oracle {
public:
    explicit Oracle(const std::vector<Guard>& guards)
        : guards_(guards), active_(guards.size(), false) {}

    std::vector<Event> step(Tick now, std::uint64_t count) {
        if (count) arrivals_.emplace_back(now, count);
        std::vector<Event> transitions;
        for (std::size_t id = 0; id < guards_.size(); ++id) {
            bool active = evaluate(guards_[id], now);
            if (active != active_[id]) transitions.emplace_back(id, active);
            active_[id] = active;
        }
        return transitions;
    }

    // In idle time every active predicate is monotone. Binary search the
    // first changed observation, independently of the index's deadline logic.
    std::optional<Tick> next_expiry(Tick now) const {
        Tick longest = 0;
        for (std::size_t id = 0; id < guards_.size(); ++id) {
            if (active_[id]) longest = std::max(longest, guards_[id].window);
        }
        if (!longest || now == max_tick) return std::nullopt;
        Tick hi = now + std::min(longest, max_tick - now);
        if (!changed_at(hi)) return std::nullopt;
        Tick lo = now + 1;
        while (lo < hi) {
            Tick mid = lo + (hi - lo) / 2;
            if (changed_at(mid)) hi = mid;
            else lo = mid + 1;
        }
        return lo;
    }

private:
    bool evaluate(Guard guard, Tick now) const {
        std::uint64_t needed = guard.k;
        for (auto it = arrivals_.rbegin(); it != arrivals_.rend(); ++it) {
            if (now - it->first >= guard.window) break;
            if (it->second >= needed) return true;
            needed -= it->second;
        }
        return false;
    }

    bool changed_at(Tick t) const {
        for (std::size_t id = 0; id < guards_.size(); ++id) {
            if (active_[id] && !evaluate(guards_[id], t)) return true;
        }
        return false;
    }

    std::vector<Guard> guards_;
    std::vector<bool> active_;
    std::vector<Arrival> arrivals_;
};

class Replay {
public:
    Replay(const std::vector<Guard>& guards, std::string label)
        : index_(guards), oracle_(guards), label_(std::move(label)) {
        require(!index_.next_expiry(), label_ + " initial expiry");
    }

    void step(Tick now, std::uint64_t count) {
        std::vector<Event> got;
        index_.step(now, count, [&](std::uint32_t id, bool active) {
            got.emplace_back(id, active);
        });
        auto expected = oracle_.step(now, count);
        std::sort(got.begin(), got.end());
        const std::string context = label_ + " step=" + std::to_string(++steps_)
            + " time=" + std::to_string(now) + " count=" + std::to_string(count);
        require(got == expected, context + " transition mismatch");
        auto expected_deadline = oracle_.next_expiry(now);
        auto got_deadline = index_.next_expiry();
        auto show = [](std::optional<Tick> t) { return t ? std::to_string(*t) : "none"; };
        require(got_deadline == expected_deadline, context + " next expiry="
                + show(got_deadline) + " expected=" + show(expected_deadline));
        ++checked_steps;
    }

    void drain() {
        while (auto due = index_.next_expiry()) step(*due, 0);
    }

    Index& index() { return index_; }

private:
    Index index_;
    Oracle oracle_;
    std::string label_;
    unsigned steps_ = 0;
};

template<class F>
void require_invalid(F operation, const std::string& label) {
    try {
        operation();
    } catch (const std::invalid_argument&) {
        return;
    }
    throw std::runtime_error(label + " expected std::invalid_argument");
}

void boundaries() {
    for (Guard bad : std::vector<Guard>{{0, 4}, {3, 0}, {0, 0}}) {
        require_invalid([&] { Index index(std::vector<Guard>{{1, 1}, bad}); },
                        "invalid guard");
    }

    Replay empty({}, "empty registry");
    empty.step(0, max_tick);
    empty.step(max_tick, 0);

    // Coalesced expiry and replacement at the same observation, repeated
    // timestamps, duplicate predicates and jumps over intermediate deadlines.
    Replay sequence({{1, 1}, {1, 4}, {2, 4}, {2, 4}, {4, 4}, {2, 6}},
                    "boundary sequence");
    for (auto event : std::vector<Arrival>{{0, 1}, {0, 1}, {0, 0}, {1, 1},
             {4, 2}, {4, 0}, {5, 0}, {5, 3}, {14, 1}, {14, 0}}) {
        sequence.step(event.first, event.second);
    }
    sequence.drain();

    Replay clocks({{1, 1}, {2, 4}, {3, max_tick}}, "backward rejection");
    clocks.step(5, 1);
    bool emitted = false;
    require_invalid([&] {
        clocks.index().step(4, 9, [&](std::uint32_t, bool) { emitted = true; });
    }, "backward step");
    require(!emitted, "backward step emitted a callback");
    clocks.step(5, 1);
    clocks.step(6, 0);
    clocks.drain();

    Replay end({{1, 1}, {1, 5}, {2, 2}, {2, max_tick}, {3, max_tick}},
               "last representable clock tick");
    end.step(0, 1);
    end.step(max_tick - 5, 1);
    end.step(max_tick - 2, 1);
    end.step(max_tick - 1, 0);
    end.step(max_tick, 0);
    end.step(max_tick, max_tick);

    // These values require compressed arrival handling and saturating
    // arithmetic even when the same timestamp occurs twice.
    constexpr std::uint32_t max_k = std::numeric_limits<std::uint32_t>::max();
    Replay bursts({{1, 4}, {max_k, 1}, {max_k - 1, 7}, {max_k, 7}},
                  "compressed bursts");
    bursts.step(10, 1'000'000'000'000'000ULL);
    bursts.step(10, max_tick);
    bursts.step(11, max_k - 1);
    bursts.step(11, 1);
    bursts.step(17, 0);
    bursts.step(18, max_tick);
    bursts.drain();

    std::vector<Guard> narrow;
    for (std::uint32_t k = 1; k <= 96; ++k) {
        narrow.push_back({k, 3});
        narrow.push_back({k, 5});
    }
    Replay dense_ranks(narrow, "many ranks sharing two windows");
    for (auto point : std::vector<Arrival>{{0, 94}, {1, 1}, {1, 2}, {3, 9},
            {4, 63}, {8, 0}, {max_tick - 5, max_tick}, {max_tick - 3, 32},
            {max_tick - 1, 0}, {max_tick, 0}, {max_tick, max_tick}}) {
        dense_ranks.step(point.first, point.second);
    }

    std::vector<Guard> varied;
    for (std::uint32_t k = 1; k <= 160; ++k) {
        varied.push_back({k, k + 1});
        varied.push_back({k, k + 2});
    }
    Replay many_timers(varied, "rank timer update and timestamp representation change");
    for (auto point : std::vector<Arrival>{{0, 161}, {1, 20}, {1, 2}, {3, 65},
            {5, 1}, {20, 0}, {23, 3}, {175, 0}, {176, 100},
            {max_tick - 160, 162}, {max_tick - 157, 3}, {max_tick, 0}}) {
        many_timers.step(point.first, point.second);
    }
    many_timers.drain();

    // A caller may ask for its first deadline only after running snapshots.
    Index late(varied);
    Oracle late_oracle(varied);
    for (auto point : std::vector<Arrival>{{2, 102}, {3, 16}, {5, 0}}) {
        late.step(point.first, point.second, [](std::uint32_t, bool) {});
        late_oracle.step(point.first, point.second);
    }
    auto due = late_oracle.next_expiry(5);
    require(late.next_expiry() == due && due.has_value(), "first lazy timer build");
    Index copied = late;
    std::vector<Event> copied_edges;
    copied.step(*due, 0, [&](std::uint32_t id, bool state) { copied_edges.emplace_back(id, state); });
    std::sort(copied_edges.begin(), copied_edges.end());
    require(copied_edges == late_oracle.step(*due, 0), "copy after lazy timer build");
    require(copied.next_expiry() == late_oracle.next_expiry(*due), "copied heap minimum");
    require(late.next_expiry() == due, "original timer changed after copying");
    bool threw = false;
    try {
        late.step(*due, 0, [](std::uint32_t, bool) { throw 49; });
    } catch (int code) { threw = code == 49; }
    require(threw && late.next_expiry() == copied.next_expiry(), "throw during heap expiry");
    late.step(*due, 0, [](std::uint32_t, bool) {
        throw std::runtime_error("failed heap expiry was not committed");
    });
}

void callback_failure() {
    Index index(std::vector<Guard>{{1, 4}, {2, 6}, {1, 4}});
    struct SinkFailure {};
    unsigned calls = 0;
    bool caught = false;
    try {
        index.step(8, 2, [&](std::uint32_t, bool) {
            ++calls;
            throw SinkFailure{};
        });
    } catch (const SinkFailure&) {
        caught = true;
    }
    require(caught && calls == 1, "failed sink exception or delivery count");
    require(index.next_expiry() == 12, "failed sink first committed deadline");
    std::vector<Event> got;
    auto collect = [&](std::uint32_t id, bool active) { got.emplace_back(id, active); };
    index.step(8, 0, collect);
    require(got.empty(), "failed sink replayed dropped callbacks");
    index.step(12, 0, collect);
    std::sort(got.begin(), got.end());
    require(got == std::vector<Event>{{0, false}, {2, false}},
            "failed sink did not commit duplicate transitions");
    require(index.next_expiry() == 14, "failed sink second committed deadline");
    got.clear();
    index.step(14, 0, collect);
    require(got == std::vector<Event>{{1, false}} && !index.next_expiry(),
            "failed sink did not commit all groups");
}

void exhaustive() {
    constexpr unsigned length = 4;
    constexpr unsigned choices = 9;
    const std::vector<Guard> guards{{1, 1}, {1, 3}, {2, 1}, {2, 3}, {4, 2}, {2, 3}};
    for (unsigned code = 0; code < choices * choices * choices * choices; ++code) {
        Replay replay(guards, "exhaustive case " + std::to_string(code));
        unsigned state = code;
        Tick now = 0;
        for (unsigned position = 0; position < length; ++position) {
            unsigned choice = state % choices;
            state /= choices;
            constexpr unsigned gaps[] = {0, 1, 3};
            now += gaps[choice / 3];
            replay.step(now, choice % 3);
        }
        replay.drain();
    }
}

void randomized() {
    std::mt19937_64 rng(0x5c3492f911ULL);
    for (unsigned trial = 0; trial < 100; ++trial) {
        std::vector<Guard> guards;
        for (unsigned g = 0; g < 30; ++g) {
            guards.push_back({static_cast<std::uint32_t>(1 + rng() % 20), 1 + rng() % 40});
        }
        guards.push_back(guards.front());
        Replay replay(guards, "random case " + std::to_string(trial));
        Tick now = 0;
        for (unsigned step = 0; step < 150; ++step) {
            now += rng() % 7;
            auto count = rng() % 13;
            replay.step(now, count < 6 ? 0 : count - 5);
        }
        replay.drain();
    }
}
} // namespace

int main() {
    try {
        boundaries();
        callback_failure();
        exhaustive();
        randomized();
        std::cout << "Passed " << checked_steps << " oracle-checked observations.\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
