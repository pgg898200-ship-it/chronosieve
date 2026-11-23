// Compare exact event-driven schedulers, including crossings between
// arrivals. Build from the repository root with C++17.
#include "../include/kthwake.hpp"

#include <algorithm>
#include <array>
#include <charconv>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

using U = std::uint64_t;
using Id = std::uint32_t;
using Time = std::optional<U>;
using Guard = kthwake::Guard;
using Tick = std::pair<U, U>;
using Edge = std::tuple<U, Id, bool>;
constexpr U umax = std::numeric_limits<U>::max();

// Both baselines share a single timestamp ring with constant-time rank access.
// The benchmark sizes it to the largest rank in each input fixture.
struct Timestamps {
    std::vector<U> buf;
    std::size_t tail = 0, count = 0;
    explicit Timestamps(std::size_t max_rank) : buf(max_rank) {}
    void append(U now, U n) {
        std::size_t length = static_cast<std::size_t>(std::min<U>(buf.size(), n));
        std::size_t first = std::min(length, buf.size() - tail);
        std::fill_n(buf.begin() + tail, first, now);
        std::fill_n(buf.begin(), length - first, now);
        tail += length;
        if (tail >= buf.size()) tail -= buf.size();
        count = std::min(count + length, buf.size());
    }
    U kth(Id k) const {
        auto pos = tail >= k ? tail - k : buf.size() - (k - tail);
        return buf[pos];
    }
};

// Replace all timers in bulk on an arrival via linear-time heapify, since
// every eligible rank's timestamp may change. During idle time update only
// the heap entries at its minimum. Neither stale nodes nor polling ticks.
struct TimerHeap {
    using Item = std::pair<U, std::size_t>;
    std::vector<Item> items;
    std::greater<Item> comp;
    Time next() const { return items.empty() ? Time{} : Time{items.front().first}; }
    void build() { std::make_heap(items.begin(), items.end(), comp); }
    std::size_t pop() {
        std::pop_heap(items.begin(), items.end(), comp);
        auto id = items.back().second;
        items.pop_back();
        return id;
    }
    void push(U t, std::size_t g) {
        items.emplace_back(t, g);
        std::push_heap(items.begin(), items.end(), comp);
    }
    // Specialized comparator receives the same root-replacement optimization
    // as KthWake. The next key of an idle rank strictly increases.
    void replace_root(Time due) {
        auto replacement = items.front();
        if (due) replacement.first = *due;
        else {
            replacement = items.back();
            items.pop_back();
            if (items.empty()) return;
        }
        std::size_t i = 0;
        auto child = i * 2 + 1;
        while (child < items.size()) {
            if (child + 1 < items.size() && items[child + 1].first < items[child].first)
                ++child;
            if (replacement.first <= items[child].first) break;
            items[i] = items[child];
            i = child;
            child = i * 2 + 1;
        }
        items[i] = replacement;
    }
};

struct Rule { Id k; U w; Id id; };
struct Group { Id k; std::size_t begin, end, cut; };

// Per-pair timers share duplicates, and per-rank timers additionally group
// widths. Each group maintains a sorted suffix of active subscribers.
template<bool by_rank, bool single_sift = false>
class HeapScheduler {
    Timestamps stamps;
    TimerHeap heap;
    std::vector<Rule> rules;
    std::vector<Group> groups;

    template<class Emit>
    Time observe(std::size_t g, U now, Emit& emit) {
        auto& group = groups[g];
        U stamp = stamps.kth(group.k);
        U elapsed = now - stamp;
        while (group.cut < group.end && rules[group.cut].w <= elapsed)
            emit(rules[group.cut++].id, false);
        while (group.cut > group.begin && rules[group.cut - 1].w > elapsed)
            emit(rules[--group.cut].id, true);
        if (group.cut < group.end && rules[group.cut].w <= umax - stamp)
            return stamp + rules[group.cut].w;
        return {};
    }

public:
    explicit HeapScheduler(const std::vector<Guard>& gs)
        : stamps(std::max_element(gs.begin(), gs.end(),
                     [](auto a, auto b) { return a.k < b.k; })->k) {
        for (std::size_t i = 0; i < gs.size(); ++i)
            rules.push_back({gs[i].k, gs[i].window, static_cast<Id>(i)});
        std::sort(rules.begin(), rules.end(), [](auto a, auto b) {
            return std::tie(a.k, a.w, a.id) < std::tie(b.k, b.w, b.id);
        });
        for (std::size_t i = 0; i < rules.size();) {
            std::size_t j = i + 1;
            while (j < rules.size() && rules[i].k == rules[j].k
                   && (by_rank || rules[i].w == rules[j].w)) ++j;
            groups.push_back({rules[i].k, i, j, j});
            i = j;
        }
        heap.items.reserve(groups.size());
    }

    Time next_expiry() const { return heap.next(); }

    template<class Emit>
    void step(U now, U n, Emit emit) {
        if (n) {
            stamps.append(now, n);
            heap.items.clear();
            for (std::size_t i = 0; i < groups.size(); ++i) {
                if (groups[i].k > stamps.count) break;
                if (auto due = observe(i, now, emit)) heap.items.emplace_back(*due, i);
            }
            heap.build();
        } else {
            while (!heap.items.empty() && heap.items.front().first <= now) {
                if constexpr (single_sift) {
                    auto i = heap.items.front().second;
                    heap.replace_root(observe(i, now, emit));
                } else {
                    auto i = heap.pop();
                    if (auto due = observe(i, now, emit)) heap.push(*due, i);
                }
            }
        }
    }
};

U unsigned_decimal(const std::string& token) {
    U value;
    const char* end = token.data() + token.size();
    auto result = std::from_chars(token.data(), end, value);
    if (result.ec != std::errc{} || result.ptr != end)
        throw std::invalid_argument("expected an unsigned decimal integer");
    return value;
}

template<class Add>
void read_pairs(const char* path, Add add) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error(std::string("cannot open ") + path);
    bool any = false;
    std::size_t line_number = 0;
    std::string line;
    while (std::getline(f, line)) {
        ++line_number;
        std::istringstream row(line);
        std::string first, second, extra;
        if (!(row >> first)) continue;
        try {
            if (!(row >> second) || row >> extra)
                throw std::invalid_argument("expected two columns");
            add(unsigned_decimal(first), unsigned_decimal(second));
            any = true;
        } catch (const std::exception& error) {
            throw std::runtime_error(std::string(path) + ":" + std::to_string(line_number)
                                     + ": " + error.what());
        }
    }
    if (f.bad()) throw std::runtime_error(std::string("read failed: ") + path);
    if (!any) throw std::runtime_error(std::string("empty fixture: ") + path);
}

template<class Engine, class Emit>
U drive(Engine& engine, const std::vector<Tick>& ticks, Emit sink) {
    U idle = 0;
    auto step = [&](U t, U n) {
        engine.step(t, n, [&](Id id, bool active) { sink(t, id, active); });
    };
    for (auto [arrival, n] : ticks) {
        if (!n) continue;  // A scheduled wakeup covers every idle crossing.
        for (auto d = engine.next_expiry(); d && *d < arrival; d = engine.next_expiry()) {
            step(*d, 0);
            ++idle;
        }
        // Equal deadline and arrival are a single post-arrival observation.
        step(arrival, n);
    }
    while (auto d = engine.next_expiry()) { step(*d, 0); ++idle; }
    return idle;
}

template<class Engine>
std::vector<Edge> collect(const std::vector<Guard>& rules, const std::vector<Tick>& ticks) {
    Engine engine(rules);
    std::vector<Edge> out;
    drive(engine, ticks, [&](U t, Id id, bool state) { out.emplace_back(t, id, state); });
    std::sort(out.begin(), out.end());
    return out;
}

U mix(U v) {
    v = (v ^ (v >> 30)) * 0xbf58476d1ce4e5b9ULL;
    v = (v ^ (v >> 27)) * 0x94d049bb133111ebULL;
    return v ^ (v >> 31);
}

struct Record { U ns, idle, edges, digest; };

template<class Engine>
Record measure(const std::vector<Guard>& rules, const std::vector<Tick>& ticks) {
    Engine engine(rules);
    using Clock = std::chrono::steady_clock;
    U edges = 0, digest = 0;
    auto start = Clock::now();
    U idle = drive(engine, ticks, [&](U t, Id id, bool state) {
        digest += mix(t * 0x9e3779b97f4a7c15ULL + 2 * U(id) + state);
        ++edges;
    });
    auto end = Clock::now();
    return {static_cast<U>(std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count()),
            idle, edges, digest};
}

int main(int argc, char** argv) try {
    if (argc < 3 || argc > 4) throw std::runtime_error("usage: timers GUARDS TICKS [REPS]");
    std::vector<Guard> rules;
    read_pairs(argv[1], [&](U k, U w) {
        if (!k || k > UINT32_MAX || !w) throw std::runtime_error("invalid guard");
        rules.push_back({static_cast<Id>(k), w});
    });
    std::vector<Tick> ticks;
    read_pairs(argv[2], [&](U t, U n) {
        if (!ticks.empty() && ticks.back().first >= t)
            throw std::runtime_error("expected strictly ascending tick times");
        ticks.emplace_back(t, n);
    });
    U reps = argc == 4 ? unsigned_decimal(argv[3]) : 9;
    if (reps == 0 || reps > 1000) throw std::runtime_error("reps must be 1..1000");
    using Index = kthwake::Index;
    using PerPair = HeapScheduler<false>;
    using PerRank = HeapScheduler<true>;
    using SingleSift = HeapScheduler<true, true>;
    auto reference = collect<PerPair>(rules, ticks);
    if (reference != collect<Index>(rules, ticks) || reference != collect<PerRank>(rules, ticks)
        || reference != collect<SingleSift>(rules, ticks))
        throw std::runtime_error("full chronological transition lists disagree");
    using Runner = Record (*)(const std::vector<Guard>&, const std::vector<Tick>&);
    std::array<Runner, 4> methods{measure<Index>, measure<PerPair>, measure<PerRank>, measure<SingleSift>};
    const std::array<std::string, 4> names{"kthwake", "pair-heap", "rank-heap", "rank-heap-sift"};
    std::cout << "rep,method,elapsed_ns,idle_wakeups,edges,timestamped_digest\n";
    for (unsigned rep = 0; rep < reps; ++rep) {
        for (unsigned offset = 0; offset < methods.size(); ++offset) {
            auto i = (rep + offset) % methods.size();
            auto r = methods[i](rules, ticks);
            std::cout << rep << ',' << names[i] << ',' << r.ns << ',' << r.idle << ','
                      << r.edges << ',' << r.digest << '\n';
        }
    }
    return 0;
} catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
}
