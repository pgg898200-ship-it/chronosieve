// g++ -std=c++17 -O3 -DNDEBUG -Iinclude bench/bench.cpp -o /tmp/guardshift-bench
#include "guardshift.hpp"
#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <limits>
#include <iostream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using guardshift::Change;
using guardshift::Id;
using guardshift::Term;
using guardshift::Update;
using guardshift::Value;
using Guards = std::vector<std::vector<Term>>;
using Clock = std::chrono::steady_clock;

namespace {

struct Workload {
    std::string name;
    std::vector<Value> initial;
    Guards guards;
    std::vector<std::vector<Update>> patches;
};

void sort_changes(std::vector<Change>& changes) {
    std::sort(changes.begin(), changes.end(), [](const Change& x, const Change& y) {
        return x.guard < y.guard;
    });
}

bool holds(const Term& t, Value v) { return t.low <= v && v <= t.high; }

class Scan {
    const Guards& guards;
    std::vector<Value> values;
    std::vector<uint8_t> enabled;

    bool evaluate(Id i) const {
        for (const auto& term : guards[i])
            if (!holds(term, values[term.signal])) return false;
        return true;
    }
public:
    Scan(const std::vector<Value>& initial, const Guards& g)
        : guards(g), values(initial), enabled(g.size()) {
        for (Id i = 0; i < g.size(); ++i) enabled[i] = evaluate(i);
    }
    bool active(Id i) const { return enabled[i]; }
    std::vector<Change> apply(const std::vector<Update>& patch) {
        for (auto u : patch) values[u.signal] = u.value;
        std::vector<Change> result;
        for (Id i = 0; i < guards.size(); ++i) {
            bool next = evaluate(i);
            if (next != bool(enabled[i])) {
                enabled[i] = next;
                result.push_back({i, next});
            }
        }
        return result;
    }
};

// Both incidence indexes remember only guards whose truth count has changed.
class Counts {
    std::vector<int> missing;
    std::vector<uint8_t> enabled, dirty;
    std::vector<Id> touched;
public:
    Counts(const std::vector<Value>& v, const Guards& g)
        : missing(g.size()), enabled(g.size()), dirty(g.size()) {
        for (Id i = 0; i < g.size(); ++i) {
            for (const auto& t : g[i]) missing[i] += !holds(t, v[t.signal]);
            enabled[i] = missing[i] == 0;
        }
    }
    bool active(Id i) const { return enabled[i]; }
    void adjust(Id i, int delta) {
        missing[i] += delta;
        if ((missing[i] == 0) != bool(enabled[i]) && !dirty[i]) {
            touched.push_back(i);
            dirty[i] = true;
        }
    }
    std::vector<Change> finish() {
        std::vector<Change> result;
        for (Id i : touched) {
            bool next = missing[i] == 0;
            if (next != bool(enabled[i])) {
                result.push_back({i, next});
                enabled[i] = next;
            }
            dirty[i] = false;
        }
        touched.clear();
        sort_changes(result);
        return result;
    }
};

class Incidence {
    struct Ref { Value low, high; Id id; };
    std::vector<std::vector<Ref>> refs;
    std::vector<Value> values;
    Counts counts;
public:
    Incidence(const std::vector<Value>& initial, const Guards& g)
        : refs(initial.size()), values(initial), counts(initial, g) {
        for (Id i = 0; i < g.size(); ++i)
            for (const auto& t : g[i]) refs[t.signal].push_back({t.low, t.high, i});
    }
    bool active(Id i) const { return counts.active(i); }
    std::vector<Change> apply(const std::vector<Update>& patch) {
        for (auto u : patch) {
            Value old = values[u.signal];
            if (old == u.value) continue;
            for (const Ref& r : refs[u.signal]) {
                bool before = r.low <= old && old <= r.high;
                bool after = r.low <= u.value && u.value <= r.high;
                if (before != after) counts.adjust(r.id, before ? 1 : -1);
            }
            values[u.signal] = u.value;
        }
        return counts.finish();
    }
};

struct Edge { Value at; Id id; };
struct Series { std::vector<Edge> low, high; };
class Edges {
    std::vector<Series> series;
    static auto first_inclusive(const std::vector<Edge>& a, Value v) {
        return std::lower_bound(a.begin(), a.end(), v, [](Edge x, Value y) { return x.at < y; });
    }
    static auto first_exclusive(const std::vector<Edge>& a, Value v) {
        return std::upper_bound(a.begin(), a.end(), v, [](Value x, Edge y) { return x < y.at; });
    }
public:
    Edges(size_t signals, const Guards& g) : series(signals) {
        for (Id i = 0; i < g.size(); ++i)
            for (auto t : g[i]) {
                series[t.signal].low.push_back({t.low, i});
                series[t.signal].high.push_back({t.high, i});
            }
        for (auto& s : series) {
            auto less = [](Edge a, Edge b) { return a.at < b.at; };
            std::sort(s.low.begin(), s.low.end(), less);
            std::sort(s.high.begin(), s.high.end(), less);
        }
    }
    template<class F>
    void visit(Id signal, Value old, Value now, F&& f) const {
        if (old == now) return;
        const auto& s = series[signal];
        auto [lo, hi] = std::minmax(old, now);
        // Inclusive bounds. Term truth increases when crossing a low upwards.
        for (auto it = first_exclusive(s.low, lo); it != s.low.end() && it->at <= hi; ++it)
            f(it->id, now > old ? -1 : 1);
        for (auto it = first_inclusive(s.high, lo); it != s.high.end() && it->at < hi; ++it)
            f(it->id, now > old ? 1 : -1);
    }
};

class EndpointCounts {
    Edges edges;
    std::vector<Value> values;
    Counts counts;
public:
    EndpointCounts(const std::vector<Value>& initial, const Guards& g)
        : edges(initial.size(), g), values(initial), counts(initial, g) {}
    bool active(Id i) const { return counts.active(i); }
    std::vector<Change> apply(const std::vector<Update>& patch) {
        for (const auto& u : patch) {
            edges.visit(u.signal, values[u.signal], u.value,
                        [&](Id id, int delta) { counts.adjust(id, delta); });
            values[u.signal] = u.value;
        }
        return counts.finish();
    }
};

// 1D baseline with no guard counters, re-evaluation, or sorting. A jump
// across both ends has no effect. Each patch in this workload sets one signal.
class Endpoint1D {
    Edges edges;
    std::vector<Term> terms;
    std::vector<Value> values;
public:
    Endpoint1D(const std::vector<Value>& v, const Guards& g) : edges(v.size(), g), values(v) {
        for (const auto& row : g) {
            if (row.size() != 1) throw std::logic_error("expected unary guard");
            terms.push_back(row.front());
        }
    }
    bool active(Id i) const { return holds(terms[i], values[terms[i].signal]); }
    std::vector<Change> apply(const std::vector<Update>& patch) {
        if (patch.size() != 1) throw std::logic_error("expected singleton patch");
        const auto u = patch.front();
        Value old = values[u.signal];
        std::vector<Change> result;
        edges.visit(u.signal, old, u.value, [&](Id i, int delta) {
            bool before = holds(terms[i], old), after = holds(terms[i], u.value);
            if (before != after && after == (delta < 0)) result.push_back({i, after});
        });
        values[u.signal] = u.value;
        sort_changes(result);
        return result;
    }
};

// Width-specialized stabbing competitor. Two point queries suffice to emit
// every changed unary interval. All measured unary inputs fit these bounded
// integer coordinates, so point - max_width is representable.
class Width1D {
    struct Row { Value low, high; Id id; };
    std::vector<Row> sorted;
    std::vector<Term> terms;
    Value max_width = 0, value;
    void emit(Value point, Value other, bool state, std::vector<Change>& result) const {
        auto it = std::lower_bound(sorted.begin(), sorted.end(), point - max_width,
                                   [](const Row& r, Value bound) { return r.low < bound; });
        for (; it != sorted.end() && it->low <= point; ++it)
            if (point <= it->high && (other < it->low || it->high < other))
                result.push_back({it->id, state});
    }
public:
    Width1D(const std::vector<Value>& v, const Guards& g) : value(v.at(0)) {
        for (Id i = 0; i < g.size(); ++i) {
            const auto& t = g[i].at(0);
            max_width = std::max(max_width, t.high - t.low);
            sorted.push_back({t.low, t.high, i});
            terms.push_back(t);
        }
        std::sort(sorted.begin(), sorted.end(), [](const Row& a, const Row& b) {
            return a.low < b.low;
        });
    }
    bool active(Id i) const { return holds(terms[i], value); }
    std::vector<Change> apply(const std::vector<Update>& patch) {
        if (patch.size() != 1) throw std::logic_error("expected singleton patch");
        std::vector<Change> result;
        Value next = patch[0].value;
        if (next != value) {
            emit(value, next, false, result);
            emit(next, value, true, result);
            sort_changes(result);
            value = next;
        }
        return result;
    }
};

// General interval-stabbing index. Sort rows by their lower endpoint and
// store the largest upper endpoint for each subtree. Query the old and new
// position, discarding intervals that hold at both. Counters implement AND.
class StabCounts {
    struct Row { Value low, high; Id id; };
    struct Tree {
        static constexpr size_t block = 16;
        std::vector<Row> rows;
        std::vector<Value> high;
        size_t leaves = 1;
        void build() {
            std::sort(rows.begin(), rows.end(), [](const Row& a, const Row& b) {
                return a.low < b.low;
            });
            while (leaves * block < rows.size()) leaves *= 2;
            high.assign(2 * leaves, std::numeric_limits<Value>::min());
            for (size_t i = 0; i < rows.size(); ++i)
                high[leaves + i / block] = std::max(high[leaves + i / block], rows[i].high);
            for (size_t i = leaves - 1; i > 0; --i) high[i] = std::max(high[2*i], high[2*i+1]);
        }
        void emit(size_t node, size_t lo, size_t hi, Value x, Value other,
                  int delta, Counts& counts) const {
            if (lo >= rows.size() || rows[lo].low > x || high[node] < x) return;
            if (hi - lo == block) {
                for (size_t j = lo; j < std::min(hi, rows.size()) && rows[j].low <= x; ++j)
                    if (rows[j].high >= x && (other < rows[j].low || other > rows[j].high))
                        counts.adjust(rows[j].id, delta);
                return;
            }
            size_t mid = lo + (hi - lo)/2;
            emit(2*node, lo, mid, x, other, delta, counts);
            emit(2*node+1, mid, hi, x, other, delta, counts);
        }
    };
    std::vector<Tree> by_signal;
    std::vector<Value> values;
    Counts counts;
public:
    StabCounts(const std::vector<Value>& initial, const Guards& g)
        : by_signal(initial.size()), values(initial), counts(initial, g) {
        for (Id i = 0; i < g.size(); ++i)
            for (auto t : g[i]) by_signal[t.signal].rows.push_back({t.low,t.high,i});
        for (auto& tree : by_signal) tree.build();
    }
    bool active(Id i) const { return counts.active(i); }
    std::vector<Change> apply(const std::vector<Update>& patch) {
        for (auto u : patch) {
            Value old = values[u.signal];
            if (old == u.value) continue;
            const Tree& t = by_signal[u.signal];
            t.emit(1, 0, t.leaves * Tree::block, old, u.value, 1, counts);
            t.emit(1, 0, t.leaves * Tree::block, u.value, old, -1, counts);
            values[u.signal] = u.value;
        }
        return counts.finish();
    }
};

// Snapshot baseline for low endpoint cardinality. Each cell between sorted
// boundaries stores the entire field's satisfaction bitmap, including a set
// bit for guards with no term on that field. Intersect current cells once.
class SlabBits {
    struct Event { Value at; Id id; bool state; };
    struct Field {
        std::vector<Value> bounds;
        std::vector<uint64_t> cells;
        size_t current = 0;
        size_t lookup(Value v) const {
            return std::upper_bound(bounds.begin(), bounds.end(), v) - bounds.begin();
        }
    };
    size_t words;
    std::vector<Field> fields;
    std::vector<uint64_t> active_bits;
    std::vector<const uint64_t*> selected;
    uint64_t intersection(size_t word) const {
        if (selected.size() == 2) return selected[0][word] & selected[1][word];
        uint64_t result = ~UINT64_C(0);
        for (const auto* bitmap : selected) result &= bitmap[word];
        return result;
    }
public:
    SlabBits(const std::vector<Value>& v, const Guards& g)
        : words((g.size()+63)/64), fields(v.size()), active_bits(words), selected(v.size()) {
        std::vector<std::vector<Event>> events(v.size());
        std::vector<std::vector<uint64_t>> base(v.size(), std::vector<uint64_t>(words, ~UINT64_C(0)));
        for (auto& b : base)
            if (g.size()%64) b.back() &= (UINT64_C(1) << (g.size()%64)) - 1;
        for (Id id = 0; id < g.size(); ++id)
            for (auto t : g[id]) {
                base[t.signal][id/64] &= ~(UINT64_C(1) << (id%64));
                events[t.signal].push_back({t.low, id, true});
                if (t.high < std::numeric_limits<Value>::max())
                    events[t.signal].push_back({t.high+1, id, false});
            }
        for (Id s = 0; s < fields.size(); ++s) {
            auto& f = fields[s];
            auto& ev = events[s];
            std::sort(ev.begin(), ev.end(), [](Event x, Event y) { return x.at < y.at; });
            auto& snapshot = base[s];
            f.cells = snapshot;
            for (size_t j = 0; j < ev.size();) {
                Value bound = ev[j].at;
                do {
                    const auto& e = ev[j++];
                    auto& word = snapshot[e.id/64];
                    uint64_t mask = UINT64_C(1) << (e.id%64);
                    if (e.state) word |= mask; else word &= ~mask;
                } while (j < ev.size() && ev[j].at == bound);
                f.bounds.push_back(bound);
                f.cells.insert(f.cells.end(), snapshot.begin(), snapshot.end());
            }
            f.current = f.lookup(v[s]);
            selected[s] = &f.cells[f.current * words];
        }
        for (size_t j = 0; j < words; ++j) active_bits[j] = intersection(j);
    }
    bool active(Id id) const { return (active_bits[id/64] >> (id%64)) & 1; }
    std::vector<Change> apply(const std::vector<Update>& patch) {
        bool changed = false;
        for (auto u : patch) {
            auto& f = fields[u.signal];
            size_t next = f.lookup(u.value);
            if (next != f.current) {
                f.current = next;
                selected[u.signal] = &f.cells[next * words];
                changed = true;
            }
        }
        std::vector<Change> result;
        if (!changed) return result;
        for (size_t j = 0; j < words; ++j) {
            uint64_t next = intersection(j);
            uint64_t delta = next ^ active_bits[j];
            active_bits[j] = next;
            while (delta) {
                unsigned bit = __builtin_ctzll(delta);
                result.push_back({static_cast<Id>(j*64 + bit), bool((next >> bit) & 1)});
                delta &= delta - 1;
            }
        }
        return result;
    }
};

uint64_t mix(uint64_t h, uint64_t x) {
    return (h ^ x) * UINT64_C(1099511628211);
}

Workload make_work(const std::string& name, Id count, unsigned steps, unsigned seed) {
    std::mt19937_64 rng(seed);
    auto draw = [&](int low, int high) -> int {
        return low + static_cast<int>(rng() % static_cast<unsigned>(high - low + 1));
    };
    bool teleport = name == "teleport";
    bool unary = name == "unary" || teleport;
    bool crowded = name == "churn";
    unsigned arity = unary ? 1 : 4;
    int extent = crowded ? 128 : 32768;
    Id signals = unary ? 1 : 32;
    Workload w{name, std::vector<Value>(signals), Guards(count), {}};
    for (auto& row : w.guards) {
        Id start = static_cast<Id>(draw(0, signals - 1));
        for (unsigned j = 0; j < arity; ++j) {
            Id s = (start + 7 * j) % signals;
            if (teleport) {
                int low = draw(0, 999900);
                row.push_back({s, low, low + draw(1, 100)});
                continue;
            }
            int center = draw(-extent, extent);
            int radius = draw(extent / 4, extent);
            row.push_back({s, center - radius, center + radius});
        }
    }
    auto values = w.initial;
    for (unsigned t = 0; t < steps; ++t) {
        std::vector<Update> patch;
        Id start = static_cast<Id>(draw(0, signals - 1));
        unsigned width = name == "atomic" ? 4 : 1;
        for (unsigned j = 0; j < width; ++j) {
            Id s = (start + j * 3) % signals;
            int delta = draw(-64, 64);
            Value next = name == "jump" || crowded ? draw(-extent, extent) :
                std::clamp<Value>(values[s] + delta, -extent, extent);
            if (teleport) next = draw(0, 1000000);
            patch.push_back({s, next});
            values[s] = next;
        }
        w.patches.push_back(std::move(patch));
    }
    return w;
}

Workload make_quantized(Id count, unsigned steps, unsigned seed) {
    std::mt19937_64 rng(seed);
    Workload w{"quantized", {32,32}, Guards(count), {}};
    for (auto& guard : w.guards)
        for (Id field = 0; field < 2; ++field) {
            Value a = rng()%64, b = rng()%64;
            guard.push_back({field, std::min(a,b), std::max(a,b)});
        }
    for (unsigned i = 0; i < steps; ++i)
        w.patches.push_back({{0, static_cast<Value>(rng()%64)}, {1, static_cast<Value>(rng()%64)}});
    return w;
}

Workload make_mixed(unsigned seed, Id rows = 100000, unsigned steps = 512) {
    // The two conditions mimic numeric partition routing with both
    // local buckets and broader regional bands, plus an eligibility gate.
    constexpr Id signals = 8;
    constexpr int domain = 1000000;
    std::mt19937_64 rng(seed);
    Workload w{"mixed_jump", std::vector<Value>(signals, domain/2), Guards(rows), {}};
    for (auto& guard : w.guards) {
        Id route = rng() % (signals/2);
        int width = rng() % 10 ? 1 + rng()%100 : 1000 + rng()%90000;
        int low = rng() % (domain-width+1);
        guard.push_back({route, low, low+width});
        Id gate = signals/2 + rng() % (signals/2);
        guard.push_back({gate, static_cast<Value>(rng()%250000),
                         domain - static_cast<Value>(rng()%250000)});
    }
    for (unsigned i = 0; i < steps; ++i) {
        Id route = rng() % (signals/2);
        w.patches.push_back({{route, static_cast<Value>(rng() % (domain+1))}});
        if (i % 8 == 0) {
            Id gate = signals/2 + rng() % (signals/2);
            w.patches.back().push_back({gate, static_cast<Value>(rng() % (domain+1))});
        }
    }
    return w;
}

Workload make_azure(const std::string& path, Id count, unsigned steps, unsigned seed) {
    std::ifstream input(path);
    if (!input) throw std::runtime_error("cannot open trace: " + path);
    std::string line;
    if (!std::getline(input, line) || line.find("field,value") != 0)
        throw std::runtime_error("expected CSV header field,value");
    // The importer emits field 0 and 1 as a pair for every request.
    constexpr size_t calibration = 128;
    std::vector<std::array<Value, 2>> samples;
    while (samples.size() < calibration + steps && std::getline(input, line)) {
        if (line.rfind("0,", 0) != 0) throw std::runtime_error("expected field 0");
        Value first = std::stoll(line.substr(2));
        if (!std::getline(input, line) || line.rfind("1,", 0) != 0)
            throw std::runtime_error("expected field 1");
        samples.push_back({first, std::stoll(line.substr(2))});
    }
    if (samples.size() <= calibration) throw std::runtime_error("trace too short");
    Workload w{"azure", {samples[calibration - 1][0], samples[calibration - 1][1]}, Guards(count), {}};
    std::array<std::vector<Value>, 2> sorted;
    for (size_t i = 0; i < calibration; ++i)
        for (Id s = 0; s < 2; ++s) sorted[s].push_back(samples[i][s]);
    for (auto& column : sorted) std::sort(column.begin(), column.end());
    std::mt19937_64 rng(seed);
    for (auto& guard : w.guards)
        for (Id s = 0; s < 2; ++s) {
            size_t lo = rng() % (calibration - 1);
            size_t hi = lo + 1 + rng() % (calibration - 1 - lo);
            guard.push_back({s, sorted[s][lo], sorted[s][hi]});
        }
    for (size_t i = calibration; i < samples.size(); ++i)
        w.patches.push_back({{0, samples[i][0]}, {1, samples[i][1]}});
    return w;
}

uint64_t ns(Clock::duration d) {
    return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(d).count());
}

using Oracle = std::vector<std::vector<uint64_t>>;
std::vector<uint64_t> canonical(const std::vector<Change>& changes) {
    std::vector<uint64_t> coded;
    coded.reserve(changes.size());
    for (const auto& c : changes) coded.push_back(2ULL * c.guard + c.active);
    for (size_t i = 1; i < changes.size(); ++i)
        if (changes[i].guard <= changes[i-1].guard)
            throw std::runtime_error("duplicate or out-of-order change");
    return coded;
}

Oracle make_oracle(const Workload& w) {
    Scan s(w.initial, w.guards);
    Oracle result;
    for (const auto& patch : w.patches) result.push_back(canonical(s.apply(patch)));
    return result;
}

template<class Engine>
void measure(const Workload& w, const Oracle& expected, const std::string& method,
             unsigned seed, unsigned rep) {
    auto start = Clock::now();
    Engine engine(w.initial, w.guards);
    uint64_t build_ns = ns(Clock::now() - start);
    std::vector<uint64_t> durations;
    durations.reserve(w.patches.size());
    uint64_t checksum = 14695981039346656037ULL, sum_ns = 0, changes = 0, updates = 0;
    for (size_t p = 0; p < w.patches.size(); ++p) {
        start = Clock::now();
        auto output = engine.apply(w.patches[p]);
        uint64_t duration = ns(Clock::now() - start);
        durations.push_back(duration);
        sum_ns += duration;
        auto coded = canonical(output);
        if (coded != expected[p]) {
            throw std::runtime_error(method + " disagreed with full scan on " + w.name +
                                     " patch " + std::to_string(p));
        }
        checksum = mix(checksum, coded.size());
        for (auto c : coded) checksum = mix(checksum, c);
        changes += coded.size();
        updates += w.patches[p].size();
    }
    auto final_values = w.initial;
    for (const auto& patch : w.patches)
        for (auto u : patch) final_values[u.signal] = u.value;
    Scan final_truth(final_values, w.guards);
    for (Id i = 0; i < w.guards.size(); ++i)
        if (engine.active(i) != final_truth.active(i))
            throw std::runtime_error(method + " disagreed on final active state");
    std::sort(durations.begin(), durations.end());
    auto percentile = [&](size_t pc) { return durations[(pc * durations.size() + 99) / 100 - 1]; };
    std::cout << std::setprecision(12)
              << "{\"method\":\"" << method << "\",\"scenario\":\"" << w.name
              << "\",\"seed\":" << seed << ",\"rep\":" << rep
              << ",\"guards\":" << w.guards.size() << ",\"signals\":" << w.initial.size()
              << ",\"patches\":" << w.patches.size() << ",\"updates\":" << updates
              << ",\"changes\":" << changes << ",\"checksum\":\"" << checksum << "\""
              << ",\"build_ns\":" << build_ns << ",\"elapsed_ns\":" << sum_ns
              << ",\"ns_per_patch\":" << double(sum_ns) / w.patches.size()
              << ",\"p50_ns\":" << percentile(50) << ",\"p99_ns\":" << percentile(99)
              << "}\n";
}

void run(const Workload& w, unsigned seed, unsigned repeats) {
    auto oracle = make_oracle(w);
    std::vector<unsigned> order{0, 1, 2, 3};
    if (w.name == "mixed_jump") order.push_back(6);
    if (w.name == "azure" || w.name == "quantized") order.push_back(7);
    if (w.name == "unary" || w.name == "teleport") {
        order.push_back(4);
        order.push_back(5);
    }
    for (unsigned rep = 0; rep < repeats; ++rep) {
        for (unsigned i : order) switch (i) {
            case 0: measure<guardshift::Index>(w, oracle, "guardshift", seed, rep); break;
            case 1: measure<Incidence>(w, oracle, "incidence_count", seed, rep); break;
            case 2: measure<EndpointCounts>(w, oracle, "endpoint_count", seed, rep); break;
            case 3: measure<Scan>(w, oracle, "full_scan", seed, rep); break;
            case 7: measure<SlabBits>(w, oracle, "slab_bits", seed, rep); break;
            case 6: measure<StabCounts>(w, oracle, "stabbing_count", seed, rep); break;
            case 5: measure<Width1D>(w, oracle, "bounded_width_1d", seed, rep); break;
            case 4: measure<Endpoint1D>(w, oracle, "direct_1d", seed, rep); break;
        }
        std::rotate(order.begin(), order.begin() + 1, order.end());
    }
}
} // namespace

#ifndef GUARDSHIFT_BENCH_LIBRARY
int main(int argc, char** argv) {
    try {
        std::string flag = argc > 1 ? argv[1] : "--quick";
        if ((argc != 1 && argc != 2 && argc != 4) ||
            (argc == 4 && std::string(argv[2]) != "--trace") ||
            (flag != "--full" && flag != "--quick" && flag != "--million" && flag != "--mixed")) {
            std::cerr << "usage: guardshift-bench [--quick|--full|--million|--mixed] [--trace replay/azure-code.csv]\n";
            return 2;
        }
        if (flag == "--mixed") {
            if (argc == 4) throw std::runtime_error("--mixed uses synthetic workloads");
            for (unsigned seed : {17u, 42u, 217u}) run(make_mixed(seed), seed, 5);
            return 0;
        }
        if (flag == "--million") {
            if (argc == 4) throw std::runtime_error("--million uses synthetic scale workloads");
            for (std::string scenario : {"sparse", "teleport"})
                run(make_work(scenario, 1000000, 192, 17), 17, 2);
            return 0;
        }
        bool full = flag == "--full";
        std::vector<Id> sizes = full ? std::vector<Id>{1024, 8192, 32768} : std::vector<Id>{1024, 8192};
        std::vector<unsigned> seeds = full ? std::vector<unsigned>{17, 42, 217} : std::vector<unsigned>{17};
        for (Id size : sizes)
            for (unsigned seed : seeds)
                for (std::string scenario : {"sparse", "atomic", "jump", "churn", "unary"})
                    run(make_work(scenario, size, full ? 1200 : 384, seed),
                        seed, full ? 5 : 2);
        for (unsigned seed : seeds)
            run(make_work("teleport", 100000, full ? 512 : 80, seed), seed, full ? 5 : 2);
        for (Id size : sizes)
            for (unsigned seed : seeds)
                run(make_quantized(size, full ? 1200 : 384, seed), seed, full ? 5 : 2);
        if (argc == 4)
            for (unsigned seed : seeds)
                run(make_azure(argv[3], 8192, full ? 1200 : 384, seed), seed, full ? 5 : 2);
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
#endif
