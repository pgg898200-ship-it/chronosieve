// Reuse the published benchmark's independent native implementations.
#define GUARDSHIFT_BENCH_LIBRARY
#include "bench.cpp"

#ifdef GUARDSHIFT_WITH_SUPERINTERVALS
#include "superintervals_adapters.hpp"
#endif

#include <charconv>
#include <unordered_set>

namespace {
constexpr size_t MiB = 1024 * 1024;
constexpr size_t max_fields = 4096;
constexpr size_t max_rows = 1000000;
constexpr size_t max_items = 4000000;

[[noreturn]] void invalid(size_t line, const char* reason) {
    throw std::runtime_error("line " + std::to_string(line) + ": " + reason);
}

Value number(const std::string& text, size_t line) {
    Value value{};
    const char* end = text.data() + text.size();
    auto result = std::from_chars(text.data(), end, value);
    if (result.ec != std::errc{} || result.ptr != end) invalid(line, "expected a signed 64-bit decimal integer");
    return value;
}

Workload read_trace(std::istream& input) {
    Workload w{"custom", {}, {}, {}};
    std::vector<size_t> owner, slot;
    size_t bytes = 0, items = 0, lineno = 0;
    std::string line;
    while (std::getline(input, line)) {
        ++lineno;
        bytes += line.size() + 1;
        if (line.size() > MiB || bytes > 64 * MiB) invalid(lineno, "trace size limit exceeded");
        const auto comment = line.find('#');
        if (comment != std::string::npos) line.resize(comment);
        std::istringstream fields(line);
        std::string command, token;
        if (!(fields >> command)) continue;
        if (command == "init") {
            if (!w.initial.empty()) invalid(lineno, "init may appear only once");
            while (fields >> token) {
                if (w.initial.size() == max_fields) invalid(lineno, "too many signals");
                w.initial.push_back(number(token, lineno));
            }
            if (w.initial.empty()) invalid(lineno, "init needs at least one value");
            owner.assign(w.initial.size(), max_rows);
            slot.assign(w.initial.size(), max_items);
            continue;
        }
        if (w.initial.empty()) invalid(lineno, "expected init before rules or patches");
        const bool guard = command == "guard";
        if (guard) {
            if (!w.patches.empty()) invalid(lineno, "guards must precede all patches");
            if (w.guards.size() == max_rows) invalid(lineno, "too many guards");
            w.guards.emplace_back();
        } else if (command == "patch") {
            if (w.guards.empty()) invalid(lineno, "define at least one guard before the first patch");
            if (w.patches.size() == max_rows) invalid(lineno, "too many patches");
            w.patches.emplace_back();
        } else {
            invalid(lineno, "expected init, guard or patch");
        }
        while (fields >> token) {
            if (++items > max_items) invalid(lineno, "too many terms and writes");
            const Value id = number(token, lineno);
            if (id < 0 || static_cast<size_t>(id) >= w.initial.size()) invalid(lineno, "unknown signal");
            const auto signal = static_cast<Id>(id);
            if (!(fields >> token)) invalid(lineno, "missing value");
            Value value = number(token, lineno);
            if (guard) {
                if (!(fields >> token)) invalid(lineno, "missing upper bound");
                const Value high = number(token, lineno);
                if (high < value) invalid(lineno, "reversed interval");
                const auto guard_id = w.guards.size() - 1;
                if (owner[signal] == guard_id) invalid(lineno, "repeated signal in guard");
                owner[signal] = guard_id;
                w.guards.back().push_back({signal, value, high});
            } else {
                auto& patch = w.patches.back();
                if (slot[signal] == max_items) {
                    slot[signal] = patch.size();
                    patch.push_back({signal, value});
                } else {
                    patch[slot[signal]].value = value;
                }
            }
        }
        if (!guard) {
            for (auto u : w.patches.back()) slot[u.signal] = max_items;
        }
    }
    if (input.bad()) throw std::runtime_error("error reading trace");
    if (w.patches.empty()) throw std::runtime_error("at least one patch is required");
    return w;
}

bool slab_fits(const Workload& w) {
    if (w.initial.size() > 4) return false;
    const size_t words = (w.guards.size() + 63) / 64;
    const size_t max_cells = 8 * MiB / sizeof(uint64_t) / words;
    size_t cells = w.initial.size();
    std::vector<std::unordered_set<Value>> cuts(cells);
    for (const auto& row : w.guards) {
        for (Term t : row) {
            cells += cuts[t.signal].insert(t.low).second;
            if (t.high != std::numeric_limits<Value>::max())
                cells += cuts[t.signal].insert(t.high + 1).second;
            if (cells > max_cells) return false;
        }
    }
    return cells <= max_cells;
}

bool unary_fits(const Workload& w) {
    if (w.initial.size() != 1) return false;
    // Width1D does signed differences. These bounds keep every difference
    // and point-minus-width operation in its representable range.
    constexpr Value limit = (Value{1} << 61);
    auto bounded = [](Value v) { return -limit <= v && v <= limit; };
    if (!bounded(w.initial[0])) return false;
    for (const auto& g : w.guards)
        if (g.size() != 1 || !bounded(g[0].low) || !bounded(g[0].high)) return false;
    for (const auto& p : w.patches)
        if (p.size() != 1 || !bounded(p[0].value)) return false;
    return true;
}

template<class Engine>
void measure_local(const Workload& w, const Oracle& oracle, const Scan& final_truth,
                   const char* name, unsigned round) {
    const auto start = Clock::now();
    Engine engine(w.initial, w.guards);
    const auto build = ns(Clock::now() - start);
    uint64_t total = 0, changes = 0;
    std::vector<uint64_t> times;
    times.reserve(w.patches.size());
    for (size_t i = 0; i < w.patches.size(); ++i) {
        auto begin = Clock::now();
        auto result = engine.apply(w.patches[i]);
        auto elapsed = ns(Clock::now() - begin);
        if (canonical(result) != oracle[i]) throw std::runtime_error(std::string(name) + " transition mismatch at patch " + std::to_string(i));
        times.push_back(elapsed);
        total += elapsed;
        changes += result.size();
    }
    for (Id i = 0; i < w.guards.size(); ++i)
        if (engine.active(i) != final_truth.active(i)) throw std::runtime_error(std::string(name) + " final state mismatch");
    std::sort(times.begin(), times.end());
    auto pct = [&](size_t n) { return times[(times.size() * n + 99) / 100 - 1]; };
    std::cout << "{\"method\":\"" << name << "\",\"rep\":" << round
              << ",\"signals\":" << w.initial.size() << ",\"guards\":" << w.guards.size()
              << ",\"patches\":" << times.size() << ",\"changes\":" << changes
              << ",\"build_ns\":" << build << ",\"elapsed_ns\":" << total
              << ",\"ns_per_patch\":" << std::setprecision(12) << double(total) / times.size()
              << ",\"p50_ns\":" << pct(50) << ",\"p99_ns\":" << pct(99)
              << ",\"verified\":true}\n";
}

void compare_methods(const Workload& w) {
    Scan truth(w.initial, w.guards);
    Oracle oracle;
    oracle.reserve(w.patches.size());
    size_t saved_changes = 0;
    for (const auto& p : w.patches) {
        auto result = truth.apply(p);
        saved_changes += result.size();
        if (saved_changes > max_items) throw std::runtime_error("oracle exceeds 4 million transitions; use a shorter trace");
        oracle.push_back(canonical(result));
    }
    using Method = void (*)(const Workload&, const Oracle&, const Scan&, const char*, unsigned);
    std::vector<std::pair<const char*, Method>> methods{
        {"guardshift", measure_local<guardshift::Index>}, {"incidence_count", measure_local<Incidence>},
        {"endpoint_count", measure_local<EndpointCounts>}, {"stabbing_count", measure_local<StabCounts>},
        {"full_scan", measure_local<Scan>}
    };
    if (slab_fits(w)) methods.push_back({"slab_bits", measure_local<SlabBits>});
    if (unary_fits(w)) {
        methods.push_back({"direct_1d", measure_local<Endpoint1D>});
        methods.push_back({"bounded_width_1d", measure_local<Width1D>});
    }
#ifdef GUARDSHIFT_WITH_SUPERINTERVALS
    using Map = si::IntervalMap<Value, Id>;
    using Eytz = si::IntervalMapEytz<Value, Id>;
    methods.push_back({"superintervals", measure_local<SuperIntervalsAdapter<Map>>});
    methods.push_back({"superintervals_eytz", measure_local<SuperIntervalsAdapter<Eytz>>});
    methods.push_back({"superintervals_edges", measure_local<SuperIntervalsEdges<Map>>});
    methods.push_back({"superintervals_edges_eytz", measure_local<SuperIntervalsEdges<Eytz>>});
#endif
    for (unsigned round = 0; round < 3; ++round) {
        for (const auto& entry : methods) entry.second(w, oracle, truth, entry.first, round);
        std::rotate(methods.begin(), methods.begin() + 1, methods.end());
    }
}
} // namespace

int main(int argc, char** argv) {
    if (argc > 2 || (argc == 2 && std::string(argv[1]) == "--help")) {
        std::cerr << "usage: guardshift-replay [FILE|-]\n";
        return argc > 2 ? 2 : 0;
    }
    try {
        if (argc == 1 || std::string(argv[1]) == "-") {
            compare_methods(read_trace(std::cin));
        } else {
            std::ifstream file(argv[1]);
            if (!file) throw std::runtime_error("cannot open input file");
            compare_methods(read_trace(file));
        }
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
