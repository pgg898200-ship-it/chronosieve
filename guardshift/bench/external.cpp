// Compare the actual SuperIntervals C++ library under the same workload,
// oracle, compiler and measurement harness as Guardshift's native controls.
#define GUARDSHIFT_BENCH_LIBRARY
#include "bench.cpp"
#undef GUARDSHIFT_BENCH_LIBRARY
#include "superintervals.hpp"

namespace {

template <class Map>
class SuperIntervalsAdapter {
    struct Field {
        Map intervals;
        std::vector<std::size_t> previous;
        std::vector<std::size_t> next;
        Value value{};
    };
    std::vector<Field> fields;
    Counts counts;

public:
    SuperIntervalsAdapter(const std::vector<Value>& initial, const Guards& guards)
        : fields(initial.size()), counts(initial, guards) {
        for (Id id = 0; id < guards.size(); ++id) {
            for (const auto& term : guards[id]) {
                fields[term.signal].intervals.add(term.low, term.high, id);
            }
        }
        for (std::size_t s = 0; s < fields.size(); ++s) {
            Field& field = fields[s];
            field.value = initial[s];
            field.intervals.build();
            if (!field.intervals.starts.empty() && field.value >= field.intervals.starts.front())
                field.intervals.search_idxs(field.value, field.value, field.previous);
        }
    }

    bool active(Id guard) const { return counts.active(guard); }

    std::vector<Change> apply(const std::vector<Update>& patch) {
        for (const Update& u : patch) {
            Field& field = fields[u.signal];
            if (u.value == field.value) continue;
            // SuperIntervals returns indices in descending order. Cache the last
            // answer and merge both lists, avoiding a second point query or sort.
            field.next.clear();
            if (!field.intervals.starts.empty() && u.value >= field.intervals.starts.front())
                field.intervals.search_idxs(u.value, u.value, field.next);
            std::size_t a = 0, b = 0;
            const auto& ids = field.intervals.data;
            while (a < field.previous.size() && b < field.next.size()) {
                const auto old_hit = field.previous[a];
                const auto new_hit = field.next[b];
                if (old_hit == new_hit) {
                    ++a;
                    ++b;
                } else if (old_hit > new_hit) {
                    counts.adjust(ids[old_hit], 1);
                    ++a;
                } else {
                    counts.adjust(ids[new_hit], -1);
                    ++b;
                }
            }
            while (a < field.previous.size()) counts.adjust(ids[field.previous[a++]], 1);
            while (b < field.next.size()) counts.adjust(ids[field.next[b++]], -1);
            field.previous.swap(field.next);
            field.value = u.value;
        }
        return counts.finish();
    }
};

// An alternative adapter indexes entry and exit events as singleton points.
// A single range search then returns all crossed endpoints on either leg.
template <class Map>
class SuperIntervalsEdges {
    std::vector<Map> fields;
    std::vector<Value> values;
    std::vector<Id> hits;
    Counts counts;

public:
    SuperIntervalsEdges(const std::vector<Value>& initial, const Guards& guards)
        : fields(initial.size()), values(initial), counts(initial, guards) {
        if (guards.size() > std::numeric_limits<Id>::max() / 2) {
            throw std::length_error("benchmark crossing ID out of range");
        }
        struct Entry { Value at; Id event; };
        std::vector<std::vector<Entry>> entries(fields.size());
        for (Id id = 0; id < guards.size(); ++id) {
            for (const auto& term : guards[id]) {
                auto& field = entries[term.signal];
                if (term.low != std::numeric_limits<Value>::min()) {
                    field.push_back({term.low, (id << 1) | 1u});
                }
                if (term.high != std::numeric_limits<Value>::max()) {
                    const auto end = term.high + 1;
                    field.push_back({end, id << 1});
                }
            }
        }
        for (std::size_t i = 0; i < fields.size(); ++i) {
            auto& input = entries[i];
            std::sort(input.begin(), input.end(),
                      [](const Entry& a, const Entry& b) { return a.at < b.at; });
            fields[i].reserve(input.size());
            for (const auto& entry : input) fields[i].add(entry.at, entry.at, entry.event);
            std::vector<Entry>().swap(input);
            fields[i].build();
        }
    }

    bool active(Id guard) const { return counts.active(guard); }

    std::vector<Change> apply(const std::vector<Update>& patch) {
        for (const Update& update : patch) {
            auto& previous = values[update.signal];
            if (previous == update.value) continue;
            const bool up = update.value > previous;
            const auto low = std::min(previous, update.value);
            const auto high = std::max(previous, update.value);
            hits.clear();
            const auto& map = fields[update.signal];
            if (!map.starts.empty() && high >= map.starts.front())
                map.search_values(low + 1, high, hits);
            for (const auto& hit : hits) {
                counts.adjust(hit >> 1, bool(hit & 1u) == up ? -1 : 1);
            }
            previous = update.value;
        }
        return counts.finish();
    }
};

void compare(const Workload& w, unsigned seed, unsigned repeats) {
    const auto oracle = make_oracle(w);
    for (unsigned rep = 0; rep < repeats; ++rep) {
        for (unsigned k = 0; k < 5; ++k) {
            switch ((k + rep) % 5) {
                case 0:
                    measure<guardshift::Index>(w, oracle, "guardshift", seed, rep);
                    break;
                case 1:
                    measure<SuperIntervalsAdapter<si::IntervalMap<Value, Id>>>(
                        w, oracle, "superintervals", seed, rep);
                    break;
                case 2:
                    measure<SuperIntervalsAdapter<si::IntervalMapEytz<Value, Id>>>(
                        w, oracle, "superintervals_eytz", seed, rep);
                    break;
                case 3:
                    measure<SuperIntervalsEdges<si::IntervalMap<Value, Id>>>(
                        w, oracle, "superintervals_edges", seed, rep);
                    break;
                case 4:
                    measure<SuperIntervalsEdges<si::IntervalMapEytz<Value, Id>>>(
                        w, oracle, "superintervals_edges_eytz", seed, rep);
                    break;
            }
        }
    }
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 2) throw std::invalid_argument("usage: external replay/azure-code.csv");
        for (unsigned seed : {17u, 42u, 217u}) {
            compare(make_work("atomic", 8192, 1200, seed), seed, 3);
            compare(make_work("atomic", 32768, 1200, seed), seed, 3);
            compare(make_azure(argv[1], 8192, 1200, seed), seed, 3);
            compare(make_mixed(seed), seed, 3);
            compare(make_work("teleport", 100000, 512, seed), seed, 3);
            compare(make_work("jump", 32768, 512, seed), seed, 3);
            compare(make_work("sparse", 1000000, 192, seed), seed, 3);
            compare(make_mixed(seed, 1000000, 192), seed, 3);
        }
    } catch (const std::exception& ex) {
        std::cerr << ex.what() << '\n';
        return 1;
    }
}
