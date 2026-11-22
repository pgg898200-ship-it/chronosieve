#ifndef GUARDSHIFT_HPP
#define GUARDSHIFT_HPP

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace guardshift {

using Value = std::int64_t;
using Id = std::uint32_t;

struct Term {
    Id signal;
    Value low;
    Value high;
};

struct Update {
    Id signal;
    Value value;
};

struct Change {
    Id guard;
    bool active;
};

// A compiled, fixed set of guards over a mutable vector of signed integers.
// Instances are single-writer. Synchronize access when sharing across threads.
class Index {
public:
    Index(const std::vector<Value>& initial,
          const std::vector<std::vector<Term>>& guards)
        : signals_(checked_size(initial.size())),
          guards_(checked_size(guards.size())),
          changed_signals_(initial.size()),
          touched_guards_(guards.size()) {
        for (std::size_t i = 0; i < initial.size(); ++i) {
            signals_[i].value = initial[i];
        }

        std::vector<std::size_t> owner(initial.size(), guards.size());
        for (std::size_t i = 0; i < guards.size(); ++i) {
            Guard& guard = guards_[i];
            for (const Term& term : guards[i]) {
                if (term.signal >= signals_.size()) {
                    throw std::out_of_range("guard contains an unknown signal");
                }
                if (term.low > term.high || owner[term.signal] == i) {
                    throw std::invalid_argument("reversed interval or repeated signal in guard");
                }
                owner[term.signal] = i;
                Signal& signal = signals_[term.signal];
                if (signal.value < term.low || signal.value > term.high) {
                    ++guard.missing;
                }
                const auto id = static_cast<Id>(i);
                if (term.low != std::numeric_limits<Value>::min()) {
                    signal.edges.push_back({term.low, term.high, id, true});
                }
                if (term.high != std::numeric_limits<Value>::max()) {
                    signal.edges.push_back({term.high + 1, term.low, id, false});
                }
            }
            guard.active = guard.missing == 0;
        }

        for (Signal& signal : signals_) {
            std::sort(signal.edges.begin(), signal.edges.end(),
                      [](const Edge& a, const Edge& b) { return a.at < b.at; });
        }
        if (build_bitmaps()) return;
        for (Signal& signal : signals_) {
            build_blocks(signal);
            const auto first_above = std::upper_bound(
                signal.edges.begin(), signal.edges.end(), signal.value,
                [](Value value, const Edge& edge) { return value < edge.at; });
            signal.cursor = static_cast<std::size_t>(first_above - signal.edges.begin());
        }
    }

    bool active(Id guard) const {
        if (bitmap_fields_.empty()) return guards_.at(guard).active;
        if (guard >= bitmap_guard_count_) throw std::out_of_range("unknown guard");
        return (truth_[guard / word_bits] >> (guard % word_bits)) & 1;
    }

    // Last write to each signal wins. Emit at most one change per guard,
    // ordered by ID, comparing the states before and after the entire patch.
    std::vector<Change> apply(const std::vector<Update>& patch) {
        for (const Update& update : patch) {
            if (update.signal >= signals_.size()) {
                throw std::out_of_range("patch contains an unknown signal");
            }
        }
        if (!bitmap_fields_.empty()) return apply_bitmaps(patch);
        std::size_t changed_count = 0;
        for (const Update& update : patch) {
            Signal& signal = signals_[update.signal];
            if (!signal.staged) {
                changed_signals_[changed_count++] = update.signal;
                signal.staged = true;
            }
            signal.pending = update.value;
        }

        for (std::size_t i = 0; i < changed_count; ++i) {
            Signal& signal = signals_[changed_signals_[i]];
            move_cursor(signal, signal.value, signal.pending);
        }

        std::vector<Change> result;
        try {
            for (std::size_t i = 0; i < touched_count_; ++i) {
                const Id id = touched_guards_[i];
                const Guard& guard = guards_[id];
                const bool after = guard.missing == 0;
                if (after != guard.active) {
                    result.push_back({id, after});
                }
            }
            std::sort(result.begin(), result.end(),
                      [](const Change& a, const Change& b) { return a.guard < b.guard; });
        } catch (...) {
            // The original values are retained until the result is materialized.
            // Scratch space for the reverse walk was allocated at construction.
            for (std::size_t i = changed_count; i > 0; --i) {
                Signal& signal = signals_[changed_signals_[i - 1]];
                move_cursor(signal, signal.pending, signal.value);
                signal.staged = false;
            }
            clear_touched();
            throw;
        }

        for (const Change& change : result) {
            guards_[change.guard].active = change.active;
        }
        for (std::size_t i = 0; i < changed_count; ++i) {
            Signal& signal = signals_[changed_signals_[i]];
            signal.value = signal.pending;
            signal.staged = false;
        }
        clear_touched();
        return result;
    }

private:
    struct Edge {
        Value at;
        Value opposite;  // Inclusive opposite endpoint of the same interval.
        Id guard;
        bool enters_upward;
    };
    struct Block {
        Value max_high = std::numeric_limits<Value>::min();
        Value min_low = std::numeric_limits<Value>::max();
    };
    static constexpr std::size_t block_size = 32;
    static constexpr std::size_t jump_threshold = 128;
    struct Signal {
        std::vector<Edge> edges;
        std::vector<Block> tree;
        std::size_t leaves = 0;
        std::size_t cursor = 0;  // First threshold strictly above current value.
        Value value = 0;
        Value pending = 0;
        bool staged = false;
    };
    struct Guard {
        std::size_t missing = 0;
        bool active = false;
        bool touched = false;
    };

    // A slab is the run of signal values between two consecutive thresholds.
    // For small dictionaries of shared thresholds, store one compatible-guard
    // bitset per slab. Each conjunction becomes an intersection of a few rows.
    using Word = std::uint64_t;
    static constexpr std::size_t word_bits = std::numeric_limits<Word>::digits;
    static constexpr std::size_t max_bitmap_fields = 4;
    static constexpr std::size_t bitmap_bytes = 8 * 1024 * 1024;
    struct BitmapField {
        std::vector<Value> cuts;
        std::vector<Word> rows;
        std::size_t slab = 0;
    };
    std::vector<BitmapField> bitmap_fields_;
    std::size_t bitmap_guard_count_ = 0;
    std::vector<Word> truth_, pending_truth_;

    bool build_bitmaps() {
        if (signals_.empty() || signals_.size() > max_bitmap_fields ||
            guards_.size() < 256) return false;
        const auto words = 1 + (guards_.size() - 1) / word_bits;
        const auto max_slabs = bitmap_bytes / sizeof(Word) / words;
        std::size_t slabs = signals_.size();
        std::size_t edges = 0;
        for (const Signal& field : signals_) {
            edges += field.edges.size();
            for (std::size_t i = 0; i < field.edges.size(); ++i) {
                if (i == 0 || field.edges[i-1].at != field.edges[i].at) {
                    if (++slabs > max_slabs) return false;
                }
            }
        }
        // Sparse distinct cutpoints suit the cursor and jump index better.
        if (edges / slabs < 16 || slabs > max_slabs) return false;

        bitmap_guard_count_ = guards_.size();
        bitmap_fields_.resize(signals_.size());
        truth_.resize(words);
        pending_truth_.resize(words);
        for (std::size_t id = 0; id < guards_.size(); ++id) {
            if (guards_[id].active) truth_[id / word_bits] |= Word{1} << (id % word_bits);
        }
        for (std::size_t i = 0; i < signals_.size(); ++i) {
            Signal& signal = signals_[i];
            BitmapField& field = bitmap_fields_[i];
            for (const Edge& e : signal.edges) {
                if (field.cuts.empty() || field.cuts.back() != e.at) field.cuts.push_back(e.at);
            }
            field.rows.resize((field.cuts.size() + 1) * words);
            // Below the first finite threshold, unmentioned signals pass, as
            // do terms with a lower bound at the minimum representable value.
            std::fill_n(field.rows.begin(), words, ~Word{0});
            if (guards_.size() % word_bits != 0) {
                field.rows[words - 1] = (Word{1} << (guards_.size() % word_bits)) - 1;
            }
            for (const Edge& e : signal.edges) {
                if (e.enters_upward) field.rows[e.guard / word_bits] &= ~(Word{1} << (e.guard % word_bits));
            }
            std::size_t edge = 0;
            for (std::size_t cut = 0; cut < field.cuts.size(); ++cut) {
                Word* row = field.rows.data() + (cut + 1) * words;
                std::copy_n(row - words, words, row);
                while (edge < signal.edges.size() && signal.edges[edge].at == field.cuts[cut]) {
                    const Id id = signal.edges[edge++].guard;
                    row[id / word_bits] ^= Word{1} << (id % word_bits);
                }
            }
            field.slab = std::upper_bound(field.cuts.begin(), field.cuts.end(), signal.value) - field.cuts.begin();
            std::vector<Edge>().swap(signal.edges);
        }
        std::vector<Guard>().swap(guards_);
        std::vector<Id>().swap(touched_guards_);
        return true;
    }

    static unsigned first_set_bit(Word bits) noexcept {
#if defined(__GNUC__) || defined(__clang__)
        return static_cast<unsigned>(__builtin_ctzll(bits));
#else
        unsigned shift = 0;
        while ((bits & 1) == 0) { bits >>= 1; ++shift; }
        return shift;
#endif
    }

    std::vector<Change> apply_bitmaps(const std::vector<Update>& patch) {
        std::array<Value, max_bitmap_fields> values{};
        std::array<std::size_t, max_bitmap_fields> slabs{};
        std::array<const Word*, max_bitmap_fields> rows{};
        for (std::size_t i = 0; i < signals_.size(); ++i) values[i] = signals_[i].value;
        for (const Update& update : patch) values[update.signal] = update.value;
        bool moved = false;
        for (std::size_t i = 0; i < signals_.size(); ++i) {
            const BitmapField& field = bitmap_fields_[i];
            const auto last = field.slab;
            const auto& cuts = field.cuts;
            slabs[i] = last;
            if ((last > 0 && values[i] < cuts[last - 1]) ||
                (last < cuts.size() && values[i] >= cuts[last])) {
                slabs[i] = std::upper_bound(cuts.begin(), cuts.end(), values[i]) - cuts.begin();
                moved = true;
            }
            rows[i] = field.rows.data() + slabs[i] * truth_.size();
        }
        std::vector<Change> changes;
        if (moved) {
            for (std::size_t w = 0; w < truth_.size(); ++w) {
                Word next = rows[0][w];
                for (std::size_t i = 1; i < bitmap_fields_.size(); ++i) next &= rows[i][w];
                pending_truth_[w] = next;
                Word diff = next ^ truth_[w];
                while (diff != 0) {
                    const auto bit = first_set_bit(diff);
                    changes.push_back({static_cast<Id>(w * word_bits + bit), bool((next >> bit) & 1)});
                    diff &= diff - 1;
                }
            }
            // Commit after all result allocations succeed. A failed allocation
            // leaves the old truth, field slabs and signal values intact.
            truth_.swap(pending_truth_);
        }
        for (std::size_t i = 0; i < signals_.size(); ++i) {
            signals_[i].value = values[i];
            bitmap_fields_[i].slab = slabs[i];
        }
        return changes;
    }

    static std::size_t checked_size(std::size_t size) {
        if (size > 0 && size - 1 > std::numeric_limits<Id>::max()) {
            throw std::length_error("guard or signal ID exceeds uint32_t");
        }
        return size;
    }

    void cross(const Edge& edge, bool upward) noexcept {
        Guard& guard = guards_[edge.guard];
        if (edge.enters_upward == upward) {
            --guard.missing;
        } else {
            ++guard.missing;
        }
        if ((guard.missing == 0) != guard.active && !guard.touched) {
            guard.touched = true;
            touched_guards_[touched_count_++] = edge.guard;
        }
    }

    // Summarize blocks of sorted edges. For a jump from a to b (a < b),
    // an interior interval containing neither endpoint contributes no change.
    // A start inside (a,b] qualifies only when its high reaches b. An end+1
    // inside (a,b] qualifies only when its low reaches a.
    static void build_blocks(Signal& signal) {
        if (signal.edges.size() <= jump_threshold) {
            return;
        }
        const auto blocks = 1 + (signal.edges.size() - 1) / block_size;
        signal.leaves = 1;
        while (signal.leaves < blocks) {
            signal.leaves *= 2;
        }
        signal.tree.resize(2 * signal.leaves);
        for (std::size_t i = 0; i < signal.edges.size(); ++i) {
            const Edge& edge = signal.edges[i];
            Block& leaf = signal.tree[signal.leaves + i / block_size];
            if (edge.enters_upward) {
                leaf.max_high = std::max(leaf.max_high, edge.opposite);
            } else {
                leaf.min_low = std::min(leaf.min_low, edge.opposite);
            }
        }
        for (std::size_t i = signal.leaves - 1; i > 0; --i) {
            const Block& left = signal.tree[2 * i];
            const Block& right = signal.tree[2 * i + 1];
            signal.tree[i] = {std::max(left.max_high, right.max_high),
                              std::min(left.min_low, right.min_low)};
        }
    }

    void visit_jump(const Signal& signal, std::size_t node, std::size_t left,
                    std::size_t right, std::size_t begin, std::size_t end,
                    Value low, Value high, bool upward) noexcept {
        const Block& summary = signal.tree[node];
        if (summary.max_high < high && summary.min_low > low) {
            return;
        }
        if (right - left == block_size) {
            for (auto i = std::max(left, begin); i < std::min(right, end); ++i) {
                const Edge& e = signal.edges[i];
                if (e.enters_upward ? e.opposite >= high : e.opposite <= low) {
                    cross(e, upward);
                }
            }
            return;
        }
        const auto mid = left + (right - left) / 2;
        if (begin < mid) {
            visit_jump(signal, 2 * node, left, mid, begin, end, low, high, upward);
        }
        if (end > mid) {
            visit_jump(signal, 2 * node + 1, mid, right, begin, end, low, high, upward);
        }
    }

    void move_cursor(Signal& signal, Value original, Value target) noexcept {
        const auto& edges = signal.edges;
        auto cursor = signal.cursor;
        const bool upward = target > original;
        const bool jump = upward
            ? edges.size() - cursor > jump_threshold && edges[cursor + jump_threshold].at <= target
            : cursor > jump_threshold && edges[cursor - 1 - jump_threshold].at > target;
        if (jump) {
            const auto first_above = std::upper_bound(
                edges.begin(), edges.end(), target,
                [](Value value, const Edge& edge) { return value < edge.at; });
            const auto next = static_cast<std::size_t>(first_above - edges.begin());
            visit_jump(signal, 1, 0, signal.leaves * block_size,
                       std::min(cursor, next), std::max(cursor, next),
                       std::min(original, target), std::max(original, target), upward);
            signal.cursor = next;
        } else if (upward) {
            while (cursor < edges.size() && edges[cursor].at <= target) {
                cross(edges[cursor++], true);
            }
            signal.cursor = cursor;
        } else {
            while (cursor > 0 && edges[cursor - 1].at > target) {
                cross(edges[--cursor], false);
            }
            signal.cursor = cursor;
        }
    }

    void clear_touched() noexcept {
        for (std::size_t i = 0; i < touched_count_; ++i) {
            guards_[touched_guards_[i]].touched = false;
        }
        touched_count_ = 0;
    }

    std::vector<Signal> signals_;
    std::vector<Guard> guards_;
    std::vector<Id> changed_signals_;
    std::vector<Id> touched_guards_;
    std::size_t touched_count_ = 0;
};

}  // namespace guardshift

#endif  // GUARDSHIFT_HPP
