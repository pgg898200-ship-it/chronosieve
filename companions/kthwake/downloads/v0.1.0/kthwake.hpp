#ifndef KTHWAKE_HPP
#define KTHWAKE_HPP

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <exception>
#include <limits>
#include <optional>
#include <stdexcept>
#include <vector>
#include <unordered_set>

namespace kthwake {

struct Guard {
    std::uint32_t k;
    std::uint64_t window;
};

// A single ordered event stream. Guard IDs are zero-based input indices.
// Calls and callbacks run synchronously on one thread. A callback must not
// reenter this index. Equal timestamps across calls are allowed.
class Index {
public:
    explicit Index(const std::vector<Guard>& guards) {
        if (!guards.empty() &&
            guards.size() - 1 > std::numeric_limits<std::uint32_t>::max()) {
            throw std::length_error("guard ID exceeds uint32_t");
        }
        std::vector<Entry> entries;
        entries.reserve(guards.size());
        for (std::size_t i = 0; i < guards.size(); ++i) {
            const auto& g = guards[i];
            if (!g.k || !g.window) {
                throw std::invalid_argument("k and window must both be positive");
            }
            entries.push_back({g.window, static_cast<std::uint32_t>(i), g.k});
        }
        std::sort(entries.begin(), entries.end(), [](const Entry& a, const Entry& b) {
            if (a.k != b.k) return a.k < b.k;
            if (a.window != b.window) return a.window < b.window;
            return a.id < b.id;
        });
        for (std::size_t begin = 0; begin < entries.size();) {
            const auto k = entries[begin].k;
            auto end = begin + 1;
            while (end < entries.size() && entries[end].k == k) ++end;
            groups_.push_back({begin, end, end, k, 0, 0});
            begin = end;
        }
        if (groups_.empty()) return;
        max_k_ = groups_.back().k;
        // Sorting by rank is sufficient when only a few ranks are present.
        // Look for a significantly smaller window partition before building
        // the alternative layout. Stop counting widths as soon as it loses.
        if (groups_.size() >= 4) {
            std::unordered_set<std::uint64_t> widths;
            const auto limit = groups_.size() / 4;
            for (const auto& e : entries) {
                widths.insert(e.window);
                if (widths.size() > limit) break;
            }
            if (widths.size() <= limit) {
                std::sort(entries.begin(), entries.end(), [](const Entry& a, const Entry& b) {
                    if (a.window != b.window) return a.window < b.window;
                    if (a.k != b.k) return a.k < b.k;
                    return a.id < b.id;
                });
                for (std::size_t begin = 0; begin < entries.size();) {
                    const auto w = entries[begin].window;
                    auto end = begin + 1;
                    while (end < entries.size() && entries[end].window == w) ++end;
                    windows_.push_back({begin, end, begin, w, 0});
                    begin = end;
                }
                std::vector<Group>().swap(groups_);
                flatten(entries);
                return;
            }
        }
        if (groups_.size() >= 32) timers_.resize(groups_.size());
        if (max_k_ <= 1024) {
            std::size_t capacity = 1;
            while (capacity < max_k_) capacity *= 2;
            ring_.resize(capacity);
        }
        flatten(entries);
    }

    // Evaluate the post-batch snapshot. Each occurrence is stamped 'now'.
    // An occurrence at s counts iff now-s < window. Emit (ID, new_state)
    // exactly once per changed guard. The initial snapshot is all false.
    // If a callback throws, the new snapshot is committed, subsequent
    // callbacks are skipped, and the original exception is rethrown.
    template<class Emit>
    void step(std::uint64_t now, std::uint64_t arrivals, Emit&& emit) {
        if (now < clock_) throw std::invalid_argument("time moved backward");
        if (!arrivals && (now == clock_ || (!dirty_ && (!expiry_ || now < *expiry_)))) {
            clock_ = now;
            return;
        }
        if (arrivals && max_k_) {
            if (ring_.empty()) append(now, arrivals);
            else append_small(now, arrivals);
        }
        clock_ = now;
        expiry_.reset();
        dirty_ = !deadlines_requested_ && arrivals != 0;
        std::exception_ptr failure;
        auto notify = [&](std::size_t first, std::size_t last, bool active) {
            if (failure) return;
            try {
                for (auto i = first; i < last; ++i) emit(ids_[i], active);
            } catch (...) {
                failure = std::current_exception();
            }
        };

        const bool scheduled = deadlines_requested_ && !timers_.empty();
        if (scheduled && !arrivals) {
            expire_ranks(now, notify);
        } else if (!windows_.empty()) {
            step_windows(now, notify);
        } else if (!arrivals) {
            step_ranks<true>(now, [](Group& g) { return g.stamp; }, notify);
        } else if (!ring_.empty()) {
            const auto mask = ring_.size() - 1;
            if (ring_explicit_ == retained_) {
                auto stamp = [&](Group& g) { return ring_[(ring_head_ - g.k) & mask]; };
                if (dirty_ || scheduled) step_ranks<false>(now, stamp, notify);
                else step_ranks<true>(now, stamp, notify);
            } else {
                auto stamp = [&](Group& g) {
                    return g.k <= ring_explicit_ ? ring_[(ring_head_ - g.k) & mask] : bulk_stamp_;
                };
                if (dirty_ || scheduled) step_ranks<false>(now, stamp, notify);
                else step_ranks<true>(now, stamp, notify);
            }
        } else {
            auto stamp = [&](Group& g) { return kth_stamp(g); };
            if (dirty_ || scheduled) step_ranks<false>(now, stamp, notify);
            else step_ranks<true>(now, stamp, notify);
        }
        if (scheduled && arrivals) rebuild_timers();
        if (failure) std::rethrow_exception(failure);
    }

    // Assuming no more arrivals, first representable future tick at which a
    // guard becomes false. A deadline equal to UINT64_MAX is representable.
    [[nodiscard]] std::optional<std::uint64_t> next_expiry() const noexcept {
        deadlines_requested_ = true;
        if (!timers_.empty() && !timers_ready_) {
            rebuild_timers();
            dirty_ = false;
        } else if (dirty_) {
            if (windows_.empty()) {
                for (const auto& g : groups_) {
                    if (g.k > retained_) break;
                    if (g.split < g.end) consider(g.stamp, keys_[g.split]);
                }
            } else {
                for (const auto& w : windows_) consider_window(w);
            }
            dirty_ = false;
        }
        return expiry_;
    }

private:
    struct Entry {
        std::uint64_t window;
        std::uint32_t id;
        std::uint32_t k;
    };
    struct Group {
        std::size_t begin, end, split;
        std::uint32_t k;
        std::uint64_t run_serial, stamp;
    };
    struct WindowGroup {
        std::size_t begin, end, split;
        std::uint64_t window, left_serial;
    };
    struct Run {
        std::uint64_t stamp;
        std::uint64_t end;  // cumulative occurrence count at end of run, modulo 2^64
    };

    void flatten(const std::vector<Entry>& entries) {
        keys_.reserve(entries.size());
        ids_.reserve(entries.size());
        for (const auto& e : entries) {
            keys_.push_back(windows_.empty() ? e.window : e.k);
            ids_.push_back(e.id);
        }
    }

    struct Timer {
        std::uint64_t due;
        std::size_t group;
    };
    // One deadline per rank. The heap's root can only move forward on
    // idle ticks, so replacing it needs a single downward sift. Rebuild
    // bottom-up on arrivals, which can change every supporting timestamp.
    void sift_timer(std::size_t i, Timer item) const noexcept {
        auto child = i * 2 + 1;
        while (child < timer_count_) {
            if (child + 1 < timer_count_ && timers_[child + 1].due < timers_[child].due) ++child;
            if (item.due <= timers_[child].due) break;
            timers_[i] = timers_[child];
            i = child;
            child = i * 2 + 1;
        }
        timers_[i] = item;
    }
    bool rank_deadline(const Group& g, std::uint64_t& due) const noexcept {
        if (g.split == g.end) return false;
        const auto w = keys_[g.split];
        if (w > std::numeric_limits<std::uint64_t>::max() - g.stamp) return false;
        due = g.stamp + w;
        return true;
    }
    void refresh_minimum() const noexcept {
        if (timer_count_) expiry_ = timers_[0].due;
        else expiry_.reset();
    }
    void rebuild_timers() const noexcept {
        timer_count_ = 0;
        for (std::size_t i = 0; i < groups_.size(); ++i) {
            const auto& g = groups_[i];
            if (g.k > retained_) break;
            std::uint64_t due;
            if (rank_deadline(g, due)) timers_[timer_count_++] = {due, i};
        }
        for (auto i = timer_count_ / 2; i > 0; --i) sift_timer(i - 1, timers_[i - 1]);
        timers_ready_ = true;
        refresh_minimum();
    }
    template<class Emit>
    void expire_ranks(std::uint64_t now, Emit& emit) {
        while (timer_count_ && timers_[0].due <= now) {
            auto timer = timers_[0];
            auto& g = groups_[timer.group];
            adjust(g, now - g.stamp, false, emit);
            if (rank_deadline(g, timer.due)) sift_timer(0, timer);
            else if (--timer_count_) sift_timer(0, timers_[timer_count_]);
        }
        refresh_minimum();
    }

    void consider(std::uint64_t stamp, std::uint64_t window) const noexcept {
        if (window <= std::numeric_limits<std::uint64_t>::max() - stamp) {
            const auto due = stamp + window;
            if (!expiry_ || due < *expiry_) expiry_ = due;
        }
    }

    // The active set is a prefix for window groups and a suffix for rank
    // groups. Sparse crossings walk adjacent keys. A distant crossing uses
    // binary search, followed by one contiguous callback pass.
    template<class Partition, class Emit>
    void adjust(Partition& g, std::uint64_t threshold, bool prefix, Emit& emit) {
        auto cut = g.split;
        if (cut < g.end && keys_[cut] <= threshold) {
            if (g.end - cut > 8 && keys_[cut + 8] <= threshold) {
                auto end = std::upper_bound(keys_.begin() + cut + 9, keys_.begin() + g.end,
                    threshold);
                cut = static_cast<std::size_t>(end - keys_.begin());
            } else {
                do { ++cut; } while (cut < g.end && keys_[cut] <= threshold);
            }
            emit(g.split, cut, prefix);
        } else if (cut > g.begin && keys_[cut - 1] > threshold) {
            if (cut - g.begin > 8 && keys_[cut - 9] > threshold) {
                auto end = std::upper_bound(keys_.begin() + g.begin, keys_.begin() + cut - 9,
                    threshold);
                cut = static_cast<std::size_t>(end - keys_.begin());
            } else {
                do { --cut; } while (cut > g.begin && keys_[cut - 1] > threshold);
            }
            emit(cut, g.split, !prefix);
        }
        g.split = cut;
    }

    template<bool Deadlines, class Stamp, class Emit>
    void step_ranks(std::uint64_t now, Stamp stamp, Emit& emit) {
        for (auto& g : groups_) {
            if (g.k > retained_) break;
            g.stamp = stamp(g);
            const auto age = now - g.stamp;
            adjust(g, age, false, emit);
            if constexpr (Deadlines) {
                if (g.split < g.end) consider(g.stamp, keys_[g.split]);
            }
        }
    }

    template<class Emit>
    void step_windows(std::uint64_t now, Emit& emit) {
        for (auto& w : windows_) {
            // A window's first live run only advances. Check its cursor and
            // neighbor first, then binary-search across large time jumps.
            auto offset = w.left_serial - first_serial_;
            std::size_t left = offset <= history_.size() ? static_cast<std::size_t>(offset) : 0;
            if (left < history_.size() && now - history_[left].stamp >= w.window) {
                ++left;
                if (left < history_.size() && now - history_[left].stamp >= w.window) {
                    auto hi = history_.size();
                    ++left;
                    while (left < hi) {
                        auto mid = left + (hi - left) / 2;
                        if (now - history_[mid].stamp >= w.window) left = mid + 1;
                        else hi = mid;
                    }
                }
            }
            w.left_serial = first_serial_ + left;
            auto count = left == 0 ? retained_ : tail_ - history_[left - 1].end;
            adjust(w, count, true, emit);
            if (!dirty_) consider_window(w);
        }
    }

    void consider_window(const WindowGroup& w) const noexcept {
        if (w.split == w.begin) return;
        // The largest active rank has the oldest supporting occurrence.
        // Its expiry is exactly the first falling edge for this window.
        const auto k = static_cast<std::uint32_t>(keys_[w.split - 1]);
        const auto left = static_cast<std::size_t>(w.left_serial - first_serial_);
        consider(history_[rank_position(k, left)].stamp, w.window);
    }

    void append_small(std::uint64_t now, std::uint64_t n) {
        // A batch covering every requested rank is a constant-time reset.
        if (n >= max_k_) {
            retained_ = max_k_;
            ring_explicit_ = 0;
            ring_head_ = 0;
            bulk_stamp_ = now;
            return;
        }
        // Expand at most 64 timestamps per call. An interior batch larger
        // than that switches once to compressed history, with at most 1024
        // previous timestamps to import, independent of input multiplicity.
        if (n > 64) {
            auto mask = ring_.size() - 1;
            std::deque<Run> imported;
            std::uint64_t end = 0;
            auto add = [&](std::uint64_t stamp, std::uint64_t count) {
                end += count;
                if (!imported.empty() && imported.back().stamp == stamp) imported.back().end = end;
                else imported.push_back({stamp, end});
            };
            if (retained_ > ring_explicit_) add(bulk_stamp_, retained_ - ring_explicit_);
            for (auto k = ring_explicit_; k > 0; --k) {
                add(ring_[(ring_head_ - k) & mask], 1);
            }
            history_.swap(imported);
            tail_ = end;
            std::vector<std::uint64_t>().swap(ring_);
            append(now, n);
            return;
        }
        auto count = static_cast<std::size_t>(n);
        auto first = std::min(count, ring_.size() - ring_head_);
        std::fill_n(ring_.begin() + ring_head_, first, now);
        std::fill_n(ring_.begin(), count - first, now);
        ring_head_ = (ring_head_ + count) & (ring_.size() - 1);
        ring_explicit_ = std::min<std::uint64_t>(max_k_, ring_explicit_ + n);
        retained_ = std::min<std::uint64_t>(max_k_, retained_ + n);
    }

    void append(std::uint64_t now, std::uint64_t arrivals) {
        // Only the most recent max_k occurrences can affect any guard.
        // A whole batch adds at most one run, even for UINT64_MAX arrivals.
        const auto n = std::min<std::uint64_t>(arrivals, max_k_);
        const auto next_end = tail_ + n;
        if (history_.empty() || history_.back().stamp != now) {
            history_.push_back({now, next_end});
        } else {
            history_.back().end = next_end;
        }
        tail_ = next_end;
        retained_ = std::min<std::uint64_t>(max_k_, retained_ + n);
        // Oldest retained endpoint can lag by at most 2*max_k before pruning.
        // Modular subtraction therefore gives its exact bounded age on wrap.
        while (tail_ - history_.front().end >= max_k_) {
            history_.pop_front();
            ++first_serial_;
        }
    }

    std::uint64_t kth_stamp(Group& group) const {
        const auto size = history_.size();
        std::size_t pos;
        if (size == retained_) {
            // All retained occurrences occupy different timestamp runs.
            pos = size - group.k;
        } else {
            // A rank's run can only move forward. Its old position (or the
            // oldest retained run after eviction) usually needs zero or one
            // adjustment. For a larger jump find the first run with fewer
            // than k newer occurrences by binary search.
            const auto offset = group.run_serial - first_serial_;
            pos = offset < size ? static_cast<std::size_t>(offset) : 0;
            pos = rank_position(group.k, pos);
        }
        group.run_serial = first_serial_ + pos;
        return history_[pos].stamp;
    }

    std::size_t rank_position(std::uint32_t k, std::size_t pos) const noexcept {
        if (tail_ - history_[pos].end < k) return pos;
        ++pos;
        if (tail_ - history_[pos].end < k) return pos;
        auto hi = history_.size() - 1;
        ++pos;
        while (pos < hi) {
            const auto mid = pos + (hi - pos) / 2;
            if (tail_ - history_[mid].end >= k) pos = mid + 1;
            else hi = mid;
        }
        return pos;
    }

    std::vector<WindowGroup> windows_;
    std::vector<std::uint64_t> ring_;
    std::size_t ring_head_ = 0;
    std::uint64_t ring_explicit_ = 0;
    std::uint64_t bulk_stamp_ = 0;
    mutable std::vector<Timer> timers_;
    mutable std::size_t timer_count_ = 0;
    mutable bool timers_ready_ = false;
    mutable bool dirty_ = false;
    mutable bool deadlines_requested_ = false;
    std::vector<std::uint64_t> keys_;
    std::vector<std::uint32_t> ids_;
    std::vector<Group> groups_;
    std::deque<Run> history_;
    std::uint32_t max_k_ = 0;
    std::uint64_t tail_ = 0;
    std::uint64_t retained_ = 0;
    std::uint64_t first_serial_ = 0;
    std::uint64_t clock_ = 0;
    mutable std::optional<std::uint64_t> expiry_;
};

}  // namespace kthwake

#endif  // KTHWAKE_HPP
