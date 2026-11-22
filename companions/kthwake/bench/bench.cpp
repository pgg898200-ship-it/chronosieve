#include "../include/kthwake.hpp"
#include <algorithm>
#include <charconv>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

using Guard = kthwake::Guard;
using U = std::uint64_t;
using Clock = std::chrono::steady_clock;
struct Tick { U now, arrivals; };
struct Entry { U window; std::uint32_t rank, id; };
static_assert(sizeof(Entry) == 16);
struct Group {
    U value;
    size_t begin, end, cut, left = 0;
};
constexpr U umax = std::numeric_limits<U>::max();

template<bool ByRank> struct SortedIndex {
    static U group_of(const Entry& e) { return ByRank ? e.rank : e.window; }
    static U key_of(const Entry& e) { return ByRank ? e.window : e.rank; }
    std::vector<Entry> entries;
    std::vector<Group> groups;
    explicit SortedIndex(const std::vector<Guard>& guards) {
        entries.reserve(guards.size());
        for (size_t i = 0; i < guards.size(); ++i) {
            const auto [rank, window] = guards[i];
            entries.push_back({window, rank,
                               static_cast<std::uint32_t>(i)});
        }
        std::sort(entries.begin(), entries.end(), [](auto a, auto b) {
            if (group_of(a) != group_of(b)) return group_of(a) < group_of(b);
            if (key_of(a) != key_of(b)) return key_of(a) < key_of(b);
            return a.id < b.id;
        });
        for (size_t first = 0; first < entries.size();) {
            size_t last = first + 1;
            while (last < entries.size() && group_of(entries[last]) == group_of(entries[first])) ++last;
            groups.push_back({group_of(entries[first]), first, last, ByRank ? last : first});
            first = last;
        }
    }

    // Reuse the old partition if neighboring keys still bracket the query.
    size_t partition(const Group& g, U threshold) const {
        if ((g.cut == g.begin || key_of(entries[g.cut - 1]) <= threshold) &&
            (g.cut == g.end || key_of(entries[g.cut]) > threshold)) return g.cut;
        auto it = std::upper_bound(entries.begin() + g.begin, entries.begin() + g.end,
                                   threshold, [](U v, const Entry& e) { return v < key_of(e); });
        return static_cast<size_t>(it - entries.begin());
    }

    template<class Emit>
    void transition(Group& g, size_t cut, bool active_prefix, Emit& emit) {
        if (cut == g.cut) return;
        const bool state = (cut > g.cut) == active_prefix;
        for (size_t i = std::min(cut,g.cut); i < std::max(cut,g.cut); ++i) {
            emit(entries[i].id, state);
        }
        g.cut = cut;
    }
};

// A timestamp ring gives O(1) rank lookup regardless of the stream length.
// If a burst covers the entire ring, write each retained slot at most once.
template<bool CacheDeadline> struct RankIndex : SortedIndex<true> {
    std::vector<U> ring;
    size_t write = 0, filled = 0;
    std::optional<U> due;
    explicit RankIndex(const std::vector<Guard>& gs) : SortedIndex(gs) {
        ring.resize(groups.back().value);
    }
    template<class Emit> void step(U now, U n, Emit emit) {
        if constexpr (CacheDeadline) {
            if (!n && (!due || now < *due)) return;
        }
        if (n >= ring.size()) {
            std::fill(ring.begin(), ring.end(), now);
            write = 0;
            filled = ring.size();
        } else if (n) {
            size_t count = static_cast<size_t>(n);
            size_t first = std::min(count, ring.size()-write);
            std::fill_n(ring.begin()+write, first, now);
            std::fill_n(ring.begin(), count-first, now);
            write += count;
            if (write >= ring.size()) write -= ring.size();
            filled = std::min(ring.size(), filled+count);
        }
        if constexpr (CacheDeadline) due.reset();
        for (auto& g : groups) {
            if (g.value > filled) break;
            size_t i = write >= g.value ? write-g.value : ring.size()-(g.value-write);
            U t = ring[i];
            size_t cut = partition(g, now - t);
            transition(g,cut,false,emit);
            if (CacheDeadline && cut < g.end && key_of(entries[cut]) <= umax-t) {
                U d = t + key_of(entries[cut]);
                if (!due || d < *due) due = d;
            }
        }
    }
};

// A single compressed event log is shared by every distinct window. Each
// cursor crosses each nonempty tick at most once. Cumulative counts use
// unsigned modular subtraction; the generator checks total count beforehand.
template<bool CacheDeadline> struct WindowIndex : SortedIndex<false> {
    struct LogEntry { U now, prefix; };
    std::vector<LogEntry> log;
    U total = 0;
    std::optional<U> due;
    explicit WindowIndex(const std::vector<Guard>& gs) : SortedIndex(gs) {}
    template<class Emit> void step(U now, U n, Emit emit) {
        if constexpr (CacheDeadline) {
            if (!n && (!due || now < *due)) return;
        }
        if (n) {
            total += n;
            log.push_back({now,total});
        }
        if constexpr (CacheDeadline) due.reset();
        for (auto& g : groups) {
            while (g.left < log.size() && now - log[g.left].now >= g.value) ++g.left;
            U count = total - (g.left ? log[g.left-1].prefix : 0);
            size_t cut = partition(g,count);
            transition(g,cut,true,emit);
            if (CacheDeadline && cut > g.begin) {
                // The first live event gives a cheap conservative deadline.
                U t = log[g.left].now;
                if (g.value <= umax - t) {
                    U d = t + g.value;
                    if (!due || d < *due) due = d;
                }
            }
        }
    }
};

struct Digest {
    U count = 0, value = 0;
    void operator()(std::uint32_t id, bool state) {
        U x = U(id) * 2 + state + 0x9e3779b97f4a7c15ULL;
        x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
        x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
        value += x ^ (x >> 31);
        ++count;
    }
};

static U parse(const std::string& s) {
    U v = 0;
    auto [p,e] = std::from_chars(s.data(),s.data() + s.size(),v);
    if (e != std::errc{} || p != s.data()+s.size()) throw std::runtime_error("invalid integer: "+s);
    return v;
}
template<class Accept>
void pairs(const std::string& path, Accept add) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("cannot open "+path);
    std::string line;
    size_t index = 0;
    while (std::getline(f,line)) {
        ++index;
        std::istringstream in(line);
        std::string a,b,c;
        if (!(in>>a)) continue;
        if (a.front()=='#') continue;
        if (!(in>>b) || in>>c) throw std::runtime_error("expected two integers at "+path+":"+std::to_string(index));
        add(parse(a),parse(b));
    }
}

struct Options {
    std::string shape = "many-windows", tick_path, guard_path, name;
    bool include_lazy = true;
    U size = 10000, steps = 1024, reps = 3, seed = 4242;
};
Options options(int argc, char** argv) {
    Options o;
    for (int i=1;i<argc;i+=2) {
        std::string k = argv[i];
        if (i+1>=argc) throw std::runtime_error("missing argument to "+k);
        std::string v = argv[i+1];
        if (k=="--lazy" && (v=="0" || v=="1")) o.include_lazy=(v=="1");
        else if (k=="--shape") o.shape=v;
        else if (k=="--name") o.name=v;
        else if (k=="--size") o.size=parse(v);
        else if (k=="--steps") o.steps=parse(v);
        else if (k=="--reps") o.reps=parse(v);
        else if (k=="--seed") o.seed=parse(v);
        else if (k=="--ticks") o.tick_path=v;
        else if (k=="--guards") o.guard_path=v;
        else throw std::runtime_error("unknown option "+k);
    }
    if (!o.size || o.size>UINT32_MAX || !o.reps || !o.steps) throw std::runtime_error("invalid benchmark dimensions");
    if (o.name.empty()) o.name=o.shape;
    if (o.name.find_first_not_of("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-")!=std::string::npos)
        throw std::runtime_error("invalid workload name");
    return o;
}
std::vector<Guard> guards(const Options& o,std::mt19937_64& rng) {
    std::vector<Guard> out;
    if (!o.guard_path.empty()) {
        pairs(o.guard_path,[&](U k,U w) {
            if (!k || k>UINT32_MAX || !w) throw std::runtime_error("invalid guard");
            out.push_back({static_cast<std::uint32_t>(k),w});
        });
    } else {
        out.reserve(o.size);
        for (U i=0;i<o.size;++i) {
            U k,w;
            if (o.shape=="many-windows" || o.shape=="bursty") {
                k = 1ULL << (i%8);
                w = 1 + i/8;
            } else if (o.shape=="few-windows") {
                k = 1+i/8;
                w = 1ULL << (3+i%8);
            } else if (o.shape=="duplicates") {
                k = 1ULL << (i%4);
                w = 64 + 64*((i/4)%8);
            } else throw std::runtime_error("unsupported shape: "+o.shape);
            out.push_back({static_cast<std::uint32_t>(k),w});
        }
        std::shuffle(out.begin(),out.end(),rng);
    }
    if (out.empty() || out.size()>UINT32_MAX) throw std::runtime_error("invalid guard count");
    return out;
}
std::vector<Tick> ticks(const Options& o,std::mt19937_64& rng) {
    std::vector<Tick> out;
    U total = 0;
    auto add = [&](U now, U n) {
        if (!out.empty() && now <= out.back().now) throw std::runtime_error("ticks must strictly increase");
        if (n>umax-total) throw std::runtime_error("cumulative arrivals exceed uint64");
        total+=n; out.push_back({now,n});
    };
    if (!o.tick_path.empty()) pairs(o.tick_path,add);
    else for (U i=0;i<o.steps;++i) {
        U n;
        if (o.shape=="bursty") n=(i%128==0 ? 128+rng()%256 : 0);
        else n=(rng()%4==0 ? 1+rng()%4 : 0);
        add(i,n);
    }
    if (out.empty()) throw std::runtime_error("empty trace");
    return out;
}
using Edge = std::pair<std::uint32_t,bool>;
template<class Engine>
void collect(Engine& idx,Tick t,std::vector<Edge>& out) {
    out.clear();
    idx.step(t.now,t.arrivals,[&](std::uint32_t id,bool state){out.emplace_back(id,state);});
    std::sort(out.begin(),out.end());
}
void check(const std::vector<Guard>& gs,const std::vector<Tick>& ts, bool include_lazy) {
    kthwake::Index kernel(gs);
    WindowIndex<true> window(gs);
    RankIndex<true> rank(gs);
    std::unique_ptr<WindowIndex<false>> lazy_window;
    std::unique_ptr<RankIndex<false>> lazy_rank;
    if (include_lazy) {
        lazy_window = std::make_unique<WindowIndex<false>>(gs);
        lazy_rank = std::make_unique<RankIndex<false>>(gs);
    }
    std::vector<Edge> a,b,c;
    for (auto t : ts) {
        collect(kernel,t,a);
        collect(window,t,b);
        collect(rank,t,c);
        if (a!=b || a!=c) throw std::runtime_error("per-tick edge disagreement at "+std::to_string(t.now));
        if (include_lazy) {
            collect(*lazy_window,t,b);
            collect(*lazy_rank,t,c);
            if (a!=b || a!=c) throw std::runtime_error("lazy edge disagreement at "+std::to_string(t.now));
        }
    }
}

double ns(Clock::duration d) { return std::chrono::duration<double,std::nano>(d).count(); }
template<class Engine>
void run(const std::string& method,U rep,const Options& o,const std::vector<Guard>& gs,
         const std::vector<Tick>& ts, U arrivals, U occupied, size_t ranks, size_t windows) {
    std::vector<double> lat(ts.size());
    auto start=Clock::now();
    Engine engine(gs);
    auto built=Clock::now();
    Digest digest;
    auto begin=Clock::now();
    for (size_t i=0;i<ts.size();++i) {
        auto before=Clock::now();
        engine.step(ts[i].now,ts[i].arrivals,[&](std::uint32_t id,bool state) {digest(id,state);});
        lat[i]=ns(Clock::now()-before);
    }
    auto end=Clock::now();
    std::sort(lat.begin(),lat.end());
    std::cout << "{\"name\":\"" << o.name << "\",\"method\":\"" << method
              << "\",\"rep\":" << rep << ",\"seed\":" << o.seed << ",\"guards\":" << gs.size()
              << ",\"distinct_k\":" << ranks << ",\"distinct_windows\":" << windows
              << ",\"ticks\":" << ts.size() << ",\"nonempty_ticks\":" << occupied
              << ",\"arrivals\":" << arrivals << ",\"edges\":" << digest.count << ",\"digest\":\"" << digest.value
              << "\",\"build_ns\":" << static_cast<U>(ns(built-start))
              << ",\"p50_ns\":" << lat[(lat.size()-1)/2]
              << ",\"p99_ns\":" << lat[lat.size() - 1 - lat.size()/100]
              << ",\"elapsed_ns\":" << static_cast<U>(ns(end-begin)) << "}\n";
}
int main(int argc,char** argv) try {
    Options o=options(argc,argv);
    std::mt19937_64 guard_rng(o.seed),tick_rng(o.seed+1);
    auto gs=guards(o,guard_rng);
    auto ts=ticks(o,tick_rng);
    U arrivals=0, occupied=0;
    for (auto t:ts) { arrivals+=t.arrivals; occupied+=t.arrivals!=0; }
    std::map<U,bool> ks,ws;
    for (auto [k,w]:gs) { ks.emplace(k,true); ws.emplace(w,true); }
    check(gs,ts,o.include_lazy);
    const U methods = o.include_lazy ? 5 : 3;
    for (U rep=0;rep<o.reps;++rep) {
        for (U pos=0;pos<methods;++pos) switch ((rep+pos)%methods) {
            case 0: run<kthwake::Index>("kthwake",rep,o,gs,ts,arrivals,occupied,ks.size(),ws.size()); break;
            case 1: run<RankIndex<true>>("group-by-k",rep,o,gs,ts,arrivals,occupied,ks.size(),ws.size()); break;
            case 2: run<WindowIndex<true>>("group-by-window",rep,o,gs,ts,arrivals,occupied,ks.size(),ws.size()); break;
            case 3: run<RankIndex<false>>("rank-snapshot",rep,o,gs,ts,arrivals,occupied,ks.size(),ws.size()); break;
            case 4: run<WindowIndex<false>>("window-snapshot",rep,o,gs,ts,arrivals,occupied,ks.size(),ws.size()); break;
        }
    }
    return 0;
} catch (const std::exception& e) {
    std::cerr<<"bench: "<<e.what()<<'\n';
    return 1;
}
