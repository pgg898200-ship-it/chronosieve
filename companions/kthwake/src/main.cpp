#include "kthwake.hpp"

#include <charconv>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>

namespace {
constexpr const char* usage = R"(Usage: kthwake [--idle] GUARDS TICKS
GUARDS contains one positive k,window pair per line. IDs start at zero.
TICKS contains ordered time,arrivals pairs. Use - to read ticks from stdin.
Decimal integers may be separated by comma, space or tab. # starts a comment.
Output is JSONL: one {"t":time,"id":id,"active":boolean} per changed rule.
Each tick is an atomic snapshot. --idle also emits idle transitions between
input rows and drains remaining representable deadlines after end of input.
)";

struct Pair { std::uint64_t a, b; };

void trim(std::string_view& value) {
    value.remove_prefix(std::min(value.size(), value.find_first_not_of(" \t\r")));
}

std::uint64_t number(std::string_view& value) {
    std::uint64_t out = 0;
    const auto end = value.data() + value.size();
    const auto result = std::from_chars(value.data(), end, out, 10);
    if (result.ec != std::errc{}) throw std::invalid_argument("expected an unsigned decimal integer");
    value.remove_prefix(static_cast<std::size_t>(result.ptr - value.data()));
    return out;
}

template<class Consumer>
void rows(std::istream& input, const std::string& name, Consumer consume) {
    std::string line;
    std::size_t line_number = 0;
    while (std::getline(input, line)) {
        ++line_number;
        std::string_view s(line);
        s = s.substr(0, s.find('#'));
        trim(s);
        if (s.empty()) continue;
        try {
            auto a = number(s);
            if (s.empty() || s.find_first_of(" ,\t") != 0)
                throw std::invalid_argument("expected two columns");
            trim(s);
            if (!s.empty() && s.front() == ',') s.remove_prefix(1);
            trim(s);
            auto b = number(s);
            trim(s);
            if (!s.empty()) throw std::invalid_argument("unexpected trailing data");
            consume(Pair{a,b});
        } catch (const std::exception& e) {
            throw std::runtime_error(name + ":" + std::to_string(line_number) + ": " + e.what());
        }
    }
    if (input.bad()) throw std::runtime_error("read failed: " + name);
}

std::ifstream open_input(const std::string& path) {
    std::ifstream file(path);
    if (!file) throw std::runtime_error("cannot open " + path);
    return file;
}

int run(int argc, char** argv) {
    if (argc == 2 && std::string_view(argv[1]) == "--help") {
        std::cout << usage;
        return 0;
    }
    bool idle = argc > 1 && std::string_view(argv[1]) == "--idle";
    if (argc != (idle ? 4 : 3)) {
        std::cerr << usage;
        return 2;
    }
    const std::string guard_path = argv[idle ? 2 : 1], tick_path = argv[idle ? 3 : 2];
    auto guard_input = open_input(guard_path);
    std::vector<kthwake::Guard> guards;
    rows(guard_input, guard_path, [&](Pair p) {
        if (p.a == 0 || p.a > UINT32_MAX || p.b == 0)
            throw std::invalid_argument("require 1 <= k <= UINT32_MAX and window > 0");
        guards.push_back({static_cast<std::uint32_t>(p.a), p.b});
    });
    kthwake::Index index(guards);
    auto apply = [&](std::uint64_t t, std::uint64_t count) {
        index.step(t, count, [t](std::uint32_t id, bool active) {
            std::cout << "{\"t\":" << t << ",\"id\":" << id << ",\"active\":"
                      << (active ? "true}\n" : "false}\n");
        });
    };
    std::ifstream tick_file;
    if (tick_path != "-") tick_file = open_input(tick_path);
    rows(tick_path == "-" ? std::cin : tick_file, tick_path, [&](Pair p) {
        if (idle) {
            while (auto expiry = index.next_expiry()) {
                if (*expiry >= p.a) break;
                apply(*expiry, 0);
            }
        }
        apply(p.a, p.b);
    });
    if (idle) {
        while (auto expiry = index.next_expiry()) apply(*expiry, 0);
    }
    std::cout.flush();
    if (!std::cout) throw std::runtime_error("output write failed");
    return 0;
}
}
int main(int argc, char** argv) {
    std::ios::sync_with_stdio(false);
    std::cin.tie(nullptr);
    try { return run(argc, argv); }
    catch (const std::exception& e) { std::cerr << "kthwake: " << e.what() << '\n'; return 2; }
}
