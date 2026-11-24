// Compare the actual SuperIntervals C++ library under the same workload,
// oracle, compiler and measurement harness as Guardshift's native controls.
#define GUARDSHIFT_BENCH_LIBRARY
#include "bench.cpp"
#undef GUARDSHIFT_BENCH_LIBRARY
#include "superintervals_adapters.hpp"

namespace {

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
        const bool crossover = argc == 2 && std::string(argv[1]) == "--crossover";
        if (argc != 2) throw std::invalid_argument("usage: external replay/azure-code.csv | --crossover");
        for (unsigned seed : {17u, 42u, 217u}) {
            compare(make_work("atomic", 8192, 1200, seed), seed, 3);
            if (crossover) {
                compare(make_work("jump", 32768, 512, seed), seed, 3);
                continue;
            }
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
