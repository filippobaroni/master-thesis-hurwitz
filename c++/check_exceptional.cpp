#include <atomic>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <mutex>
#include <thread>
#include <random>
#include <string>
#include <sstream>
#include <vector>

#include <gmpxx.h>

#pragma GCC diagnostic ignored "-Wdeprecated-copy"
#pragma GCC diagnostic ignored "-Wstrict-aliasing"
#pragma GCC diagnostic ignored "-Wunused-parameter"
#include "cpp-btree/btree_map.h"
#pragma GCC diagnostic pop

#include "combinatorics.hpp"
#include "imod.hpp"
#include "ndvector.hpp"

#include "zheng.hpp"

template<typename I, typename T, uint32_t n>
struct _check_exceptional_data_impl {
    auto operator () (const uint32_t d, const std::vector<std::array<uint32_t, n>>& data) {
        auto P = partitions_table<T>(d)[d];
        
        std::atomic<uint32_t> idx = 0;
        std::vector<std::thread> threads;
        std::mutex mutex;
        std::vector<std::array<uint32_t, n>> exceptional;
        
        debug << "Partitions: " << P.size() << std::endl;
        
        timer::start("r and s");
        auto r_and_s = compute_r_and_s<I, T>(d);
        timer::end("r and s");
        debug << "Secondary partitions: " << r_and_s.size() << std::endl;
        
        debug << "Chunk size: " << 1 << std::endl;
        
        debug << "RAM usage: " << get_used_RAM() << std::endl;
        
        timer::start("exceptional data check");
        for(int t = 0; t < THREADS; ++t) {
            threads.emplace_back([&, t]() {
                while(true) {
                    uint32_t myidx = idx++;
                    if(myidx >= data.size()) {
                        break;
                    }
                    if(data.size() < 100 or myidx % (data.size() / 100) == 0) {
                        std::scoped_lock lock(mutex);
                        std::cerr << std::setw(3) << (100 * myidx / data.size()) << std::setw(0) << "%    in " << timer::elapsed("exceptional data check") << "s" << std::endl;
                    }
                    auto mydatum = data[myidx];
                    I mycoeff = 0;
                    for(const auto& [r, s] : r_and_s) {
                        std::y_combinator([&](auto rec, uint32_t k, uint32_t j, I c) -> void {
                            for(uint32_t i = j; i < s.size(); ++i) {
                                if(std::get<1>(s[i]) != mydatum[k]) {
                                    continue;
                                }
                                if(k + 1 == n) {
                                    mycoeff += c * std::get<0>(s[i]);
                                } else {
                                    rec(k + 1, i, c * std::get<0>(s[i]));
                                }
                            }
                        })(0, 0, r);
                    }
                    if(mycoeff == 0) {
                        std::scoped_lock lock(mutex);
                        exceptional.push_back(mydatum);
                    }
                }
            });
        }
        for(auto& t : threads) {
            t.join();
        }
        std::sort(exceptional.begin(), exceptional.end());
        timer::end("exceptional data check");
        return exceptional;
    }
};
template<typename I, typename T>
struct _check_exceptional_data_impl<I, T, 3> {
    auto operator () (const uint32_t d, const std::vector<std::array<uint32_t, 3>>& data) {
        auto P = partitions_table<T>(d)[d];
        
        std::atomic<uint32_t> idx = 0;
        std::vector<std::thread> threads;
        std::mutex mutex;
        std::vector<std::array<uint32_t, 3>> exceptional;
        
        debug << "Partitions: " << P.size() << std::endl;
        
        timer::start("r and s");
        auto r_and_s = compute_r_and_s<I, T>(d);
        timer::end("r and s");
        debug << "Secondary partitions: " << r_and_s.size() << std::endl;
        
        debug << "Chunk size: " << 1 << std::endl;
        
        debug << "RAM usage: " << get_used_RAM() << std::endl;
        
        timer::start("exceptional data check");
        for(int t = 0; t < THREADS; ++t) {
            threads.emplace_back([&, t]() {
                while(true) {
                    uint32_t myidx = idx++;
                    if(myidx >= data.size()) {
                        break;
                    }
                    if(data.size() < 100 or myidx % (data.size() / 100) == 0) {
                        std::scoped_lock lock(mutex);
                        std::cerr << std::setw(3) << (100 * myidx / data.size()) << std::setw(0) << "%    in " << timer::elapsed("exceptional data check") << "s" << std::endl;
                    }
                    auto mydatum = data[myidx];
                    I mycoeff = 0;
                    for(const auto& [r, s] : r_and_s) {
                        for(uint32_t i = 0; i < s.size(); ++i) {
                            if(std::get<1>(s[i]) != mydatum[0]) {
                                continue;
                            }
                            for(uint32_t j = i; j < s.size(); ++j) {
                                if(std::get<1>(s[j]) != mydatum[1]) {
                                    continue;
                                }
                                for(uint32_t k = j; k < s.size(); ++k) {
                                    if(std::get<1>(s[k]) == mydatum[2]) {
                                        mycoeff += r * std::get<0>(s[i]) * std::get<0>(s[j]) * std::get<0>(s[k]);
                                    }
                                }
                            }
                        }
                    }
                    if(mycoeff == 0) {
                        std::scoped_lock lock(mutex);
                        exceptional.push_back(mydatum);
                    }
                }
            });
        }
        for(auto& t : threads) {
            t.join();
        }
        std::sort(exceptional.begin(), exceptional.end());
        timer::end("exceptional data check");
        return exceptional;
    }
};

template<typename I, typename T, uint32_t n>
auto check_exceptional_data(uint32_t d, const std::vector<std::array<uint32_t, n>>& data) {
    return _check_exceptional_data_impl<I, T, n>{}(d, data);
}

int main(int argc, char** argv) {
    //using T = imod<1000000009>;
    using T = mpq_class;
    constexpr int n = 5;
    
    debug.open("debug.txt");
    
    assert(argc == 2);
    int d = atoi(argv[1]);
    
    auto p_table = partitions_table<uint8_t>(d);
    
    std::vector<std::array<uint32_t, n>> data;
    for(std::string line; std::getline(std::cin, line); ) {
        if(line.empty() or line[0] != '[') {
            break;
        }
        data.emplace_back();
        uint32_t i = 0;
        std::istringstream in(line);
        for(char bracket; in >> std::ws >> bracket; ) {
            std::vector<uint8_t> p;
            for(int x; in >> x >> std::ws; ) {
                p.push_back(x);
                if(in.peek() == ']') {
                    in.ignore();
                    break;
                }
            }
            data.back()[i++] = find_partition_idx(p_table, p);
        }
    }
    
    auto E = check_exceptional_data<T, uint8_t, n>(d, data);
    
    for(const auto& e : E) {
        for(auto i : e) {
            print_partition(std::cout, p_table[d][i]) << " ";
        }
        std::cout << "\n";
    }
}
