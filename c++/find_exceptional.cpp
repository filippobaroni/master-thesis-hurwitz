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
#include <vector>

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
struct _exceptional_data_impl {
    auto operator () (const uint32_t d) {
        auto P = partitions_table<T>(d)[d];
        std::vector<uint32_t> PS;
        for(const auto& p : P) {
            PS.push_back(p.size());
        }
        auto hurwitz = [&](int sum_len) {
            return (sum_len + n * d) % 2 == 0 and (int(n) - 2) * int(d) + 2 >= sum_len;
        };
        
        std::atomic<uint32_t> idx = 1;
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
        
        timer::start("exceptional data");
        for(int t = 0; t < THREADS; ++t) {
            threads.emplace_back([&, t]() {
                while(true) {
                    uint32_t myidx = idx++;
                    if(myidx >= P.size()) {
                        break;
                    }
                    ndvector<n - 1, I> myres(make_ntuple<n - 1, size_t>(P.size()));
                    uint32_t curr = 0;
                    for(const auto& [r, s] : r_and_s) {
                        std::array<uint32_t, n - 1> p;
                        uint32_t sum_len = 0;
                        for(uint32_t i0 = 0; i0 < s.size(); ++i0) {
                            if(std::get<1>(s[i0]) != myidx) {
                                continue;
                            }
                            sum_len = PS[std::get<1>(s[i0])];
                            std::y_combinator([&](auto rec, uint32_t k, uint32_t j, I c) -> void {
                                for(uint32_t i = j; i < s.size(); ++i) {
                                    p[k] = std::get<1>(s[i]);
                                    sum_len += PS[p[k]];
                                    if(k + 2 == n) {
                                        if(hurwitz(sum_len)) {
                                            myres[array_to_tuple(p)] += c * std::get<0>(s[i]);
                                        }
                                    } else {
                                        rec(k + 1, i, c * std::get<0>(s[i]));
                                    }
                                    sum_len -= PS[p[k]];
                                }
                            })(0, i0, r * std::get<0>(s[i0]));
                        }
                        ++curr;
                        if(r_and_s.size() < 100 or curr % (r_and_s.size() / 100) == 0) {
                            std::scoped_lock lock(mutex);
                            std::cerr << "[" << std::setw(4) << myidx << "]   "  << std::setw(3) << (100 * curr / r_and_s.size()) << std::setw(0) << "%    in " << timer::elapsed("exceptional data") << "s" << std::endl;
                        }
                    }
                    {
                        decltype(exceptional) myexc;
                        std::array<uint32_t, n - 1> p;
                        uint32_t sum_len = 0;
                        uint32_t i0 = myidx;
                        sum_len = PS[i0];
                        std::y_combinator([&](auto rec, uint32_t k, uint32_t j) -> void {
                            for(uint32_t i = j; i < P.size(); ++i) {
                                p[k] = i;
                                sum_len += PS[i];
                                if(k + 2 == n) {
                                    if(hurwitz(sum_len) and myres[array_to_tuple(p)] == 0) {
                                        std::array<uint32_t, n> pp;
                                        pp[0] = i0;
                                        for(uint32_t l = 1; l < n; ++l) {
                                            pp[l] = p[l - 1];
                                        }
                                        myexc.push_back(pp);
                                    }
                                } else {
                                    rec(k + 1, i);
                                }
                                sum_len -= PS[p[k]];
                            }
                        })(0, i0);
                        std::scoped_lock lock(mutex);
                        exceptional.insert(exceptional.end(), myexc.begin(), myexc.end());
                    }
                }
            });
        }
        for(auto& t : threads) {
            t.join();
        }
        std::sort(exceptional.begin(), exceptional.end());
        timer::end("exceptional data");
        return exceptional;
    }
};
template<typename I, typename T>
struct _exceptional_data_impl<I, T, 3> {
    auto operator () (const uint32_t d) {
        auto P = partitions_table<T>(d)[d];
        std::vector<uint32_t> PS;
        for(const auto& p : P) {
            PS.push_back(p.size());
        }
        auto hurwitz = [&](int sum_len) {
            return (sum_len + d) % 2 == 0 and int(d) + 2 >= sum_len;
        };
        
        std::atomic<uint32_t> idx = 1;
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
        
        timer::start("exceptional data");
        for(int t = 0; t < THREADS; ++t) {
            threads.emplace_back([&, t]() {
                while(true) {
                    uint32_t myidx = idx++;
                    if(myidx >= P.size()) {
                        break;
                    }
                    ndvector<2, I> myres({ P.size(), P.size() });
                    uint32_t curr = 0;
                    for(const auto& [r, s] : r_and_s) {
                        for(uint32_t i = 0; i < s.size(); ++i) {
                            if(std::get<1>(s[i]) != myidx) {
                                continue;
                            }
                            for(uint32_t j = i; j < s.size(); ++j) {
                                I c = r * std::get<0>(s[i]) * std::get<0>(s[j]);
                                for(uint32_t k = j; k < s.size(); ++k) {
                                    if(hurwitz(PS[std::get<1>(s[i])] + PS[std::get<1>(s[j])] + PS[std::get<1>(s[k])])) {
                                        myres[std::get<1>(s[j])][std::get<1>(s[k])] += c * std::get<0>(s[k]);
                                    }
                                }
                            }
                        }
                        ++curr;
                        if(r_and_s.size() < 100 or curr % (r_and_s.size() / 100) == 0) {
                            std::scoped_lock lock(mutex);
                            std::cerr << "[" << std::setw(4) << myidx << "]   "  << std::setw(3) << (100 * curr / r_and_s.size()) << std::setw(0) << "%    in " << timer::elapsed("exceptional data") << "s" << std::endl;
                        }
                    }
                    {
                        decltype(exceptional) myexc;
                        uint32_t i = myidx;
                        for(uint32_t j = i; j < P.size(); ++j) {
                            for(uint32_t k = j; k < P.size(); ++k) {
                                if(hurwitz(PS[i] + PS[j] + PS[k]) and myres[j][k] == 0) {
                                    myexc.push_back({ i, j, k });
                                }
                            }
                        }
                        std::scoped_lock lock(mutex);
                        exceptional.insert(exceptional.end(), myexc.begin(), myexc.end());
                    }
                }
            });
        }
        for(auto& t : threads) {
            t.join();
        }
        std::sort(exceptional.begin(), exceptional.end());
        timer::end("exceptional data");
        return exceptional;
    }
};

template<typename I, typename T, uint32_t n>
auto exceptional_data(uint32_t d) {
    return _exceptional_data_impl<I, T, n>{}(d);
}

int main(int argc, char** argv) {
    using imodd = imod<1000000007>;
    
    debug.open("debug.txt");
    
    assert(argc == 2);
    int d = atoi(argv[1]);
    
    auto p_table = partitions_table<uint8_t>(d);
    
    auto E = exceptional_data<imodd, uint8_t, 4>(d);
    
    for(const auto& e : E) {
        for(auto i : e) {
            print_partition(std::cout, p_table[d][i]) << " ";
        }
        std::cout << "\n";
    }
}
