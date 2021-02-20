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
        
        uint32_t first_p = 1;
        std::vector<std::thread> threads;
        std::mutex mutex;
        std::vector<std::array<uint32_t, n>> exceptional;
        
        std::vector<uint32_t> perm(P.size()), inv_perm(P.size());
        std::iota(perm.begin(), perm.end(), 0);
        std::shuffle(perm.begin() + 1, perm.end(), std::mt19937(std::random_device{}()));
        for(uint32_t i = 0; i < perm.size(); ++i) {
            inv_perm[perm[i]] = i;
        }
        
        debug << "Partitions: " << P.size() << std::endl;
        
        timer::start("r and s");
        auto r_and_s = compute_r_and_s<I, T>(d);
        long long RAM_USED = 0;
        for(const auto& [r, s] : r_and_s) {
            RAM_USED += sizeof(I) + (sizeof(I) + sizeof(T)) * s.size();
        }
        timer::end("r and s");
        debug << "Secondary partitions: " << r_and_s.size() << std::endl;
        
        const uint32_t CHUNK_SIZE = std::min(uint32_t(P.size() - 1) / THREADS + 1, uint32_t((RAM - RAM_USED) / (4 * THREADS * sizeof(I) * pow(P.size(), n - 1))));
        debug << "Chunk size: " << CHUNK_SIZE << std::endl;
        
        debug << "RAM usage: " << get_used_RAM() << "MB" << std::endl;
        
        timer::start("exceptional partitions");
        for(int t = 0; t < THREADS; ++t) {
            threads.emplace_back([&, t]() {
                while(true) {
                    uint32_t myp;
                    {
                        std::scoped_lock lock(mutex);
                        myp = first_p;
                        first_p += CHUNK_SIZE;
                        if(myp >= P.size()) {
                            break;
                        }
                    }
                    ndvector<n, I> myres(std::tuple_cat(std::tuple(CHUNK_SIZE), make_ntuple<n - 1, size_t>(P.size())));
                    uint32_t idx = 0;
                    for(const auto& [r, s] : r_and_s) {
                        std::array<uint32_t, n> p;
                        uint32_t sum_len = 0;
                        for(uint32_t i0 = 0; i0 < s.size(); ++i0) {
                            if(perm[std::get<1>(s[i0])] < myp or perm[std::get<1>(s[i0])] >= myp + CHUNK_SIZE) {
                                continue;
                            }
                            p[0] = perm[std::get<1>(s[i0])] - myp;
                            sum_len = PS[std::get<1>(s[i0])];
                            std::y_combinator([&](auto rec, uint32_t k, uint32_t j, I c) -> void {
                                for(uint32_t i = j; i < s.size(); ++i) {
                                    p[k] = std::get<1>(s[i]);
                                    sum_len += PS[p[k]];
                                    if(k + 1 == n) {
                                        if(hurwitz(sum_len)) {
                                            myres[array_to_tuple(p)] += c * std::get<0>(s[i]);
                                        }
                                    } else {
                                        rec(k + 1, i, c * std::get<0>(s[i]));
                                    }
                                    sum_len -= PS[p[k]];
                                }
                            })(1, i0, r * std::get<0>(s[i0]));
                        }
                        ++idx;
                        if(r_and_s.size() < 100 or idx % (r_and_s.size() / 100) == 0) {
                            std::scoped_lock lock(mutex);
                            std::cerr << "[" << std::setw(2) << t << "]   "  << std::setw(3) << (100 * idx / r_and_s.size()) << std::setw(0) << "%    in " << timer::elapsed("exceptional data") << "s" << std::endl;
                        }
                    }
                    {
                        decltype(exceptional) myexc;
                        std::array<uint32_t, n> p;
                        uint32_t sum_len = 0;
                        for(uint32_t i0 = myp; i0 < myp + CHUNK_SIZE and i0 < P.size(); ++i0) {
                            p[0] = i0 - myp;
                            sum_len = PS[inv_perm[i0]];
                            std::y_combinator([&](auto rec, uint32_t k, uint32_t j) -> void {
                                for(uint32_t i = j; i < P.size(); ++i) {
                                    p[k] = i;
                                    sum_len += PS[i];
                                    if(k + 1 == n) {
                                        if(hurwitz(sum_len) and myres[array_to_tuple(p)] == 0) {
                                            p[0] = inv_perm[i0];
                                            myexc.push_back(p);
                                            p[0] = i0 - myp;
                                        }
                                    } else {
                                        rec(k + 1, i);
                                    }
                                    sum_len -= PS[p[k]];
                                }
                            })(1, inv_perm[i0]);
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
    using imodd = imod<1000000009>;
    
    debug.open("debug.txt");
    
    assert(argc == 2);
    int d = atoi(argv[1]);
    
    auto p_table = partitions_table<uint8_t>(d);
    
    auto E = exceptional_data<imodd, uint8_t, 3>(d);
    
    for(const auto& e : E) {
        for(auto i : e) {
            print_partition(std::cout, p_table[d][i]) << " ";
        }
        std::cout << "\n";
    }
}
