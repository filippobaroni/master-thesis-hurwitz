#include <atomic>
#include <chrono>
#include <iostream>
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

constexpr int THREADS = 4;
constexpr long long RAM = 2048 * (1LL << 20);


namespace timer {
    btree::btree_map<std::string, decltype(std::chrono::steady_clock::now())> begin_time;
    void start(const std::string& s) {
        begin_time[s] = std::chrono::steady_clock::now();
    }
    double elapsed(const std::string& s) {
        return std::chrono::duration<double>(std::chrono::steady_clock::now() - begin_time[s]).count();
    }
    void end(const std::string& s) {
        std::cerr << s << ": " << elapsed(s) << "s\n";
    }
}


template<typename T>
std::ostream& print_partition(std::ostream& out, const std::vector<T>& p) {
    out << "[ ";
    for(auto x : p) {
        out << int(x) << " ";
    }
    return out << "]";
}

template<typename I, typename T>
class R_and_S {
    T d;
    decltype(partitions_table<T>(d)) p_table;
    decltype(conjugacy_classes<I, T>(p_table)) conj;
    decltype(multiplication_table(p_table)) mult;
    decltype(character_tables<I>(p_table)) ch;
    std::vector<std::vector<std::vector<I>>> scoeff;
    std::vector<I> R;
public:
    R_and_S(T d_) : d(d_),
        p_table(partitions_table<T>(d)),
        conj(conjugacy_classes<I, T>(p_table)),
        mult(multiplication_table(p_table)),
        ch(character_tables<I>(p_table)),
        scoeff(d + 1) {
            std::vector<I> factorial(d + 1), factorial_inv(d + 1);
            factorial[0] = factorial_inv[0] = 1;
            for(uint32_t j = 1; j <= uint32_t(d); ++j) {
                factorial[j] = factorial[j - 1] * I(j);
                factorial_inv[j] = factorial_inv[j - 1] / I(j);
            }
            for(T i = 0; i <= d; ++i) {
                scoeff[i].resize(ch[i].size());
                for(uint32_t j = 0; j < ch[i].size(); ++j) {
                    scoeff[i][j].resize(ch[i][j].size());
                    for(uint32_t k = 0; k < ch[i][j].size(); ++k) {
                        scoeff[i][j][k] = ch[i][j][k] * conj[i][k] / ch[i][j][0];
                    }
                }
            }
            iterate_on_secondary_partitions(p_table, d, [&](auto omega) {
                uint32_t k = omega.size();
                // Compute r(omega)
                I r = (k % 2) ? (1) : (-1);
                r *= factorial[k - 1];
                btree::btree_map<std::tuple<T, uint32_t>, uint32_t> multiplicities;
                for(const auto& [i, j] : omega) {
                    ++multiplicities[{i, j}];
                }
                for(const auto& [i, j] : multiplicities) {
                    r *= factorial_inv[j];
                }
                for(const auto& [i, j] : omega) {
                    I x = ch[i][j][0] * factorial_inv[i];
                    r *= x * x;
                }
                R.push_back(r);
            });
    }
    template<typename F>
    void iterate(F f) {
        auto it = R.begin();
        iterate_on_secondary_partitions(p_table, d, [&](auto omega) {
            multivariate_polynomial<I, T> s = { 0, {{1, 0}} };
            for(const auto& [i, j] : omega) {
                decltype(s) p = { i, {} };
                std::get<1>(p).reserve(p_table[i].size());
                for(uint32_t nu = 0; nu < p_table[i].size(); ++nu) {
                    std::get<1>(p).emplace_back(scoeff[i][j][nu], nu);
                }
                s = mult_two_polynomials(mult, s, p);
            }
            std::get<1>(s).erase(std::get<1>(s).begin());
            std::get<1>(s).shrink_to_fit();
            f(*it, std::get<1>(s));
            ++it;
        });
    }
};


template<typename I, typename T, uint32_t n>
auto exceptional_partitions(uint32_t d) {
    auto P = partitions_table<T>(d)[d];
    std::vector<uint32_t> PS;
    for(const auto& p : P) {
        PS.push_back(p.size());
    }
    auto hurwitz = [&](int sum_len) {
        return (sum_len + n * d) % 2 == 0 and (int(n) - 2) * int(d) + 2 >= sum_len;
    };
    auto RS = R_and_S<I, T>(d);

    std::vector<std::thread> threads;
    std::mutex mutex;
    std::vector<std::array<uint32_t, n>> exceptional;
    const uint32_t CHUNK_SIZE = std::min(uint32_t(P.size() - 1) / THREADS + 1, uint32_t(RAM / (THREADS * sizeof(I) * pow(P.size(), n - 1))));
    uint32_t first_p = 1, completed = 0;
    
    std::vector<uint32_t> perm(P.size()), inv_perm(P.size());
    std::iota(perm.begin(), perm.end(), 0);
    std::shuffle(perm.begin() + 1, perm.end(), std::mt19937(std::random_device{}()));
    for(uint32_t i = 0; i < perm.size(); ++i) {
        inv_perm[perm[i]] = i;
    }
    
    std::cerr << "Partitions: " << P.size() << std::endl;
    std::cerr << "Chunk size: " << CHUNK_SIZE << std::endl;
    
    timer::start("compute");
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
                std::vector<bool> visited(CHUNK_SIZE);
                RS.iterate([&](I r, const auto& s) {
                    std::array<uint32_t, n> p;
                    uint32_t sum_len = 0;
                    for(uint32_t i0 = 0; i0 < s.size(); ++i0) {
                        if(perm[std::get<1>(s[i0])] < myp or perm[std::get<1>(s[i0])] >= myp + CHUNK_SIZE) {
                            continue;
                        }
                        p[0] = perm[std::get<1>(s[i0])] - myp;
                        if(not visited[p[0]]) {
                            visited[p[0]] = true;
                            std::scoped_lock lock(mutex);
                            ++completed;
                            std::cerr << "[" << ((t < 10) ? (" ") : ("")) << t << "]   "  << completed << " / " << P.size() << "    in " << timer::elapsed("compute") << "s" << std::endl;
                        }
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
                });
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
    return exceptional;
}

int main(int argc, char** argv) {
    using imodd = imod<1000000007>;
    
    assert(argc == 2);
    int d = atoi(argv[1]);
    
    auto p_table = partitions_table<uint8_t>(d);
    //auto r_and_s = compute_r_and_s<imodd, uint8_t>(d);
    
    //std::cerr << r_and_s.size() << std::endl;
    
    timer::start("exceptional partitions");
    auto E = exceptional_partitions<imodd, uint8_t, 3>(d);
    timer::end("exceptional partitions");
    
    for(const auto& e : E) {
        for(auto i : e) {
            print_partition(std::cout, p_table[d][i]) << " ";
        }
        std::cout << "\n";
    }
}
