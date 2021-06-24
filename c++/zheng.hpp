#pragma once

#include <chrono>
#include <fstream>
#include <string>
#include <vector>

#pragma GCC diagnostic ignored "-Wdeprecated-copy"
#pragma GCC diagnostic ignored "-Wstrict-aliasing"
#pragma GCC diagnostic ignored "-Wunused-parameter"
#include "cpp-btree/btree_map.h"
#pragma GCC diagnostic pop

#include "combinatorics.hpp"

constexpr int THREADS = 4;
constexpr long long RAM = 240 * (1LL << 30);

std::ofstream debug;


namespace timer {
    btree::btree_map<std::string, decltype(std::chrono::steady_clock::now())> begin_time;
    void start(const std::string& s) {
        begin_time[s] = std::chrono::steady_clock::now();
    }
    double elapsed(const std::string& s) {
        return std::chrono::duration<double>(std::chrono::steady_clock::now() - begin_time[s]).count();
    }
    void end(const std::string& s) {
        debug << s << ": " << elapsed(s) << "s\n";
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

uint64_t get_used_RAM() { // in Mb
    std::ifstream proc("/proc/self/status");
    while(true) {
        std::string s;
        proc >> s;
        if(s == "VmRSS:") {
            uint64_t ram;
            proc >> ram;
            return ram / 1024;
        }
    }
}

template<typename I, typename T>
auto compute_r_and_s(T d) {
    auto p_table = partitions_table<T>(d);
    auto conj = conjugacy_classes<I, T>(p_table);
    auto mult = multiplication_table(p_table);
    auto ch = character_tables<I>(p_table);
    std::vector<std::vector<std::vector<I>>> scoeff(d + 1);
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
    std::vector<std::tuple<I, std::vector<std::tuple<I, uint32_t>>>> r_and_s;
    std::vector<std::vector<std::tuple<I, std::vector<std::tuple<I, uint32_t>>>>> r_and_s_partial(THREADS);
    parallel_iterate_on_secondary_partitions(THREADS, p_table, d, [&](int t, auto omega) {
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
        // Compute s(omega)
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
        // Append
        r_and_s_partial[t].emplace_back(r, std::move(std::get<1>(s)));
    });
    for(auto& part : r_and_s_partial) {
        r_and_s.insert(r_and_s.end(), std::make_move_iterator(part.begin()), std::make_move_iterator(part.end()));
    }
    return r_and_s;
}


