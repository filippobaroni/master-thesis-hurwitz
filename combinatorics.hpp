#pragma once

#include <algorithm>
#include <atomic>
#include <cmath>
#include <numeric>
#include <thread>
#include <tuple>
#include <vector>

namespace std {
    template<typename F>
    class _y_combinator_impl {
        F f;
    public:
        template<typename G>
        explicit _y_combinator_impl(G g) : f(g) { }
        template<typename ...Args>
        decltype(auto) operator () (Args&& ...args) {
            return f(std::ref(*this), std::forward<Args>(args)...);
        }
    };
    
    template<typename F>
    decltype(auto) y_combinator(F f) {
        return _y_combinator_impl<F>(f);
    }
}


template<typename T>
using partitions = std::vector<std::vector<T>>;
template<typename T>
using character_table = std::vector<std::vector<T>>;
template<typename T>
using secondary_partition = std::vector<std::tuple<T, uint32_t>>;
using mult_table = std::vector<std::vector<std::vector<std::vector<uint32_t>>>>;
template<typename I, typename T>
using multivariate_polynomial = std::tuple<T, std::vector<std::tuple<I, uint32_t>>>;


template<typename T>
auto generate_partitions(T n) {
    std::vector<T> p(n + 1, 0);
    p[1] = n;
    partitions<T> ans;
    uint32_t k = 1;
    while(k != 0) {
        T x = p[k - 1] + 1;
        T y = p[k] - 1;
        --k;
        while(x <= y) {
            p[k] = x;
            y -= x;
            ++k;
        }
        p[k] = x + y;
        ans.emplace_back(p.begin(), p.begin() + k + 1);
        std::reverse(ans.back().begin(), ans.back().end());
    }
    std::sort(ans.begin(), ans.end());
    return ans;
}

template<typename T>
auto partitions_table(T n) {
    std::vector<partitions<T>> ans(n + 1);
    ans[0] = {{}};
    for(T k = 1; k <= n; ++k) {
        ans[k] = generate_partitions<T>(k);
    }
    return ans;
}

template<typename T, typename F>
void iterate_on_secondary_partitions(const std::vector<partitions<T>>& p_table, T d, F f) {
    const auto& ps = p_table[d];
    for(const auto& p : ps) {
        secondary_partition<T> omega;
        std::y_combinator([&](auto self, uint32_t i, T pr_d, uint32_t pr_j) -> void {
            if(i == p.size()) {
                f(omega);
                return;
            }
            const auto& ps1 = p_table[p[i]];
            uint32_t j = 0;
            if(pr_d == p[i]) {
                j = pr_j;
            }
            for(; j < ps1.size(); ++j) {
                omega.emplace_back(p[i], j);
                self(i + 1, p[i], j);
                omega.pop_back();
            }
        })(0, -1, 0);
    }
}

template<typename T, typename F>
void parallel_iterate_on_secondary_partitions(const int thread_num, const std::vector<partitions<T>>& p_table, T d, F f) {
    const auto& ps = p_table[d];
    std::atomic<uint32_t> idx = 0;
    std::vector<std::thread> threads;
    for(int t = 0; t < thread_num; ++t) {
        threads.emplace_back([&, t]() {
            while(true) {
                uint32_t myidx = ++idx;
                if(myidx >= ps.size()) {
                    return;
                }
                const auto& p = ps[myidx];
                secondary_partition<T> omega;
                std::y_combinator([&](auto self, uint32_t i, T pr_d, uint32_t pr_j) -> void {
                    if(i == p.size()) {
                        f(t, omega);
                        return;
                    }
                    const auto& ps1 = p_table[p[i]];
                    uint32_t j = 0;
                    if(pr_d == p[i]) {
                        j = pr_j;
                    }
                    for(; j < ps1.size(); ++j) {
                        omega.emplace_back(p[i], j);
                        self(i + 1, p[i], j);
                        omega.pop_back();
                    }
                })(0, -1, 0);
            }
        });
    }
    for(auto& t : threads) {
        t.join();
    }
}

template<typename T>
uint32_t find_partition_idx(const std::vector<partitions<T>>& p_table, const std::vector<T>& p) {
    T n = std::accumulate(p.begin(), p.end(), T(0));
    return std::lower_bound(p_table[n].begin(), p_table[n].end(), p) - p_table[n].begin();
}

template<typename I, typename T>
auto conjugacy_classes(const std::vector<partitions<T>>& p_table) {
    std::vector<std::vector<I>> C;
    I n_fact = 1;
    for(uint32_t n = 0; n < p_table.size(); ++n) {
        if(n > 0) {
            n_fact *= I(n);
        }
        C.emplace_back();
        for(const auto& p : p_table[n]) {
            I den = 1;
            std::vector<uint32_t> freq(n + 1);
            for(auto i : p) {
                ++freq[i];
            }
            for(uint32_t j = 1; j <= n; ++j) {
                for(uint32_t l = 1; l <= freq[j]; ++l) {
                    den *= j * l;
                }
            }
            C[n].push_back(n_fact / den);
        }
    }
    return C;
}

template<typename T>
auto multiplication_table(const std::vector<partitions<T>>& p_table) {
    T d = p_table.size() - 1;
    mult_table mult(d + 1);
    for(T k = 0; k <= d; ++ k) {
        mult[k].resize(d + 1);
        for(T h = 0; h <= d; ++h) {
            if(h + k > d) {
                continue;
            }
            mult[k][h].resize(p_table[k].size());
            for(uint32_t i = 0; i < p_table[k].size(); ++i) {
                mult[k][h][i].resize(p_table[h].size());
                for(uint32_t j = 0; j < p_table[h].size(); ++j) {
                    auto u = p_table[k][i];
                    u.insert(u.end(), p_table[h][j].begin(), p_table[h][j].end());
                    sort(u.rbegin(), u.rend());
                    mult[k][h][i][j] = find_partition_idx(p_table, u);
                }
            }
        }
    }
    return mult;
}

template<typename I, typename T>
void normalize_polynomial(multivariate_polynomial<I, T>& p) {
    auto& [d, v] = p;
    std::sort(v.begin(), v.end(), [](const auto& x, const auto& y) {
        return std::get<1>(x) < std::get<1>(y);
    });
    auto it = v.begin();
    for(auto jt = v.begin() + 1; jt != v.end(); ++jt) {
        if(std::get<1>(*it) == std::get<1>(*jt)) {
            std::get<0>(*it) += std::get<0>(*jt);
        } else {
            ++it;
            *it = *jt;
        }
    }
    v.erase(it + 1, v.end());
    it = v.begin();
    for(auto jt = v.begin(); jt != v.end(); ++jt) {
        *it = *jt;
        if(std::get<0>(*it) != 0) {
            ++it;
        }
    }
    v.erase(it, v.end());
}

template<typename I, typename T>
auto mult_two_polynomials(const mult_table& M, const multivariate_polynomial<I, T>& p, const multivariate_polynomial<I, T>& q) {
    multivariate_polynomial<I, T> ans = { T(std::get<0>(p) + std::get<0>(q)), {} };
    std::get<1>(ans).reserve(std::get<1>(p).size() * std::get<1>(q).size());
    for(const auto& [c, u] : std::get<1>(p)) {
        for(const auto& [d, v] : std::get<1>(q)) {
            std::get<1>(ans).emplace_back(c * d, M[std::get<0>(p)][std::get<0>(q)][u][v]);
        }
    }
    normalize_polynomial(ans);
    return ans;
}

template<typename I, typename T>
auto mult_many_polynomials(const mult_table& M, std::vector<multivariate_polynomial<I, T>> ps) {
    std::vector<multivariate_polynomial<I, T>> ps1;
    while(ps.size() > 1) {
        std::sort(ps.begin(), ps.end(), [&](const auto& p, const auto& q) { return std::get<1>(p).size() > std::get<1>(q).size(); });
        for(uint32_t i = 0; 2 * i + 1 < ps.size(); ++i) {
            ps1.push_back(mult_two_polynomials(M, ps[i], ps[ps.size() - 1 - i]));
        }
        if(ps.size() & 1) {
            ps1.push_back(std::move(ps[ps.size() / 2]));
        }
        std::swap(ps, ps1);
        ps1.clear();
    }
    return ps[0];
    /*auto ans = ps[0];
    for(uint32_t i = 1; i < ps.size(); ++i) {
        ans = mult_two_polynomials(M, ans, ps[i]);
    }
    return ans;*/
    
    /*uint64_t product = 1;
    for(const auto& [d, v] : ps) {
        product *= v.size();
    }
    auto cmp = [&](const auto& p, const auto& q) { return std::get<1>(p).size() > std::get<1>(q).size(); };
    std::sort(ps.begin(), ps.end(), cmp);
    return std::y_combinator([&](auto rec, uint32_t b, uint32_t e, uint64_t prod) -> auto {
        //std::cerr << b << " " << e << " " << prod << std::endl;
        if(e - b == 0) {
            return multivariate_polynomial<I, T>(0, {{1, 0}});
        }
        else if(e - b == 1) {
            return ps[b];
        } else if(e - b == 2) {
            auto x = mult_two_polynomials(M, ps[b], ps[b + 1]);
            return x;
        }
        uint32_t s = sqrt(prod);
        uint32_t j = b + 1;
        uint64_t pp = std::get<1>(ps[b]).size();
        for(uint32_t i = b + 1; i < e; ++i) {
            if(pp * std::get<1>(ps[i]).size() >= s) {
                continue;
            } else {
                pp *= std::get<1>(ps[i]).size();
                std::swap(ps[i], ps[j]);
                ++j;
            }
        }
        std::sort(ps.begin() + j, ps.begin() + e, cmp);
        auto x = mult_two_polynomials(M, rec(b, j, pp), rec(j, e, prod / pp));
        return x;
    })(0, ps.size(), product);*/
    
    /*auto cmp = [](const auto& p, const auto& q) {
        return std::get<1>(p).size() < std::get<1>(q).size();
    };
    std::make_heap(ps.begin(), ps.end(), cmp);
    for(uint32_t i = ps.size(); i > 1; --i) {
        std::pop_heap(ps.begin(), ps.begin() + i, cmp);
        std::pop_heap(ps.begin(), ps.begin() + i - 1, cmp);
        ps[i - 2] = mult_two_polynomials(M, ps[i - 2], ps[i - 1]);
        std::push_heap(ps.begin(), ps.begin() + i - 1, cmp);
    }
    return ps[0];*/
}

template<typename I, typename T>
auto character_tables(const std::vector<partitions<T>>& p_table) {
    uint32_t n = p_table.size() - 1;
    std::vector<character_table<I>> ch(n + 1);
    ch[0] = {{1}};
    for(uint32_t k = 1; k <= n; ++k) {
        const auto& ps = p_table[k];
        ch[k].assign(ps.size(), std::vector<I>(ps.size(), 0));
        for(uint32_t l = 0; l < ps.size(); ++l) {
            auto lv = ps[l];
            for(uint32_t r = 0; r < ps.size(); ++r) {
                auto rv = ps[r];
                uint32_t rprime = find_partition_idx(p_table, std::vector<T>(rv.begin() + 1, rv.end()));
                for(uint32_t i = 0; i < lv.size(); ++i) {
                    auto lprimev = lv;
                    uint32_t j = i, x = rv[0];
                    while(x > 0) {
                        if(j + 1 < lv.size()) {
                            if(lprimev[j] >= lprimev[j + 1]) {
                                --lprimev[j];
                                --x;
                            } else {
                                ++j;
                            }
                        } else {
                            if(lprimev[j]) {
                                --lprimev[j];
                                --x;
                            } else {
                                break;
                            }
                        }
                    }
                    if(x == 0 and (j + 1 == lv.size() or lprimev[j + 1] <= lprimev[j])) {
                        while(lprimev.size() and lprimev.back() == 0) {
                            lprimev.pop_back();
                        }
                        uint32_t lprime = find_partition_idx(p_table, lprimev);
                        I sign = ((j - i) % 2) ? (-1) : (1);
                        ch[k][l][r] += sign * ch[k - rv[0]][lprime][rprime];
                    }
                }
            }
        }
    }
    return ch;
}
