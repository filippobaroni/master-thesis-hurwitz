#pragma once

#include <tuple>
#include <utility>
#include <vector>


template<uint32_t d, typename T>
constexpr auto make_ntuple(const T& x = T()) {
    if constexpr(d == 1) {
        return std::tuple(x);
    } else {
        return std::tuple_cat(std::tuple(x), make_ntuple<d - 1>(x));
    }
}
template<uint32_t d, typename T>
using ntuple = decltype(make_ntuple<d, T>());

template<typename Tuple>
constexpr auto tuple_tail(const Tuple& t) {
    return std::apply([](auto&&, auto&& ...xs) { return std::tuple(xs...); }, t);
}

template<size_t d, typename T, size_t ...Is>
auto _array_to_tuple_impl(const std::array<T, d>& a, std::index_sequence<Is...>) {
    return std::tuple(a[Is]...);
}
template<size_t d, typename T>
auto array_to_tuple(const std::array<T, d>& a) {
    return _array_to_tuple_impl(a, std::make_index_sequence<d>{});
}

template<uint32_t d, typename T>
class ndvector;
template<uint32_t d, typename T>
class ndvector : public std::vector<ndvector<d - 1, T>> {
public:
    ndvector(const ntuple<d, size_t>& sz = make_ntuple<d, size_t>(0), const T& x = T()) :
        std::vector<ndvector<d - 1, T>>(std::get<0>(sz), ndvector<d - 1, T>(tuple_tail(sz), x)) { }
    decltype(auto) operator [] (const size_t& w) {
        return std::vector<ndvector<d - 1, T>>::operator [] (w);
    }
    template<typename SZ>
    decltype(auto) operator [] (const SZ& w) {
        return (std::vector<ndvector<d - 1, T>>::operator [] (std::get<0>(w)))[tuple_tail(w)];
    }
};
template<typename T>
class ndvector<1, T> : public std::vector<T> {
public:
    ndvector(const ntuple<1, size_t>& sz = {0}, const T& x = T()) :
        std::vector<T>(std::get<0>(sz), x) { }
    decltype(auto) operator [] (const size_t& w) {
        return std::vector<T>::operator [] (w);
    }
    template<typename SZ>
    decltype(auto) operator [] (const SZ& w) {
        return std::vector<T>::operator [] (std::get<0>(w));
    }
};
