#pragma once

#include <type_traits>

using uint128_t = __uint128_t;

template<int M>
class imod {
    constexpr static uint128_t invM = -1ULL / M;
    
    int n;
    
    static int modular_inverse(int x, int y) {
        int z = x % y;
        if(z <= 1) {
            return z;
        }
        return y - int((uint64_t(y) * modular_inverse(y, z) - 1) / z);
    }
public:
    inline static constexpr int mod() {
        return M;
    }
    constexpr imod(int num = 0) : n(num % M) {
        n += (n < 0) * M;
    }
    inline constexpr const int& get() const {
        return n;
    }
    template<typename S, typename = std::enable_if_t<std::is_convertible_v<int, S>>>
    explicit inline constexpr operator S() const {
        return n;
    }
    friend constexpr bool operator == (const imod& x, const imod& y) {
        return x.n == y.n;
    }
    friend constexpr bool operator != (const imod& x, const imod& y) {
        return x.n != y.n;
    }
    friend constexpr bool operator < (const imod& x, const imod& y) {
        return x.n < y.n;
    }
    friend constexpr bool operator > (const imod& x, const imod& y) {
        return x.n > y.n;
    }
    friend constexpr bool operator <= (const imod& x, const imod& y) {
        return x.n <= y.n;
    }
    friend constexpr bool operator >= (const imod& x, const imod& y) {
        return x.n >= y.n;
    }
    auto inv() const {
        auto x = *this;
        x.n = modular_inverse(n, M);
        return x;
    }
    auto operator - () const {
        auto x = *this;
        x.n = M * (n > 0) - n;
        return x;
    }
    auto& operator += (const imod& other) {
        n += other.n;
        n -= (n >= M) * M;
        return *this;
    }
    friend constexpr auto operator + (imod x, const imod& y) {
        x += y;
        return x;
    }
    auto& operator -= (const imod& other) {
        n -= other.n;
        n += (n < 0) * M;
        return *this;
    }
    friend constexpr auto operator - (imod x, const imod& y) {
        x -= y;
        return x;
    }
    auto& operator *= (const imod& other) {
        uint64_t x = uint64_t(n) * uint64_t(other.n);
        uint64_t q = (invM * x) >> 64;
        n = x - q * M;
        n -= M * (n >= M);
        return *this;
    }
    friend constexpr auto operator * (imod x, const imod& y) {
        x *= y;
        return x;
    }
    auto& operator /= (const imod& other) {
        return operator *= (other.inv());
    }
    friend constexpr auto operator / (imod x, const imod& y) {
        x /= y;
        return x;
    }
};
