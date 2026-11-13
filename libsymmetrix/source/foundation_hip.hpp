#pragma once

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#include <algorithm>
#include <cstdint>

namespace symmetrix_hip {

constexpr int waves_per_simd(int threads, int blocks, int lds_bytes = 0)
{
    const int waves = (threads + 63)/64;
    const int most = std::min(8, (lds_bytes > 0 ? 65536/lds_bytes : 64)*waves/4);
    return std::max(1, std::min(std::max(2, (blocks*waves + 3)/4), most));
}

template <class T> __device__ __forceinline__ T shfl(T v, int src) { return __shfl(v, src, 32); }
template <class T> __device__ __forceinline__ T shfl_xor(T v, int mask) { return __shfl_xor(v, mask, 32); }

__device__ __forceinline__ unsigned ballot32(bool p) {
    return unsigned(__ballot(p) >> (__lane_id() & 32));
}

__device__ __forceinline__ unsigned match_any32(int v) {
    unsigned todo = ballot32(true);
    unsigned mine = 0;
    while (todo) {
        const int leader = __ffs(todo) - 1;
        const int w = __shfl(v, leader, 32);
        const unsigned m = ballot32(v == w) & todo;
        if (v == w) mine = m;
        todo &= ~m;
    }
    return mine;
}

typedef _Float16 half2_native __attribute__((ext_vector_type(2)));
__device__ __forceinline__ float2 load_half2(const void* p) {
    const half2_native v = *reinterpret_cast<const half2_native*>(p);
    return make_float2(float(v.x), float(v.y));
}

}  // namespace symmetrix_hip

#define SYMHIP_LAUNCH_BOUNDS(threads, ...) \
    __launch_bounds__((threads), symmetrix_hip::waves_per_simd((threads), __VA_ARGS__))
