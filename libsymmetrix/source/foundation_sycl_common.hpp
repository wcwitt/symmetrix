#pragma once

// Helpers shared by the SYCL foundation engines: device buffers, atomics, sub-group (warp) operations and
// kernel launches written to read like their CUDA counterparts.

#include <sycl/sycl.hpp>

#include <algorithm>
#include <climits>
#include <cstdint>
#include <cstdlib>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace symmetrix_sycl {

// ----- device memory -----

// Queue on which the engines of this process allocate; set by the first engine.
inline std::optional<sycl::queue>& process_queue()
{
    static std::optional<sycl::queue> q;
    return q;
}

// environment settings: unset or empty means the default
inline bool env_flag(const char* name, bool fallback)
{
    const char* v = std::getenv(name);
    if (not v or not *v) return fallback;
    return std::string(v) != "0";
}
inline long env_long(const char* name, long fallback, long min = LONG_MIN)
{
    const char* v = std::getenv(name);
    return std::max(min, v and *v ? std::atol(v) : fallback);
}

template <class T> struct Buffer {
    T* p = nullptr;
    size_t n = 0;
    std::optional<sycl::queue> q;
    Buffer() = default;
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
    ~Buffer() { release(); }
    void release() {
        if (p) {
            q->wait();
            sycl::free(p, *q);
        }
        p = nullptr;
        n = 0;
    }
    void reserve(size_t m) {
        if (m <= n and p) return;
        release();
        q = *process_queue();
        n = m + m/8 + 1;
        p = sycl::malloc_device<T>(n, *q);
        if (not p) throw std::runtime_error("symmetrix SYCL: out of device memory");
    }
    void upload(const std::vector<T>& v) {
        reserve(v.size());
        if (not v.empty()) q->memcpy(p, v.data(), v.size()*sizeof(T)).wait();
    }
    operator T*() const { return p; }
};

// ----- atomics -----

template <class T> inline T atomic_add(T* p, T v) {
    sycl::atomic_ref<T, sycl::memory_order::relaxed, sycl::memory_scope::device,
                     sycl::access::address_space::global_space> a(*p);
    return a.fetch_add(v);
}
template <class T> inline T atomic_add_local(T* p, T v) {
    sycl::atomic_ref<T, sycl::memory_order::relaxed, sycl::memory_scope::work_group,
                     sycl::access::address_space::generic_space> a(*p);
    return a.fetch_add(v);
}
inline unsigned atomic_or(unsigned* p, unsigned v) {
    sycl::atomic_ref<unsigned, sycl::memory_order::relaxed, sycl::memory_scope::device,
                     sycl::access::address_space::global_space> a(*p);
    return a.fetch_or(v);
}

// ----- sub-groups (CUDA "warps"; the width is set per kernel) -----

using SubGroup = sycl::sub_group;

template <class T> inline T shfl_xor(const SubGroup& sg, T v, int o) { return sycl::permute_group_by_xor(sg, v, o); }
template <class T> inline T bcast(const SubGroup& sg, T v, int lane) { return sycl::group_broadcast(sg, v, lane); }
template <class T> inline T warp_sum(const SubGroup& sg, T v) {
    return sycl::reduce_over_group(sg, v, sycl::plus<T>());
}
inline uint32_t ballot(const SubGroup& sg, bool pred) {
    uint32_t bits = 0;
    sycl::ext::oneapi::group_ballot(sg, pred).extract_bits(bits);
    return bits;
}
// lanes holding the same value as this lane (CUDA __match_any_sync), in one ballot per distinct value
inline uint32_t match_any(const SubGroup& sg, int v) {
    uint32_t pending = ballot(sg, true);
    uint32_t mine = 0;
    while (pending) {
        const int lv = bcast(sg, v, int(sycl::ctz(pending)));
        const uint32_t m = ballot(sg, v == lv);
        if (v == lv) mine = m;
        pending &= ~m;
    }
    return mine;
}
inline int popc(uint32_t x) { return int(sycl::popcount(x)); }

// ----- launches -----

// CUDA-ordered (x, y, z) dimensions as a SYCL range
inline sycl::range<3> dims(size_t x, size_t y = 1, size_t z = 1) { return sycl::range<3>(z, y, x); }
inline int blocks(long n, int per_block) { return std::max(1, int((n + per_block - 1)/per_block)); }

struct Thread {
    const sycl::nd_item<3>& it;
    int tx() const { return int(it.get_local_id(2)); }
    int bx() const { return int(it.get_group(2)); }
    int by() const { return int(it.get_group(1)); }
    int bz() const { return int(it.get_group(0)); }
    int gy() const { return int(it.get_group_range(1)); }
    int gz() const { return int(it.get_group_range(0)); }
    SubGroup sg() const { return it.get_sub_group(); }
    int lane() const { return int(it.get_sub_group().get_local_linear_id()); }
    int warp() const { return int(it.get_sub_group().get_group_linear_id()); }
    void sync() const { sycl::group_barrier(it.get_group()); }
    void syncwarp() const { sycl::group_barrier(it.get_sub_group()); }
};

}  // namespace symmetrix_sycl
