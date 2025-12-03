// ============================================================================
// slab_pool.hpp -- Slab Pool for DMA Image Processor
//
// This header defines the slab::SlabPool and related types supporting 
// efficient, lock-free memory pools with multiple consumers for high-throughput
// frame pipelines. It provides buffer headers, descriptor types, and a 
// lock-free freelist for managing shared frame buffers in DMA processing 
// pipelines.
// ============================================================================
#pragma once
#include <atomic>
#include <cstdint>
#include <cstddef>
#include <vector>
#include <memory>
#include <stdexcept>
#include <cstring>
#include <cassert>

#if __cplusplus < 201703L
namespace std {
  enum class byte : unsigned char {};
}
#endif

namespace slab {

// ============================================================================
// BufHdr: slab header - consumer bitmask + generation
// ============================================================================
struct alignas(64) BufHdr {
  std::atomic<uint8_t> pending  { 0 };   // bitmask of consumers
  uint8_t gen                   { 0 };   // generation byte (ABA guard)
  uint8_t _pad[62]              { 0 };   // padding to 64 bytes
};

// ============================================================================
// Buffer: one slab = header + pointer to aligned pixel storage.
// ============================================================================
struct Buffer {
  BufHdr hdr;
  std::byte* data { nullptr }; // aligned FRAME_BYTES
};

// ============================================================================
// Desc: tiny descriptor you pass through SPSC rings.
// ============================================================================
struct Desc {
  uint32_t id;     // slab index
  uint8_t  gen;    // generation snapshot
  uint64_t seq;    // frame number (optional, for ordering/telemetry)
};

// ============================================================================
// Freelist: lock-free freelist (MPMC) with tagged head
// ============================================================================
class Freelist {
public:
  /// Constructor for the Freelist class.
  /// @param n The number of elements in the freelist.
  explicit Freelist(uint32_t n) : next_(n), head_(pack(0, 0)) {
    for (uint32_t i = 0; i < n; ++i) {
        // next index+1; 0 means null
        uint32_t next1 = (i + 1 < n) ? (i + 2) : 0;       
        next_[i].store(next1, std::memory_order_relaxed);
      }
      // head = first (1), or 0 if empty
      head_.store(pack(0, n ? 1u : 0u), std::memory_order_relaxed);
  }

  /// Pop: returns UINT32_MAX if empty
  /// @return The index of the popped element, or UINT32_MAX if the freelist is empty.
  uint32_t pop() noexcept {
    for (;;) {
      uint64_t h = head_.load(std::memory_order_acquire);
      uint32_t idx1 = unpack_idx(h);
      if (idx1 == 0) return UINT32_MAX;            // empty
      uint32_t idx = idx1 - 1;                     // real index
      uint32_t next1 = next_[idx].load(std::memory_order_relaxed);
      uint64_t newh = pack(unpack_tag(h) + 1, next1);
      if (head_.compare_exchange_weak(h, newh, std::memory_order_acq_rel, 
                                      std::memory_order_acquire))
        return idx;
    }
  }

  /// Push index back
  /// @param idx The index to push back.
  void push(uint32_t idx) noexcept {
    uint32_t idx1 = idx + 1;
    for (;;) {
      uint64_t h = head_.load(std::memory_order_acquire);
      next_[idx].store(unpack_idx(h), std::memory_order_relaxed);
      uint64_t newh = pack(unpack_tag(h) + 1, idx1);
      if (head_.compare_exchange_weak(h, newh, std::memory_order_acq_rel, 
          std::memory_order_acquire))
        return;
    }
  }

  /// Approximate availability: walk once (debug use). O(N); avoid on hot path.
  /// @return The approximate size of the freelist.
  uint32_t approx_size() const noexcept {
    uint32_t count = 0;
    uint32_t i1 = unpack_idx(head_.load(std::memory_order_acquire));
    while (i1) { ++count; i1 = next_[i1 - 1].load(std::memory_order_relaxed); }
    return count;
  }

private:
  static inline uint64_t pack(uint32_t tag, uint32_t idx1) noexcept {
    return (uint64_t(tag) << 32) | uint64_t(idx1);
  }
  static inline uint32_t unpack_idx(uint64_t v) noexcept { 
    return uint32_t(v & 0xffffffffu); }
  static inline uint32_t unpack_tag(uint64_t v) noexcept { 
    return uint32_t(v >> 32); }

  std::vector<std::atomic<uint32_t>> next_; // next_[i] holds next index+1
  std::atomic<uint64_t> head_;              // tagged head
};

// ============================================================================
// SlabPool
// ============================================================================
class SlabPool {
public:
  /// bits you can use in BufHdr::pending (expand as needed)
  static constexpr uint8_t BIT_PROC = 0x01;
  static constexpr uint8_t BIT_ARCH = 0x02;

  /// Constructor for the SlabPool class.
  /// @param slab_count The number of slabs in the pool.
  /// @param frame_bytes The number of bytes per frame.
  /// @param alignment The alignment of the frames.
  SlabPool(uint32_t slab_count, std::size_t frame_bytes, 
           std::size_t alignment = 64)
  : N_(slab_count), frame_bytes_(frame_bytes), alignment_(alignment), 
    bufs_(slab_count), free_(slab_count) {
    if (N_ == 0 || frame_bytes_ == 0) {
      throw std::invalid_argument("invalid pool sizes");
    }
    /// Allocate aligned pixel storage per slab
    for (uint32_t i = 0; i < N_; ++i) {
      bufs_[i].hdr.pending.store(0, std::memory_order_relaxed);
      bufs_[i].hdr.gen = 0;
      bufs_[i].data = allocate_aligned(frame_bytes_, alignment_);
      if (!bufs_[i].data) throw std::bad_alloc();
    }
  }

  ~SlabPool() {
    for (uint32_t i = 0; i < N_; ++i) {
      if (bufs_[i].data) {
        deallocate_aligned(bufs_[i].data, frame_bytes_, alignment_);
      }
      bufs_[i].data = nullptr;
    }
  }

  SlabPool(const SlabPool&) = delete;
  SlabPool& operator=(const SlabPool&) = delete;

  uint32_t size() const noexcept { return N_; }

  /// Acquire a free slab index; returns UINT32_MAX if pool exhausted.
  /// @return The index of the acquired slab, or UINT32_MAX if the pool is 
  ///         exhausted.
  uint32_t acquire() noexcept {
    return free_.pop();
  }

  /// Return slab index to freelist.
  /// @param id The index of the slab to release.
  void release(uint32_t id) noexcept {
    assert(id < N_);
    free_.push(id);
  }

  Buffer&       get(uint32_t id)       noexcept { return bufs_[id]; }
  const Buffer& get(uint32_t id) const noexcept { return bufs_[id]; }
  std::size_t   frame_bytes() const noexcept { return frame_bytes_; }

private:
  static std::byte* allocate_aligned(std::size_t n, std::size_t align) {
#if (__cpp_aligned_new >= 201606L)
    return static_cast<std::byte*>(::operator new (n, std::align_val_t(align), 
                                                   std::nothrow));
#else
    void* p = nullptr;
    if (posix_memalign(&p, align, n) != 0) p = nullptr;
    return static_cast<std::byte*>(p);
#endif
  }
  static void deallocate_aligned(void* p, std::size_t, std::size_t align) {
#if (__cpp_aligned_new >= 201606L)
    ::operator delete(p, std::align_val_t(align));
#else
    free(p);
#endif
  }

  const uint32_t N_;
  const std::size_t frame_bytes_;
  const std::size_t alignment_;
  std::vector<Buffer> bufs_;
  Freelist free_;
};

} // namespace slab
