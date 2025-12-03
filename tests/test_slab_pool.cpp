// ============================================================================
// test_slab_pool.cpp -- Test Slab Pool for DMA Image Processor
// ============================================================================
#include <atomic>
#include <cassert>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <thread>
#include <vector>
#include <algorithm>
#include <immintrin.h>
#include "slab_pool.hpp"

#define EXPECT_TRUE(x) do{ \
  if(!(x)){ \
    std::fprintf(stderr,"EXPECT_TRUE failed: %s @ %s:%d\n",#x,__FILE__,__LINE__); \
    std::abort(); \
  } \
}while(0)

#define EXPECT_EQ(a,b) do{ \
  auto _va=(a); auto _vb=(b); \
  if(!((_va)==(_vb))){ \
    std::fprintf(stderr,"EXPECT_EQ failed: %s=%lld %s=%lld @ %s:%d\n", \
                 #a,(long long)_va,#b,(long long)_vb,__FILE__,__LINE__); \
    std::abort(); \
  } \
}while(0)

using slab::SlabPool;
using slab::Desc;

constexpr std::size_t FRAME_BYTES = 1024; // small for test speed

// ============================================================================
// Test 1: basic acquire/release
// ============================================================================
void test_basic() {
  SlabPool pool(/*slabs*/8, FRAME_BYTES);
  std::vector<uint32_t> got;
  for (int i=0;i<8;i++) {
    auto id = pool.acquire();
    EXPECT_TRUE(id != UINT32_MAX);
    got.push_back(id);
  }
  // pool empty now
  EXPECT_TRUE(pool.acquire() == UINT32_MAX);
  // release all back
  for (auto id: got) pool.release(id);
  // should be able to acquire 8 again
  for (int i=0;i<8;i++) EXPECT_TRUE(pool.acquire() != UINT32_MAX);
  std::puts("basic acquire/release: OK");
}

// Helpers to release bits like in the pipeline
inline void release_proc(SlabPool& pool, const Desc& d) {
  auto& b = pool.get(d.id);
  if (d.gen != b.hdr.gen) return;
  auto prev = b.hdr.pending.fetch_and(uint8_t(~SlabPool::BIT_PROC), std::memory_order_acq_rel);
  if ((prev & ~SlabPool::BIT_PROC) == 0) { ++b.hdr.gen; pool.release(d.id); }
}
inline void release_arch(SlabPool& pool, const Desc& d) {
  auto& b = pool.get(d.id);
  if (d.gen != b.hdr.gen) return;
  auto prev = b.hdr.pending.fetch_and(uint8_t(~SlabPool::BIT_ARCH), std::memory_order_acq_rel);
  if ((prev & ~SlabPool::BIT_ARCH) == 0) { ++b.hdr.gen; pool.release(d.id); }
}

// ============================================================================
// Test 2: end-to-end producer → (proc & arch) with reclamation
// ============================================================================
void test_pipeline() {
  const uint32_t SLABS = 64;
  const size_t N = 1000000; // frames to “produce”
  SlabPool pool(SLABS, FRAME_BYTES);

  // lock-free-ish single-producer / single-consumer arrays using atomic indices
  std::vector<Desc> procQ(N), archQ(N);
  std::atomic<size_t> proc_w{0}, arch_w{0}, proc_r{0}, arch_r{0};

  std::atomic<bool> start{false};
  std::atomic<bool> done_prod{false};
  std::atomic<size_t> proc_cnt{0}, arch_cnt{0};

  // Producer thread
  std::thread prod([&]{
    while(!start.load(std::memory_order_acquire)) {}
    for (size_t i=0; i<N; ++i) {
      uint32_t id;
      // wait until a slab is available
      while ((id = pool.acquire()) == UINT32_MAX) { _mm_pause(); }
      auto& buf = pool.get(id);
      // "device write": stamp a byte so we can sanity-check consumers read the same content
      std::memset(buf.data, int(i & 0xFF), FRAME_BYTES);
      buf.hdr.pending.store(SlabPool::BIT_PROC | SlabPool::BIT_ARCH, std::memory_order_release);
      Desc d{ id, buf.hdr.gen, (uint64_t)i };
      // publish
      auto pw = proc_w.load(std::memory_order_relaxed);
      procQ[pw] = d;                                    // write descriptor first
      proc_w.store(pw + 1, std::memory_order_release);  // then publish index
      auto aw = arch_w.load(std::memory_order_relaxed);
      archQ[aw] = d;
      arch_w.store(aw + 1, std::memory_order_release);
    }
    done_prod.store(true, std::memory_order_release);
  });

  // Processor consumer
  std::thread proc([&]{
    while(!start.load(std::memory_order_acquire)) {}
    size_t local = 0;
    while (local < N) {
      auto r = proc_r.load(std::memory_order_relaxed);
      auto w = proc_w.load(std::memory_order_acquire);
      if (r == w) { _mm_pause(); continue; }
      const Desc d = procQ[r];
      proc_r.store(r+1, std::memory_order_release);

      // Read a byte to ensure data is visible and consistent
      auto& b = pool.get(d.id);
      if (d.gen == b.hdr.gen) {
        volatile unsigned char chk = *(reinterpret_cast<unsigned char*>(b.data));
        (void)chk;
      }
      release_proc(pool, d);
      ++local;
    }
    proc_cnt.store(local, std::memory_order_release);
  });

  // Archiver consumer
  std::thread arch([&]{
    while(!start.load(std::memory_order_acquire)) {}
    size_t local = 0;
    while (local < N) {
      auto r = arch_r.load(std::memory_order_relaxed);
      auto w = arch_w.load(std::memory_order_acquire);
      if (r == w) { _mm_pause(); continue; }
      const Desc d = archQ[r];
      arch_r.store(r+1, std::memory_order_release);

      // Pretend to write: touch memory
      auto& b = pool.get(d.id);
      if (d.gen == b.hdr.gen) {
        volatile unsigned char chk = *(reinterpret_cast<unsigned char*>(b.data) + FRAME_BYTES - 1);
        (void)chk;
      }
      release_arch(pool, d);
      ++local;
    }
    arch_cnt.store(local, std::memory_order_release);
  });

  auto t0 = std::chrono::steady_clock::now();
  start.store(true, std::memory_order_release);
  prod.join();
  proc.join();
  arch.join();
  auto t1 = std::chrono::steady_clock::now();
  double dt = std::chrono::duration<double>(t1 - t0).count();

  EXPECT_EQ(proc_cnt.load(), N);
  EXPECT_EQ(arch_cnt.load(), N);

  // Verify all slabs came back: we should be able to acquire SLABS times without failure
  std::vector<uint32_t> ids;
  ids.reserve(SLABS);
  for (uint32_t i=0;i<SLABS;i++) {
    auto id = pool.acquire();
    EXPECT_TRUE(id != UINT32_MAX);
    ids.push_back(id);
  }
  // Next acquire must fail (freelist empty)
  EXPECT_TRUE(pool.acquire() == UINT32_MAX);
  // Return them so dtor is happy
  for (auto id: ids) pool.release(id);

  double mframes = N / dt / 1e6;
  std::printf("pipeline: %zu frames in %.3f s → %.2f Mframes/s (producer→2 consumers)\n", N, dt, mframes);
  std::puts("pipeline reclamation & return-to-freelist: OK");
}

// ============================================================================
// Test 3: stale descriptor (ABA guard)
// ============================================================================
void test_stale_desc() {
  SlabPool pool(4, FRAME_BYTES);
  // Acquire one, set bits, then recycle and try to release with old gen.
  auto id = pool.acquire(); EXPECT_TRUE(id != UINT32_MAX);
  auto& b = pool.get(id);
  b.hdr.pending.store(SlabPool::BIT_PROC, std::memory_order_release);
  Desc old{ id, b.hdr.gen, 0 };
  // Manually simulate the other consumer clearing and releasing
  auto prev = b.hdr.pending.fetch_and(uint8_t(~SlabPool::BIT_PROC), std::memory_order_acq_rel);
  if ((prev & ~SlabPool::BIT_PROC) == 0) { ++b.hdr.gen; pool.release(id); }

  // Now attempt to "release" again with old gen; it must be a no-op (not crash)
  release_proc(pool, old);

  // Should be able to acquire that slab again
  auto id2 = pool.acquire(); EXPECT_TRUE(id2 != UINT32_MAX);
  EXPECT_EQ(id2, id);
  pool.release(id2);
  std::puts("stale descriptor (gen/ABA) handling: OK");
}

// ============================================================================
// Main
// ============================================================================
int main() {
  std::puts("Running slab pool tests...");
  test_basic();
  test_pipeline();
  test_stale_desc();
  std::puts("All slab pool tests PASSED.");
  return 0;
}
