// test_spsc_ring.cpp
#include <cassert>
#include <chrono>
#include <cstdio>
#include <thread>
#include <atomic>
#include <vector>
#include <random>
#include <immintrin.h>  // for _mm_pause()
#include "spsc_ring.hpp"

// Simple expect helper
#define EXPECT_TRUE(x) do { if(!(x)){ std::fprintf(stderr,"EXPECT_TRUE failed: %s @ %s:%d\n",#x,__FILE__,__LINE__); std::abort(); } } while(0)
#define EXPECT_EQ(a,b) do { auto _va=(a); auto _vb=(b); if(!((_va)==(_vb))){ std::fprintf(stderr,"EXPECT_EQ failed: %s=%lld %s=%lld @ %s:%d\n",#a,(long long)_va,#b,(long long)_vb,__FILE__,__LINE__); std::abort(); } } while(0)

struct Desc { int id; }; // trivial payload for tests

// ============================================================================
// Test 1: basic push/pop and order
// ============================================================================
void test_basic() {
  spsc::Ring<Desc, 8> q; // capacity usable = 7
  EXPECT_TRUE(q.empty());
  EXPECT_TRUE(!q.full());

  // push a few
  EXPECT_TRUE(q.push(Desc{1}));
  EXPECT_TRUE(q.push(Desc{2}));
  EXPECT_TRUE(!q.empty());

  // pop and verify order
  Desc d;
  EXPECT_TRUE(q.pop(d)); EXPECT_EQ(d.id, 1);
  EXPECT_TRUE(q.pop(d)); EXPECT_EQ(d.id, 2);
  EXPECT_TRUE(q.empty());
}

// ============================================================================
// Test 2: full/empty boundaries
// ============================================================================
void test_full_empty() {
  spsc::Ring<Desc, 8> q; // usable 7
  // Fill to capacity-1
  for (int i=0;i<7;i++) EXPECT_TRUE(q.push(Desc{i}));
  EXPECT_TRUE(q.full());
  // Next push must fail
  EXPECT_TRUE(!q.push(Desc{999}));

  // Pop one, push one (wrap edge)
  Desc d;
  EXPECT_TRUE(q.pop(d));
  EXPECT_TRUE(q.push(Desc{777}));
  // Drain and count
  int cnt=0;
  while (q.pop(d)) ++cnt;
  EXPECT_EQ(cnt, 7); // we had 7 items total in queue at some point
  EXPECT_TRUE(q.empty());
}

// ============================================================================
// Test 3: wraparound correctness
// ============================================================================
void test_wraparound() {
  spsc::Ring<Desc, 4> q; // usable 3 → force frequent wrap
  Desc d;
  for (int round=0; round<1000; ++round) {
    for (int i=0;i<3;i++) EXPECT_TRUE(q.push(Desc{round*10+i}));
    EXPECT_TRUE(q.full());
    for (int i=0;i<3;i++) { EXPECT_TRUE(q.pop(d)); EXPECT_EQ(d.id, round*10+i); }
    EXPECT_TRUE(q.empty());
  }
}

// ============================================================================
// Test 4: stress with producer/consumer threads
// ============================================================================
void test_stress(size_t N = 1'000'000) {
  spsc::Ring<Desc, 1024> q;
  std::atomic<bool> start{false};
  std::atomic<long long> sum_produced{0};
  std::atomic<long long> sum_consumed{0};
  std::atomic<size_t> consumed{0};

  std::thread prod([&]{
    while(!start.load(std::memory_order_acquire)) { /* spin */ }
    for (int i=0; i<(int)N; ++i) {
      Desc d{i};
      while (!q.push(d)) { _mm_pause(); }
      sum_produced += i;
    }
  });

  std::thread cons([&]{
    while(!start.load(std::memory_order_acquire)) { /* spin */ }
    Desc d;
    size_t c=0;
    long long s=0;
    while (c < N) {
      if (q.pop(d)) { ++c; s += d.id; }
      else { _mm_pause(); }
    }
    consumed = c;
    sum_consumed = s;
  });

  auto t0 = std::chrono::steady_clock::now();
  start.store(true, std::memory_order_release);
  prod.join();
  cons.join();
  auto t1 = std::chrono::steady_clock::now();
  double dt = std::chrono::duration<double>(t1-t0).count();

  EXPECT_EQ(consumed.load(), N);
  EXPECT_EQ(sum_produced.load(), sum_consumed.load());

  // compute throughput (items/s)
  double mops = (double)N / dt / 1e6;
  std::printf("Stress: %zu items in %.3f s → %.2f Mops/s (push+pop)\n", N, dt, mops);
}

// ============================================================================
// Test 5: bulk ops (if you enabled them)
// ============================================================================
void test_bulk() {
  spsc::Ring<Desc, 16> q;
  std::vector<Desc> v(8);
  for (int i=0;i<8;i++) v[i].id = i;
  size_t pushed = q.push_bulk(v.data(), v.size());
  EXPECT_TRUE(pushed > 0);
  std::vector<Desc> out(8);
  size_t popped = q.pop_bulk(out.data(), out.size());
  EXPECT_EQ(popped, pushed);
  for (size_t i=0;i<popped;i++) EXPECT_EQ(out[i].id, (int)i);
}

int main() {
  std::puts("Running SPSC ring tests...");
  test_basic();
  test_full_empty();
  test_wraparound();
  test_stress();   // long-ish; lower N if needed
  test_bulk();
  std::puts("All tests PASSED.");
  return 0;
}
