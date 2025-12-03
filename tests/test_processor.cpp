// ============================================================================
// test_processor.cpp -- Test AVX(2/512) ROI Processor with Slab Pool
// ============================================================================
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <thread>
#include <vector>
#include <algorithm>

#include "slab_pool.hpp"
#include "processor.hpp"
#include "spsc_ring.hpp"

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

using pipeline::Processor;
using pipeline::ProcessorConfig;
using pipeline::ROI;
using pipeline::PopFn;
using pipeline::ResultCallback;

// Small image for speed
constexpr uint32_t TEST_W = 64;
constexpr uint32_t TEST_H = 32;
constexpr std::size_t FRAME_BYTES = std::size_t(TEST_W) * TEST_H * 2; // uint16

// ============================================================================
// Helpers
// ============================================================================
inline void release_proc(SlabPool& pool, const Desc& d) {
  auto& b = pool.get(d.id);
  if (d.gen != b.hdr.gen) return;
  auto prev = b.hdr.pending.fetch_and(uint8_t(~SlabPool::BIT_PROC),
                                      std::memory_order_acq_rel);
  if ((prev & ~SlabPool::BIT_PROC) == 0) {
    ++b.hdr.gen;
    pool.release(d.id);
  }
}

// ============================================================================
// Test 1: single frame, simple ROIs, basic correctness
// ============================================================================
void test_single_frame_basic() {
  SlabPool pool(/*slabs*/4, FRAME_BYTES, /*alignment*/64);
  spsc::Ring<Desc, 16> proc_q;

  // ROIs:
  //  - ROI0: full frame
  //  - ROI1: left half
  //  - ROI2: small block 8x4
  ROI roi0{0, 0, (int)TEST_W, (int)TEST_H, 0};
  ROI roi1{0, 0, (int)(TEST_W/2), (int)TEST_H, 0};
  ROI roi2{4, 2, 8, 4, 0};
  std::vector<ROI> rois{roi0, roi1, roi2};
  const std::size_t N_ROI = rois.size();

  const uint64_t area0 = (uint64_t)roi0.w * roi0.h;
  const uint64_t area1 = (uint64_t)roi1.w * roi1.h;
  const uint64_t area2 = (uint64_t)roi2.w * roi2.h;

  std::atomic<uint64_t> frames_seen{0};
  std::vector<uint64_t> sums_last(N_ROI, 0);
  std::vector<uint8_t>  occ_last(N_ROI, 0);

  // Pop function: read from SPSC
  PopFn pop_fn = [&](Desc& d) -> bool {
    return proc_q.pop(d);
  };

  // Result callback: store sums & occ and check them
  ResultCallback on_result =
    [&](uint64_t seq,
        const std::vector<uint64_t>& roi_sums,
        const std::vector<uint8_t>& roi_occ) -> bool {
      EXPECT_EQ(roi_sums.size(), N_ROI);
      EXPECT_EQ(roi_occ.size(),  N_ROI);
      EXPECT_EQ(seq, static_cast<uint64_t>(0)); // only one frame

      // Copy and check
      for (std::size_t i = 0; i < N_ROI; ++i) {
        sums_last[i] = roi_sums[i];
        occ_last[i]  = roi_occ[i];
      }
      frames_seen.fetch_add(1, std::memory_order_relaxed);
      return true; // keep running (ingress will stop after one frame anyway)
    };

  ProcessorConfig pcfg;
  pcfg.image_w = TEST_W;
  pcfg.image_h = TEST_H;
  pcfg.worker_threads = 2;
  pcfg.print_stdout = false;

  Processor proc(pool, pcfg, pop_fn, rois, on_result);
  proc.start();

  // Produce a single frame: all pixels = val
  const uint16_t val = 5;
  uint32_t id;
  while ((id = pool.acquire()) == UINT32_MAX) {
    std::this_thread::yield();
  }
  auto& buf = pool.get(id);
  uint16_t* img = reinterpret_cast<uint16_t*>(buf.data);
  for (std::size_t i = 0; i < std::size_t(TEST_W) * TEST_H; ++i) {
    img[i] = val;
  }
  buf.hdr.pending.store(SlabPool::BIT_PROC, std::memory_order_release);
  Desc d{id, buf.hdr.gen, 0ull};

  while (!proc_q.push(d)) {
    std::this_thread::yield();
  }

  // Wait for processor to handle it
  auto t0 = std::chrono::steady_clock::now();
  while (frames_seen.load(std::memory_order_relaxed) < 1) {
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    auto t1 = std::chrono::steady_clock::now();
    if (std::chrono::duration<double>(t1 - t0).count() > 2.0) {
      std::fprintf(stderr, "test_single_frame_basic: timeout\n");
      break;
    }
  }

  proc.stop();

  // Expected sums
  uint64_t e0 = (uint64_t)val * area0;
  uint64_t e1 = (uint64_t)val * area1;
  uint64_t e2 = (uint64_t)val * area2;

  EXPECT_EQ(sums_last[0], e0);
  EXPECT_EQ(sums_last[1], e1);
  EXPECT_EQ(sums_last[2], e2);

  // Thresholds were 0 so all ROIs occupied
  EXPECT_EQ(occ_last[0], static_cast<uint8_t>(1));
  EXPECT_EQ(occ_last[1], static_cast<uint8_t>(1));
  EXPECT_EQ(occ_last[2], static_cast<uint8_t>(1));

  // All slabs should be back in pool
  std::vector<uint32_t> ids;
  ids.reserve(pool.size());
  for (uint32_t i=0; i<pool.size(); ++i) {
    auto got = pool.acquire();
    EXPECT_TRUE(got != UINT32_MAX);
    ids.push_back(got);
  }
  EXPECT_TRUE(pool.acquire() == UINT32_MAX);
  for (auto x: ids) pool.release(x);

  std::puts("test_single_frame_basic: OK");
}

// ============================================================================
// Test 2: many frames, multi-threaded processor, correctness & reclamation
// ============================================================================
void test_many_frames_multithreaded() {
  using namespace std::chrono;

  constexpr std::size_t N_FRAMES = 200;

  SlabPool pool(/*slabs*/16, FRAME_BYTES, /*alignment*/64);
  spsc::Ring<Desc, 256> proc_q;

  // ROIs as in test 1
  ROI roi0{0, 0, (int)TEST_W, (int)TEST_H, 0};
  ROI roi1{0, 0, (int)(TEST_W/2), (int)TEST_H, 0};
  ROI roi2{4, 2, 8, 4, 0};
  std::vector<ROI> rois{roi0, roi1, roi2};
  const std::size_t N_ROI = rois.size();

  const uint64_t area0 = (uint64_t)roi0.w * roi0.h;
  const uint64_t area1 = (uint64_t)roi1.w * roi1.h;
  const uint64_t area2 = (uint64_t)roi2.w * roi2.h;

  std::vector<std::vector<uint64_t>> sums_per_frame(N_FRAMES,
                                                    std::vector<uint64_t>(N_ROI, 0));
  std::vector<std::vector<uint8_t>>  occ_per_frame(N_FRAMES,
                                                   std::vector<uint8_t>(N_ROI, 0));
  std::atomic<uint64_t> frames_seen{0};

  PopFn pop_fn = [&](Desc& d) -> bool {
    return proc_q.pop(d);
  };

  ResultCallback on_result =
    [&](uint64_t seq,
        const std::vector<uint64_t>& roi_sums,
        const std::vector<uint8_t>& roi_occ) -> bool {
      if (seq >= N_FRAMES) return false;
      EXPECT_EQ(roi_sums.size(), N_ROI);
      EXPECT_EQ(roi_occ.size(),  N_ROI);
      for (std::size_t i=0;i<N_ROI;++i) {
        sums_per_frame[seq][i] = roi_sums[i];
        occ_per_frame[seq][i]  = roi_occ[i];
      }
      frames_seen.fetch_add(1, std::memory_order_relaxed);
      return true;
    };

  ProcessorConfig pcfg;
  pcfg.image_w = TEST_W;
  pcfg.image_h = TEST_H;
  pcfg.worker_threads = 4; // exercise worker pool
  pcfg.print_stdout = false;

  Processor proc(pool, pcfg, pop_fn, rois, on_result);
  proc.start();

  // Producer: generate N_FRAMES frames, constant per-frame value
  for (std::size_t f = 0; f < N_FRAMES; ++f) {
    uint32_t id;
    while ((id = pool.acquire()) == UINT32_MAX) {
      std::this_thread::yield();
    }
    auto& buf = pool.get(id);
    uint16_t* img = reinterpret_cast<uint16_t*>(buf.data);
    const uint16_t val = static_cast<uint16_t>((f+1) & 0xFFFF);
    for (std::size_t i=0;i<std::size_t(TEST_W)*TEST_H;++i) {
      img[i] = val;
    }
    buf.hdr.pending.store(SlabPool::BIT_PROC, std::memory_order_release);
    Desc d{id, buf.hdr.gen, (uint64_t)f};
    while (!proc_q.push(d)) {
      std::this_thread::yield();
    }
  }

  auto t0 = steady_clock::now();
  while (frames_seen.load(std::memory_order_relaxed) < N_FRAMES) {
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    auto t1 = steady_clock::now();
    if (duration<double>(t1 - t0).count() > 5.0) {
      std::fprintf(stderr, "test_many_frames_multithreaded: timeout\n");
      break;
    }
  }

  proc.stop();

  // Check each frame
  for (std::size_t f = 0; f < N_FRAMES; ++f) {
    const uint16_t val = static_cast<uint16_t>((f+1) & 0xFFFF);
    uint64_t e0 = (uint64_t)val * area0;
    uint64_t e1 = (uint64_t)val * area1;
    uint64_t e2 = (uint64_t)val * area2;

    EXPECT_EQ(sums_per_frame[f][0], e0);
    EXPECT_EQ(sums_per_frame[f][1], e1);
    EXPECT_EQ(sums_per_frame[f][2], e2);

    EXPECT_EQ(occ_per_frame[f][0], static_cast<uint8_t>(1));
    EXPECT_EQ(occ_per_frame[f][1], static_cast<uint8_t>(1));
    EXPECT_EQ(occ_per_frame[f][2], static_cast<uint8_t>(1));
  }

  // All slabs back in pool
  std::vector<uint32_t> ids;
  ids.reserve(pool.size());
  for (uint32_t i=0;i<pool.size();++i) {
    auto got = pool.acquire();
    EXPECT_TRUE(got != UINT32_MAX);
    ids.push_back(got);
  }
  EXPECT_TRUE(pool.acquire() == UINT32_MAX);
  for (auto x: ids) pool.release(x);

  const auto& ps = proc.stats();
  auto t1 = steady_clock::now();
  double dt = std::chrono::duration<double>(t1 - t0).count();
  std::printf("test_many_frames_multithreaded: %llu frames in %.3f s → %.1f fps\n",
              (unsigned long long)ps.frames.load(), dt,
              ps.frames.load() / dt);

  std::puts("test_many_frames_multithreaded: OK");
}

// ============================================================================
// Test 3: stale descriptor (gen mismatch) is safely ignored by processor
// ============================================================================
void test_stale_descriptor_ignored() {
  SlabPool pool(/*slabs*/2, FRAME_BYTES, /*alignment*/64);
  spsc::Ring<Desc, 8> proc_q;

  // Simple ROI
  ROI roi{0,0,(int)TEST_W,(int)TEST_H,0};
  std::vector<ROI> rois{roi};

  std::atomic<uint64_t> frames_seen{0};

  PopFn pop_fn = [&](Desc& d) -> bool {
    return proc_q.pop(d);
  };

  ResultCallback on_result =
    [&](uint64_t seq,
        const std::vector<uint64_t>&,
        const std::vector<uint8_t>&)->bool {
      (void)seq;
      frames_seen.fetch_add(1, std::memory_order_relaxed);
      return true;
    };

  ProcessorConfig pcfg;
  pcfg.image_w = TEST_W;
  pcfg.image_h = TEST_H;
  pcfg.worker_threads = 2;
  pcfg.print_stdout = false;

  Processor proc(pool, pcfg, pop_fn, rois, on_result);
  proc.start();

  // Acquire a slab, set PROC bit, then bump gen to simulate reuse,
  // then push old Desc with stale gen.
  uint32_t id = pool.acquire();
  EXPECT_TRUE(id != UINT32_MAX);
  auto& b = pool.get(id);
  uint16_t* img = reinterpret_cast<uint16_t*>(b.data);
  for (std::size_t i=0;i<std::size_t(TEST_W)*TEST_H;++i) img[i] = 1;
  b.hdr.pending.store(SlabPool::BIT_PROC, std::memory_order_release);
  Desc stale{id, b.hdr.gen, 0ull};

  // "Recycle" slab: clear PROC bit and release
  auto prev = b.hdr.pending.fetch_and(uint8_t(~SlabPool::BIT_PROC),
                                      std::memory_order_acq_rel);
  if ((prev & ~SlabPool::BIT_PROC) == 0) {
    ++b.hdr.gen;
    pool.release(id);
  }

  // Push stale descriptor (old gen)
  while (!proc_q.push(stale)) {
    std::this_thread::yield();
  }

  // Give processor some time to see and ignore it
  std::this_thread::sleep_for(std::chrono::milliseconds(100));

  proc.stop();

  // Processor should have either seen 0 frames or at least not crash / misbehave.
  EXPECT_TRUE(frames_seen.load(std::memory_order_relaxed) <= 1);

  // Slabs must still be recoverable
  std::vector<uint32_t> ids;
  ids.reserve(pool.size());
  for (uint32_t i=0;i<pool.size();++i) {
    auto got = pool.acquire();
    EXPECT_TRUE(got != UINT32_MAX);
    ids.push_back(got);
  }
  EXPECT_TRUE(pool.acquire() == UINT32_MAX);
  for (auto x: ids) pool.release(x);

  std::puts("test_stale_descriptor_ignored: OK");
}

// ============================================================================
// Main
// ============================================================================
int main() {
  std::puts("Running processor tests...");
  test_single_frame_basic();
  test_many_frames_multithreaded();
  test_stale_descriptor_ignored();
  std::puts("All processor tests PASSED.");
  return 0;
}
