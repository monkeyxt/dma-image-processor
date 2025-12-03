// ============================================================================
// test_archiver.cpp -- Test Archiver for DMA Image Processor
// ============================================================================
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <thread>
#include <vector>
#include <fstream>
#include <filesystem>

#include "slab_pool.hpp"
#include "archiver.hpp"
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
using pipeline::Archiver;
using pipeline::ArchiverConfig;
using pipeline::PopFn;

namespace fs = std::filesystem;

constexpr uint32_t TEST_W      = 32;
constexpr uint32_t TEST_H      = 16;
constexpr std::size_t FRAME_BYTES = std::size_t(TEST_W) * TEST_H * 2; // uint16

// Simple SPSC alias
template<typename T, std::size_t N>
using Ring = spsc::Ring<T, N>;

// Helper to clear output dir
inline void reset_dir(const std::string& dir) {
  std::error_code ec;
  fs::remove_all(dir, ec);
  fs::create_directories(dir, ec);
}

// ============================================================================
// Test 1: basic single segment write (no rotation), index correctness
// ============================================================================
void test_single_segment_basic() {
  const std::string outdir = "archiver_test_basic";
  reset_dir(outdir);

  SlabPool pool(/*slabs*/8, FRAME_BYTES, /*alignment*/64);
  Ring<Desc, 64> arch_q;

  // Pop function for archiver
  PopFn pop_fn = [&](Desc& d) -> bool {
    return arch_q.pop(d);
  };

  ArchiverConfig cfg;
  cfg.output_dir      = outdir;
  cfg.file_prefix     = "frames";
  cfg.segment_bytes   = FRAME_BYTES * 4; // 4 frames per segment; we'll only write 4
  cfg.io_buffer_bytes = 1024 * 1024;
  cfg.make_index      = true;
  cfg.use_io_uring    = false;          // force stdio path for portability

  Archiver arch(pool, cfg, pop_fn);
  arch.start();

  const std::size_t N_FRAMES = 4;

  // Produce 4 frames, each with a simple byte pattern
  for (std::size_t f = 0; f < N_FRAMES; ++f) {
    uint32_t id;
    while ((id = pool.acquire()) == UINT32_MAX) {
      std::this_thread::yield();
    }
    auto& b = pool.get(id);
    std::uint8_t* bytes = reinterpret_cast<std::uint8_t*>(b.data);
    std::memset(bytes, (int)(f & 0xFF), FRAME_BYTES);

    b.hdr.pending.store(SlabPool::BIT_ARCH, std::memory_order_release);
    Desc d{id, b.hdr.gen, (uint64_t)f};

    while (!arch_q.push(d)) {
      std::this_thread::yield();
    }
  }

  // Wait for archiver to consume all frames
  auto t0 = std::chrono::steady_clock::now();
  while (arch.stats().archived_frames.load(std::memory_order_relaxed) < N_FRAMES) {
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    auto t1 = std::chrono::steady_clock::now();
    if (std::chrono::duration<double>(t1 - t0).count() > 5.0) {
      std::fprintf(stderr, "test_single_segment_basic: timeout\n");
      break;
    }
  }

  arch.stop();

  // Check stats
  const auto& st = arch.stats();
  EXPECT_EQ(st.archived_frames.load(), N_FRAMES);
  EXPECT_EQ(st.archived_bytes.load(), N_FRAMES * FRAME_BYTES);

  // Verify data file exists: frames_000000.bin
  fs::path bin_path = fs::path(outdir) / "frames_000000.bin";
  EXPECT_TRUE(fs::exists(bin_path));

  std::ifstream bin(bin_path, std::ios::binary);
  EXPECT_TRUE(bin.good());
  bin.seekg(0, std::ios::end);
  std::streamsize sz = bin.tellg();
  EXPECT_EQ((std::size_t)sz, N_FRAMES * FRAME_BYTES);
  bin.seekg(0, std::ios::beg);

  std::vector<std::uint8_t> buf((std::size_t)sz);
  bin.read(reinterpret_cast<char*>(buf.data()), sz);
  EXPECT_TRUE(bin.good());

  // Each frame was filled with byte = frame index
  for (std::size_t f = 0; f < N_FRAMES; ++f) {
    std::uint8_t expected = (std::uint8_t)(f & 0xFF);
    for (std::size_t i = 0; i < FRAME_BYTES; ++i) {
      EXPECT_EQ(buf[f * FRAME_BYTES + i], expected);
    }
  }

  // Verify index file
  fs::path idx_path = fs::path(outdir) / "frames_000000.idx";
  EXPECT_TRUE(fs::exists(idx_path));
  std::ifstream idx(idx_path);
  EXPECT_TRUE(idx.good());

  std::size_t lines = 0;
  while (idx.good()) {
    unsigned long long seq=0, off=0;
    idx >> seq >> off;
    if (!idx.good()) break;
    EXPECT_EQ(seq, (unsigned long long)lines);
    EXPECT_EQ(off, (unsigned long long)(lines * FRAME_BYTES));
    ++lines;
  }
  EXPECT_EQ(lines, N_FRAMES);

  // All slabs should have been returned to pool
  std::vector<uint32_t> ids;
  ids.reserve(pool.size());
  for (uint32_t i = 0; i < pool.size(); ++i) {
    auto id = pool.acquire();
    EXPECT_TRUE(id != UINT32_MAX);
    ids.push_back(id);
  }
  EXPECT_TRUE(pool.acquire() == UINT32_MAX);
  for (auto id : ids) pool.release(id);

  std::puts("test_single_segment_basic: OK");
}

// ============================================================================
// Test 2: rotation over multiple segments, verify sizes & rotations counter
// ============================================================================
void test_rotation_segments() {
  const std::string outdir = "archiver_test_rotate";
  reset_dir(outdir);

  SlabPool pool(/*slabs*/8, FRAME_BYTES, /*alignment*/64);
  Ring<Desc, 128> arch_q;

  PopFn pop_fn = [&](Desc& d) -> bool {
    return arch_q.pop(d);
  };

  ArchiverConfig cfg;
  cfg.output_dir      = outdir;
  cfg.file_prefix     = "frames";
  cfg.segment_bytes   = FRAME_BYTES * 3; // 3 frames per segment
  cfg.io_buffer_bytes = 512 * 1024;
  cfg.make_index      = true;
  cfg.use_io_uring    = false;

  Archiver arch(pool, cfg, pop_fn);
  arch.start();

  const std::size_t N_FRAMES = 10;

  // Produce 10 frames with distinct patterns
  for (std::size_t f = 0; f < N_FRAMES; ++f) {
    uint32_t id;
    while ((id = pool.acquire()) == UINT32_MAX) {
      std::this_thread::yield();
    }
    auto& b = pool.get(id);
    std::uint8_t* bytes = reinterpret_cast<std::uint8_t*>(b.data);
    std::memset(bytes, (int)((f + 1) & 0xFF), FRAME_BYTES);

    b.hdr.pending.store(SlabPool::BIT_ARCH, std::memory_order_release);
    Desc d{id, b.hdr.gen, (uint64_t)f};

    while (!arch_q.push(d)) {
      std::this_thread::yield();
    }
  }

  // Wait for archiver
  auto t0 = std::chrono::steady_clock::now();
  while (arch.stats().archived_frames.load(std::memory_order_relaxed) < N_FRAMES) {
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    auto t1 = std::chrono::steady_clock::now();
    if (std::chrono::duration<double>(t1 - t0).count() > 5.0) {
      std::fprintf(stderr, "test_rotation_segments: timeout\n");
      break;
    }
  }

  arch.stop();

  const auto& st = arch.stats();
  EXPECT_EQ(st.archived_frames.load(), N_FRAMES);
  EXPECT_EQ(st.archived_bytes.load(), N_FRAMES * FRAME_BYTES);

  // With segment_bytes = 3 * FRAME_BYTES and 10 frames, we expect
  // ceil(10 / 3) = 4 segments: 3+3+3+1 frames.
  const std::size_t EXPECT_SEGMENTS = 4;
  EXPECT_EQ(st.rotations.load(), EXPECT_SEGMENTS);

  // Verify each segment file size and contents
  // Segment mapping:
  //   seg0: frames 0,1,2
  //   seg1: frames 3,4,5
  //   seg2: frames 6,7,8
  //   seg3: frame  9
  std::vector<std::size_t> frames_in_seg {3,3,3,1};
  std::size_t global_frame = 0;

  for (std::size_t seg = 0; seg < EXPECT_SEGMENTS; ++seg) {
    char name[64];
    std::snprintf(name, sizeof(name), "frames_%06zu.bin", seg);
    fs::path bin_path = fs::path(outdir) / name;
    EXPECT_TRUE(fs::exists(bin_path));

    std::ifstream bin(bin_path, std::ios::binary);
    EXPECT_TRUE(bin.good());
    bin.seekg(0, std::ios::end);
    std::streamsize sz = bin.tellg();
    EXPECT_EQ((std::size_t)sz, frames_in_seg[seg] * FRAME_BYTES);
    bin.seekg(0, std::ios::beg);

    std::vector<std::uint8_t> buf((std::size_t)sz);
    bin.read(reinterpret_cast<char*>(buf.data()), sz);
    EXPECT_TRUE(bin.good());

    for (std::size_t f = 0; f < frames_in_seg[seg]; ++f) {
      std::uint8_t expected = (std::uint8_t)((global_frame + 1) & 0xFF);
      for (std::size_t i = 0; i < FRAME_BYTES; ++i) {
        EXPECT_EQ(buf[f * FRAME_BYTES + i], expected);
      }
      ++global_frame;
    }

    // Index file: offsets 0, FB, 2FB per segment, seq = global index
    std::snprintf(name, sizeof(name), "frames_%06zu.idx", seg);
    fs::path idx_path = fs::path(outdir) / name;
    EXPECT_TRUE(fs::exists(idx_path));
    std::ifstream idx(idx_path);
    EXPECT_TRUE(idx.good());

    std::size_t lines = 0;
    while (idx.good()) {
      unsigned long long seq=0, off=0;
      idx >> seq >> off;
      if (!idx.good()) break;

      std::size_t seq_global = (seg == 0) ? lines
                             : (seg == 1) ? (3 + lines)
                             : (seg == 2) ? (6 + lines)
                             : (9 + lines); // seg3 only 1 frame

      EXPECT_EQ(seq, (unsigned long long)seq_global);
      EXPECT_EQ(off, (unsigned long long)(lines * FRAME_BYTES));
      ++lines;
    }
    EXPECT_EQ(lines, frames_in_seg[seg]);
  }

  // Slabs back in pool
  std::vector<uint32_t> ids;
  ids.reserve(pool.size());
  for (uint32_t i = 0; i < pool.size(); ++i) {
    auto id = pool.acquire();
    EXPECT_TRUE(id != UINT32_MAX);
    ids.push_back(id);
  }
  EXPECT_TRUE(pool.acquire() == UINT32_MAX);
  for (auto id : ids) pool.release(id);

  std::puts("test_rotation_segments: OK");
}

// ============================================================================
// Test 3: stale descriptor (gen mismatch) is safely ignored by archiver
// ============================================================================
void test_stale_descriptor_ignored() {
  const std::string outdir = "archiver_test_stale";
  reset_dir(outdir);

  SlabPool pool(/*slabs*/2, FRAME_BYTES, /*alignment*/64);
  Ring<Desc, 8> arch_q;

  PopFn pop_fn = [&](Desc& d) -> bool {
    return arch_q.pop(d);
  };

  ArchiverConfig cfg;
  cfg.output_dir      = outdir;
  cfg.file_prefix     = "frames";
  cfg.segment_bytes   = FRAME_BYTES * 4;
  cfg.io_buffer_bytes = 256 * 1024;
  cfg.make_index      = false;
  cfg.use_io_uring    = false;

  Archiver arch(pool, cfg, pop_fn);
  arch.start();

  // Acquire a slab, set ARCH bit, then clear & release, bump gen,
  // and only then push old descriptor (stale gen) into queue.
  uint32_t id = pool.acquire();
  EXPECT_TRUE(id != UINT32_MAX);

  auto& b = pool.get(id);
  std::uint8_t* bytes = reinterpret_cast<std::uint8_t*>(b.data);
  std::memset(bytes, 0xAB, FRAME_BYTES);

  b.hdr.pending.store(SlabPool::BIT_ARCH, std::memory_order_release);
  Desc stale{id, b.hdr.gen, 0ull};

  // Clear ARCH and release (simulate normal consumer)
  auto prev = b.hdr.pending.fetch_and(uint8_t(~SlabPool::BIT_ARCH),
                                      std::memory_order_acq_rel);
  if ((prev & ~SlabPool::BIT_ARCH) == 0) {
    ++b.hdr.gen;
    pool.release(id);
  }

  // Now push stale descriptor with old gen
  while (!arch_q.push(stale)) {
    std::this_thread::yield();
  }

  // Give archiver time to pop and ignore it
  std::this_thread::sleep_for(std::chrono::milliseconds(200));
  arch.stop();

  // Should have recorded this as stale_desc >= 1, and not crashed
  const auto& st = arch.stats();
  EXPECT_TRUE(st.stale_desc.load(std::memory_order_relaxed) >= 1);

  // Slabs still recoverable
  std::vector<uint32_t> ids;
  ids.reserve(pool.size());
  for (uint32_t i = 0; i < pool.size(); ++i) {
    auto got = pool.acquire();
    EXPECT_TRUE(got != UINT32_MAX);
    ids.push_back(got);
  }
  EXPECT_TRUE(pool.acquire() == UINT32_MAX);
  for (auto x : ids) pool.release(x);

  std::puts("test_stale_descriptor_ignored: OK");
}

// ============================================================================
// Main
// ============================================================================
int main() {
  std::puts("Running archiver tests...");
  test_single_segment_basic();
  test_rotation_segments();
  test_stale_descriptor_ignored();
  std::puts("All archiver tests PASSED.");
  return 0;
}
