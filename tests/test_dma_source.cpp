// ============================================================================
// test_dma_source.cpp -- Unit tests for DMA Source
// ============================================================================
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <thread>
#include <vector>
#include <cstring>
#include <filesystem>

#include "slab_pool.hpp"
#include "dma_source.hpp"
#include <cnpy.h>

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
using pipeline::DmaSource;
using pipeline::DmaConfig;
namespace fs = std::filesystem;

// small test frame geometry
constexpr uint32_t TEST_W = 32;
constexpr uint32_t TEST_H = 16;
constexpr std::size_t FRAME_BYTES = std::size_t(TEST_W) * TEST_H * 2; // uint16

// ============================================================================
// Helpers to release bits like in the pipeline
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
inline void release_arch(SlabPool& pool, const Desc& d) {
  auto& b = pool.get(d.id);
  if (d.gen != b.hdr.gen) return;
  auto prev = b.hdr.pending.fetch_and(uint8_t(~SlabPool::BIT_ARCH),
                                      std::memory_order_acq_rel);
  if ((prev & ~SlabPool::BIT_ARCH) == 0) {
    ++b.hdr.gen;
    pool.release(d.id);
  }
}

// ============================================================================
// Test 1: basic unpaced streaming, nth_archive behavior, data correctness
// ============================================================================
void test_basic_unpaced() {
  using namespace std::chrono;

  const uint32_t SLABS = 16;
  SlabPool pool(SLABS, FRAME_BYTES, /*alignment*/64);

  // Build 4 synthetic frames with distinct byte patterns
  const std::size_t N_SRC = 4;
  std::vector<std::vector<std::byte>> storage;
  storage.resize(N_SRC);
  for (std::size_t i = 0; i < N_SRC; ++i) {
    storage[i].resize(FRAME_BYTES);
    std::uint8_t val = static_cast<std::uint8_t>(0x10 * (i + 1)); // 0x10, 0x20, ...
    std::memset(storage[i].data(), val, FRAME_BYTES);
  }

  std::vector<std::span<const std::byte>> frames;
  frames.reserve(N_SRC);
  for (auto& v : storage) {
    frames.emplace_back(v.data(), v.size());
  }

  // nth_archive = 3: every 3rd frame goes to archiver
  const uint32_t NTH_ARCHIVE = 3;

  std::vector<Desc> proc_descs;
  std::vector<Desc> arch_descs;
  std::vector<std::uint8_t> proc_vals;
  std::vector<std::uint8_t> arch_vals;

  // Callbacks: capture descriptors and first byte of data, then release bits
  pipeline::ProcCallback on_proc = [&](const Desc& d) {
    auto& b = pool.get(d.id);
    const std::uint8_t* p = reinterpret_cast<const std::uint8_t*>(b.data);
    std::uint8_t v = p[0];
    proc_descs.push_back(d);
    proc_vals.push_back(v);
    // release PROC bit (ARCH bit may still be set)
    release_proc(pool, d);
  };

  pipeline::ArchCallback on_arch = [&](const Desc& d) {
    auto& b = pool.get(d.id);
    const std::uint8_t* p = reinterpret_cast<const std::uint8_t*>(b.data);
    std::uint8_t v = p[0];
    arch_descs.push_back(d);
    arch_vals.push_back(v);
    // release ARCH bit (PROC bit may already be cleared)
    release_arch(pool, d);
  };

  DmaConfig cfg{};
  cfg.paced           = false;  // run as fast as possible
  cfg.frame_period_us = 0;
  cfg.expected_w      = TEST_W;
  cfg.expected_h      = TEST_H;
  cfg.nth_archive     = NTH_ARCHIVE;
  cfg.cpu_affinity    = -1;
  cfg.thread_nice     = 0;

  DmaSource dma(pool, cfg, on_proc, on_arch);
  dma.set_frames(std::move(frames));

  auto t0 = steady_clock::now();
  dma.start();
  // Let it run for a short time
  std::this_thread::sleep_for(std::chrono::milliseconds(100));
  dma.stop();
  auto t1 = steady_clock::now();

  const pipeline::DmaSource::Stats& st = dma.stats();
  const auto produced = st.produced.load(std::memory_order_relaxed);
  const auto offered  = st.archived_offered.load(std::memory_order_relaxed);
  const auto pool_ex  = st.pool_exhaust.load(std::memory_order_relaxed);

  EXPECT_TRUE(produced > 0);
  EXPECT_EQ(proc_descs.size(), produced);
  EXPECT_EQ(arch_descs.size(), offered);
  EXPECT_EQ(pool_ex, static_cast<std::uint64_t>(0)); // we released slabs, so no pool exhaust

  // Check nth_archive logic: every arch frame should have seq % NTH_ARCHIVE == 0 or 1-based depending on implementation.
  // Below we assume implementation uses (seq % nth_archive) == 0 for arch frames.
  for (const auto& d : arch_descs) {
    EXPECT_EQ(d.seq % NTH_ARCHIVE, static_cast<std::uint64_t>(0));
  }

  // Check data pattern: first byte in each slab should match src pattern based on seq % N_SRC
  for (std::size_t i = 0; i < proc_descs.size(); ++i) {
    const auto& d = proc_descs[i];
    std::uint64_t seq = d.seq;
    std::size_t src_idx = static_cast<std::size_t>(seq % N_SRC);
    std::uint8_t expected = static_cast<std::uint8_t>(0x10 * (src_idx + 1));
    EXPECT_EQ(proc_vals[i], expected);
  }

  // Also check for arch frames
  for (std::size_t i = 0; i < arch_descs.size(); ++i) {
    const auto& d = arch_descs[i];
    std::uint64_t seq = d.seq;
    std::size_t src_idx = static_cast<std::size_t>(seq % N_SRC);
    std::uint8_t expected = static_cast<std::uint8_t>(0x10 * (src_idx + 1));
    EXPECT_EQ(arch_vals[i], expected);
  }

  double dt = std::chrono::duration<double>(t1 - t0).count();
  double fps = produced / dt;
  std::printf("test_basic_unpaced: produced=%llu frames in %.3f s → %.1f fps, "
              "arch_offered=%llu\n",
              (unsigned long long)produced, dt, fps,
              (unsigned long long)offered);
  std::puts("test_basic_unpaced: OK");
}

// ============================================================================
// Test 2: pool exhaustion when consumer never releases slabs
// ============================================================================
void test_pool_exhaust() {
  const uint32_t SLABS = 4;
  SlabPool pool(SLABS, FRAME_BYTES, /*alignment*/64);

  // Single frame pattern, DMA will reuse it
  std::vector<std::byte> frame(FRAME_BYTES);
  std::memset(frame.data(), 0xAB, FRAME_BYTES);

  std::vector<std::span<const std::byte>> frames;
  frames.emplace_back(frame.data(), frame.size());

  std::vector<Desc> proc_descs;
  std::vector<Desc> arch_descs;

  // Callbacks that DO NOT release slabs; this should quickly exhaust the pool
  pipeline::ProcCallback on_proc = [&](const Desc& d) {
    proc_descs.push_back(d);
    // intentionally no release_proc()
  };
  pipeline::ArchCallback on_arch = [&](const Desc& d) {
    arch_descs.push_back(d);
    // intentionally no release_arch()
  };

  DmaConfig cfg{};
  cfg.paced           = false;
  cfg.frame_period_us = 0;
  cfg.expected_w      = TEST_W;
  cfg.expected_h      = TEST_H;
  cfg.nth_archive     = 2;   // some value
  cfg.cpu_affinity    = -1;
  cfg.thread_nice     = 0;

  DmaSource dma(pool, cfg, on_proc, on_arch);
  dma.set_frames(std::move(frames));

  dma.start();
  std::this_thread::sleep_for(std::chrono::milliseconds(50));
  dma.stop();

  const pipeline::DmaSource::Stats& st = dma.stats();
  const auto produced = st.produced.load(std::memory_order_relaxed);
  const auto pool_ex  = st.pool_exhaust.load(std::memory_order_relaxed);

  // We can't produce more frames than there are slabs (since we never release)
  EXPECT_TRUE(produced <= SLABS);
  EXPECT_TRUE(pool_ex > 0); // pool must have been exhausted at some point

  // Sanity: we saw at least one frame
  EXPECT_TRUE(!proc_descs.empty());

  std::printf("test_pool_exhaust: produced=%llu, pool_exhaust=%llu (SLABS=%u)\n",
              (unsigned long long)produced,
              (unsigned long long)pool_ex,
              SLABS);
  std::puts("test_pool_exhaust: OK");
}

// ============================================================================
// Test 3: paced mode sanity (very rough)
// ============================================================================
void test_paced_mode() {
  using namespace std::chrono;

  const uint32_t SLABS = 8;
  SlabPool pool(SLABS, FRAME_BYTES, /*alignment*/64);

  // Single frame
  std::vector<std::byte> frame(FRAME_BYTES);
  std::memset(frame.data(), 0xCD, FRAME_BYTES);

  std::vector<std::span<const std::byte>> frames;
  frames.emplace_back(frame.data(), frame.size());

  std::atomic<uint64_t> proc_count{0};

  pipeline::ProcCallback on_proc = [&](const Desc& d) {
    (void)d;
    ++proc_count;
    release_proc(pool, d);
  };
  pipeline::ArchCallback on_arch = [&](const Desc& d) {
    release_arch(pool, d);
  };

  const int frame_period_us = 5000; // target 200 fps

  DmaConfig cfg{};
  cfg.paced           = true;
  cfg.frame_period_us = frame_period_us;
  cfg.expected_w      = TEST_W;
  cfg.expected_h      = TEST_H;
  cfg.nth_archive     = 0;   // no archiving
  cfg.cpu_affinity    = -1;
  cfg.thread_nice     = 0;

  DmaSource dma(pool, cfg, on_proc, on_arch);
  dma.set_frames(std::move(frames));

  auto t0 = steady_clock::now();
  dma.start();
  std::this_thread::sleep_for(std::chrono::milliseconds(200)); // 0.2 s
  dma.stop();
  auto t1 = steady_clock::now();

  const pipeline::DmaSource::Stats& st = dma.stats();
  auto produced = st.produced.load(std::memory_order_relaxed);
  double dt = duration<double>(t1 - t0).count();
  double fps = produced / dt;

  // Very loose bounds: expect something around 200 fps
  EXPECT_TRUE(produced > 0);
  EXPECT_TRUE(fps > 50.0);    // not absurdly low
  EXPECT_TRUE(fps < 1000.0);  // not un-paced

  std::printf("test_paced_mode: produced=%llu in %.3f s → %.1f fps (target ~%.1f)\n",
              (unsigned long long)produced, dt, fps,
              1e6 / double(frame_period_us));
  std::puts("test_paced_mode: OK");
}

// ============================================================================
// Test 4: load npy frames via cnpy from a folder, mock DMA streaming
// ============================================================================
void test_npy_folder(const char* dir_path) {
  using namespace std::chrono;

  fs::path dir(dir_path);
  EXPECT_TRUE(fs::exists(dir));
  EXPECT_TRUE(fs::is_directory(dir));

  // Collect all .npy files in the directory
  std::vector<fs::path> npy_files;
  for (auto& e : fs::directory_iterator(dir)) {
    if (!e.is_regular_file()) continue;
    if (e.path().extension() == ".npy") {
      npy_files.push_back(e.path());
    }
  }
  EXPECT_TRUE(!npy_files.empty());

  std::sort(npy_files.begin(), npy_files.end());

  // Load npy files via cnpy and copy into contiguous byte storage
  std::vector<std::vector<std::byte>> storage;
  std::vector<std::uint16_t> src_first_vals;
  storage.reserve(npy_files.size());
  src_first_vals.reserve(npy_files.size());

  constexpr uint32_t TEST_W_CNPY = 300;
  constexpr uint32_t TEST_H_CNPY = 300;
  constexpr std::size_t FRAME_BYTES_CNPY = std::size_t(TEST_W_CNPY) * TEST_H_CNPY * 2; // uint16

  for (const auto& p : npy_files) {
    cnpy::NpyArray arr = cnpy::npy_load(p.string());
    EXPECT_EQ((std::size_t)2, arr.word_size);
    EXPECT_EQ((std::size_t)2, arr.shape.size());
    EXPECT_EQ((std::size_t)TEST_H_CNPY, arr.shape[0]);
    EXPECT_EQ((std::size_t)TEST_W_CNPY, arr.shape[1]);

    const std::uint16_t* src = arr.data<std::uint16_t>();
    src_first_vals.push_back(src[0]);

    std::vector<std::byte> buf(FRAME_BYTES_CNPY);
    std::memcpy(buf.data(), src, FRAME_BYTES_CNPY);
    storage.emplace_back(std::move(buf));
  }

  const std::size_t N_SRC = storage.size();

  std::vector<std::span<const std::byte>> frames;
  frames.reserve(N_SRC);
  for (auto& v : storage) {
    frames.emplace_back(v.data(), v.size());
  }

  const uint32_t SLABS = 32;
  SlabPool pool(SLABS, FRAME_BYTES_CNPY, /*alignment*/64);

  std::vector<Desc> proc_descs;
  std::vector<std::uint16_t> proc_vals;

  // For this test we don't care about archiving, just processing
  pipeline::ProcCallback on_proc = [&](const Desc& d) {
    auto& b = pool.get(d.id);
    const std::uint16_t* p = reinterpret_cast<const std::uint16_t*>(b.data);
    proc_descs.push_back(d);
    proc_vals.push_back(p[0]);
    release_proc(pool, d);
  };

  pipeline::ArchCallback on_arch = [&](const Desc& d) {
    // no-op, but still release so pool doesn't exhaust
    release_arch(pool, d);
  };

  DmaConfig cfg{};
  cfg.paced           = false;   // as fast as possible
  cfg.frame_period_us = 0;
  cfg.expected_w      = TEST_W_CNPY;
  cfg.expected_h      = TEST_H_CNPY;
  cfg.nth_archive     = 0;       // disable archiver for this test
  cfg.cpu_affinity    = -1;
  cfg.thread_nice     = 0;

  DmaSource dma(pool, cfg, on_proc, on_arch);
  dma.set_frames(std::move(frames));

  auto t0 = steady_clock::now();
  dma.start();
  std::this_thread::sleep_for(std::chrono::milliseconds(100));
  dma.stop();
  auto t1 = steady_clock::now();

  const auto& st = dma.stats();
  const auto produced = st.produced.load(std::memory_order_relaxed);
  const auto pool_ex  = st.pool_exhaust.load(std::memory_order_relaxed);

  EXPECT_TRUE(produced > 0);
  EXPECT_TRUE(proc_descs.size() == produced);
  EXPECT_EQ(pool_ex, static_cast<std::uint64_t>(0));

  // Check that first pixel of each produced frame matches corresponding npy,
  // cycling through src frames by seq % N_SRC.
  for (std::size_t i = 0; i < proc_descs.size(); ++i) {
    const auto& d = proc_descs[i];
    std::size_t src_idx = static_cast<std::size_t>(d.seq % N_SRC);
    std::uint16_t expected = src_first_vals[src_idx];
    EXPECT_EQ(proc_vals[i], expected);
  }

  double dt = std::chrono::duration<double>(t1 - t0).count();
  double fps = produced / dt;
  std::printf("test_npy_folder: dir=%s produced=%llu frames in %.3f s → %.1f fps\n",
              dir_path,
              (unsigned long long)produced, dt, fps);
  std::puts("test_npy_folder: OK");
}

// ============================================================================
// Main
// ============================================================================
int main() {
  std::puts("Running DMA source tests...");
  test_basic_unpaced();
  test_pool_exhaust();
  test_paced_mode();
  test_npy_folder("../images/test_images");
  std::puts("All DMA source tests PASSED.");
  return 0;
}
