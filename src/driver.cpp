// ============================================================================
// `drive.cpp` -- End-to-end driver for DMA → Processor → Archiver
//
// Usage:
//   ./dma-image-processor <npy_dir> [run_seconds]
//
//  - Loads all *.npy frames in <npy_dir> (row-major H×W, uint16).
//  - Sets up slab pool, SPSC queues, DMA source, processor, and archiver.
//  - Uses PoissonDetector to set an occupancy threshold for all ROIs.
//  - Runs for run_seconds (default 5 s) and prints stats.
// ============================================================================
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>
#include <thread>
#include <vector>

#include <cnpy.hpp>

#include "slab_pool.hpp"
#include "spsc_ring.hpp"
#include "dma_source.hpp"
#include "processor.hpp"
#include "archiver.hpp"
#include "poisson.hpp"
#include "metrics.hpp"

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

namespace fs = std::filesystem;

using slab::SlabPool;
using slab::Desc;
using pipeline::DmaSource;
using pipeline::DmaConfig;
using pipeline::Processor;
using pipeline::ProcessorConfig;
using pipeline::Archiver;
using pipeline::ArchiverConfig;
using pipeline::PipelineMetrics;

/// SPSC alias
template<typename T, std::size_t N>
using Ring = spsc::Ring<T, N>;

// ============================================================================
// Pipeline geometry / config
// ============================================================================
constexpr uint32_t IMAGE_W        = 300;
constexpr uint32_t IMAGE_H        = 300;
constexpr std::size_t FRAME_BYTES = std::size_t(IMAGE_W) * IMAGE_H * 2;
constexpr uint32_t SLABS          = 128;
constexpr int      FPS            = 1000;      // 1 kFPS target
constexpr int      FRAME_PERIOD   = 1000000 / FPS;
constexpr uint32_t NTH_ARCHIVE    = 10;        // every 10th frame archived
constexpr bool     DRAIN_QUEUES   = false;     // drain queues after processing

// ============================================================================
// ROI grid configuration
// ============================================================================
constexpr std::size_t  ROW_START     = 50;     // start row of ROI grid
constexpr std::size_t  COL_START     = 50;     // start column of ROI grid
constexpr std::size_t  ROW_END       = 250;    // end row of ROI grid
constexpr std::size_t  COL_END       = 250;    // end column of ROI grid
constexpr std::size_t  ROI_W         = 10;     // ROI width
constexpr std::size_t  ROI_H         = 10;     // ROI height
constexpr double       LAMBDA_OCC    = 10.0;   // Poisson distribution for occupied ROI
constexpr double       LAMBDA_EMPTY  = 0.5;    // Poisson distribution for empty ROI
constexpr double       FP_TARGET     = 1e-3;   // False positive rate
constexpr int          MAX_K         = 100;    // Maximum scan range

// ============================================================================
// SPSC queue configuration
// ============================================================================
constexpr uint32_t  PROC_Q_SIZE = 1024;
constexpr uint32_t  ARCH_Q_SIZE = 1024;

// ============================================================================
// DMA source configuration
// The DMA source will pace the frames to the rate. If the rate is not paced
// then the frames will be delivered as fast as possible.
// ============================================================================
constexpr bool      PACE_FRAMES   = true;
constexpr int       CPU_AFFINITY  = -1;     // no pinning
constexpr int       THREAD_NICE   = 0;      // no niceness
constexpr bool      LOOP_FRAMES   = true;   // loop through frames

// ============================================================================
// Processor configuration
// ============================================================================
constexpr uint32_t  WORKER_THREADS = 4;   // adjust per CPU
constexpr bool      PRINT_STDOUT   = false;
constexpr uint64_t  INTERVAL_PRINT = 37;  // Set some random prime number for interval printing

const std::string_view PROC_OUTPUT_DIR    = "PROC_out";

// ============================================================================
// Archiver configuration
// ============================================================================
constexpr uint64_t    SEGMENT_BYTES   = 4ull * 1024 * 1024 * 1024; // 4 GiB segments
constexpr uint64_t    IO_BUFFER_BYTES = 8ull * 1024 * 1024;
constexpr bool        MAKE_INDEX      = true;
constexpr bool        USE_IO_URING    = false;   // set true on Linux with io_uring if desired
constexpr bool        DIRECT_IO       = false;   // open with O_DIRECT (requires aligned writes)
constexpr unsigned    URING_QD        = 64;      // SQ/CQ depth
constexpr unsigned    MAX_INFLIGHT    = 32;      // cap in-flight write requests
constexpr std::size_t BLOCK_BYTES     = 4096;    // alignment & size multiple when using O_DIRECT

const std::string_view ARCH_OUTPUT_DIR    = "ARCH_out";
const std::string_view ARCH_OUTPUT_PREFIX = "frames";

// ============================================================================
// Helpers to manage bits and slab reclamation
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
// Load .npy frames from a directory into contiguous byte buffers
// ============================================================================
static std::vector<std::vector<std::byte>>
load_npy_frames(const std::string& dir) {
  std::vector<fs::path> files;
  for (auto& e : fs::directory_iterator(dir)) {
    if (!e.is_regular_file()) continue;
    if (e.path().extension() == ".npy") {
      files.push_back(e.path());
    }
  }
  std::sort(files.begin(), files.end());

  if (files.empty()) {
    throw std::runtime_error("No .npy files found in " + dir);
  }

  std::vector<std::vector<std::byte>> storage;
  storage.reserve(files.size());

  for (const auto& p : files) {
    cnpy::NpyArray arr = cnpy::npy_load(p.string());
    if (arr.word_size != 2) {
      throw std::runtime_error("npy word_size != 2 in " + p.string());
    }
    if (arr.shape.size() != 2) {
      throw std::runtime_error("npy is not 2D in " + p.string());
    }
    const std::size_t H = arr.shape[0];
    const std::size_t W = arr.shape[1];
    if (H != IMAGE_H || W != IMAGE_W) {
      throw std::runtime_error("npy shape mismatch (expected "
                               + std::to_string(IMAGE_H) + "x"
                               + std::to_string(IMAGE_W) + ") in " + p.string());
    }
    if (H * W * arr.word_size != FRAME_BYTES) {
      throw std::runtime_error("npy size mismatch vs FRAME_BYTES in " + p.string());
    }

    const std::uint16_t* src = arr.data<const std::uint16_t>();
    std::vector<std::byte> buf(FRAME_BYTES);
    std::memcpy(buf.data(), src, FRAME_BYTES);
    storage.emplace_back(std::move(buf));
  }

  std::printf("MAIN: Loaded %zu frames from %s\n", storage.size(), dir.c_str());
  return storage;
}


// ============================================================================
// Main
// ============================================================================
int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr,
      "Usage: %s <npy_dir> [run_seconds]\n"
      "Example: %s data/ 5\n",
      argv[0], argv[0]);
    return 1;
  }

  const std::string npy_dir = argv[1];
  double run_seconds = 5.0;
  if (argc >= 3) {
    run_seconds = std::atof(argv[2]);
    if (run_seconds <= 0.0) run_seconds = 5.0;
  }

  try {
    // ========================================================================
    // Load frames from .npy into host storage
    // ========================================================================
    auto storage = load_npy_frames(npy_dir);
    std::vector<std::span<const std::byte>> frames;
    frames.reserve(storage.size());
    for (auto& v : storage) {
      frames.emplace_back(v.data(), v.size());
    }
    const size_t n_frames = frames.size();

    // ========================================================================
    // Core resources: slab pool and queues
    // ========================================================================
    SlabPool pool(SLABS, FRAME_BYTES, /*alignment*/4096);

    Ring<Desc, PROC_Q_SIZE> proc_q;
    Ring<Desc, ARCH_Q_SIZE> arch_q;

    PipelineMetrics metrics;

    // ========================================================================
    // DMA source configuration and callbacks
    // ========================================================================
    DmaConfig dcfg;
    dcfg.paced           = PACE_FRAMES;
    dcfg.frame_period_us = FRAME_PERIOD;
    dcfg.expected_w      = IMAGE_W;
    dcfg.expected_h      = IMAGE_H;
    dcfg.nth_archive     = NTH_ARCHIVE;
    dcfg.cpu_affinity    = CPU_AFFINITY;
    dcfg.thread_nice     = THREAD_NICE;
    dcfg.loop_frames     = LOOP_FRAMES;

    /// on_proc: push into proc_q; if full, drop and clear PROC bit
    pipeline::ProcCallback on_proc = [&](const Desc& d){
      metrics.mark_dma_frame();
      if (!proc_q.push(d)) {
        metrics.mark_drop_proc_queue();
        release_proc(pool, d);
      }
    };
    
    /// on_arch: push into arch_q; if full, drop and clear ARCH bit
    pipeline::ArchCallback on_arch = [&](const Desc& d){
      metrics.mark_dma_arch_offered();
      if (!arch_q.push(d)) {
        metrics.mark_drop_arch_queue();
        release_arch(pool, d);
      }
    };

    DmaSource dma(pool, dcfg, on_proc, on_arch);
    dma.set_frames(std::move(frames));

    // ========================================================================
    // Generate timestamp once for both Archiver and Processor
    // ========================================================================
    auto now = std::chrono::system_clock::now();
    std::time_t now_time_t = std::chrono::system_clock::to_time_t(now);
    std::tm tm_buf;
#if defined(_WIN32) || defined(_WIN64)
    localtime_s(&tm_buf, &now_time_t);
    std::tm* now_tm = &tm_buf;
#else
    localtime_r(&now_time_t, &tm_buf);
    std::tm* now_tm = &tm_buf;
#endif
    char timestamp_str[64];
    std::strftime(timestamp_str, sizeof(timestamp_str), "%Y%m%d_%H%M%S", now_tm);

    // ========================================================================
    // Archiver setup
    // ========================================================================
    // Create timestamped subdirectory path for archiver
    std::filesystem::path archdir_path 
      = std::filesystem::path(ARCH_OUTPUT_DIR) / timestamp_str;
    
    ArchiverConfig acfg;
    acfg.output_dir      = archdir_path.string();
    acfg.file_prefix     = std::string(ARCH_OUTPUT_PREFIX);
    acfg.segment_bytes   = SEGMENT_BYTES;
    acfg.io_buffer_bytes = IO_BUFFER_BYTES;
    acfg.make_index      = MAKE_INDEX;
    acfg.use_io_uring    = USE_IO_URING;

    pipeline::PopFn arch_pop = [&](Desc& d) -> bool {
      return arch_q.pop(d);
    };

    Archiver arch(pool, acfg, arch_pop);

    // ========================================================================
    // Processor setup (ROIs + AVX worker pool)
    // ========================================================================
    
    auto rois = pipeline::build_rois_with_poisson(
        ROW_START, COL_START, ROW_END, COL_END,
        ROI_W, ROI_H,
        LAMBDA_OCC, LAMBDA_EMPTY,
        FP_TARGET, MAX_K);

    pipeline::PopFn proc_pop = [&](Desc& d) -> bool {
      return proc_q.pop(d);
    };

    // Create timestamped result file path for processor
    char result_name[128];
    std::snprintf(result_name, sizeof(result_name), "results_%s.txt", timestamp_str);
    std::filesystem::path result_path 
      = std::filesystem::path(PROC_OUTPUT_DIR) / result_name;

    /// On result callback (empty - Processor will write to file if configured)
    pipeline::ResultCallback on_result = nullptr;

    ProcessorConfig pcfg;
    pcfg.image_w        = IMAGE_W;
    pcfg.image_h        = IMAGE_H;
    pcfg.worker_threads = WORKER_THREADS;
    pcfg.print_stdout   = PRINT_STDOUT;
    pcfg.interval_print = INTERVAL_PRINT;
    pcfg.output_path    = result_path.string();  // Processor will create dir and file

    Processor proc(pool, pcfg, proc_pop, rois, on_result);

    // ========================================================================
    // Run pipeline
    // ========================================================================
    std::puts("MAIN: Starting pipeline...");
    if (PACE_FRAMES) {
      std::puts("MAIN: Pacing frames is enabled");
      std::printf("MAIN: Pacing frames to %d FPS\n", FPS);
    } else {
      std::puts("MAIN: Pacing frames is disabled");
      std::puts("MAIN: Frames will be delivered as fast as possible");
    }

    if (DRAIN_QUEUES) {
      std::puts("MAIN: Draining queues is enabled");
    } else {
      std::puts("MAIN: Draining queues is disabled");
    }

    auto t0 = std::chrono::steady_clock::now();

    arch.start();
    proc.start();
    dma.start();
    if (dcfg.loop_frames) {
      std::this_thread::sleep_for(std::chrono::duration<double>(run_seconds));
    } else {
      printf("MAIN: Waiting for DMA and PROC to finish...\n");
      while (true) {
        auto dma_done = dma.stats().produced.load();
        auto proc_done = proc.stats().frames.load();
        if (dma_done >= n_frames && proc_done >= n_frames) break;
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
      }
    }

    std::puts("MAIN: Stopping pipeline...");
    dma.stop();
    
    /// If DRAIN_QUEUES is true, wait for queues to drain so all frames are 
    /// processed. Otherwise, the pipeline will stop when the DMA is terminated.
    /// This might cause some frames to be dropped by the PROC or ARCH.
    if (DRAIN_QUEUES) {
      printf("MAIN: Waiting for queues to drain...\n");
      while (!proc_q.empty() || !arch_q.empty()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    
    proc.stop();
    arch.stop();

    auto t1 = std::chrono::steady_clock::now();
    double dt = std::chrono::duration<double>(t1 - t0).count();
    if (dt <= 0.0) dt = 1e-9;

    // ========================================================================
    // Stats and sanity checks
    // ========================================================================
    const auto& ds  = dma.stats();
    const auto& ps  = proc.stats();
    const auto& as  = arch.stats();

    // Sync metrics from component stats
    metrics.mark_dma_pool_exhaust(ds.pool_exhaust.load());
    metrics.mark_proc_frame(ps.frames.load());
    metrics.mark_proc_stale_desc(ps.stale_desc.load());
    metrics.mark_proc_empty_polls(ps.empty_polls.load());
    metrics.mark_arch_frame(as.archived_frames.load());
    metrics.mark_arch_bytes(as.archived_bytes.load());
    metrics.mark_arch_rotation(as.rotations.load());
    metrics.mark_arch_stale_desc(as.stale_desc.load());
    metrics.mark_arch_io_error(as.io_errors.load());

    /// Print unified metrics
    std::printf("\n");
    metrics.print(stdout);

    /// Sanity: slabs back in pool
    std::vector<std::uint32_t> ids;
    ids.reserve(pool.size());
    for (std::uint32_t i = 0; i < pool.size(); ++i) {
      auto id = pool.acquire();
      EXPECT_TRUE(id != UINT32_MAX);
      ids.push_back(id);
    }
    EXPECT_TRUE(pool.acquire() == UINT32_MAX);
    for (auto id : ids) pool.release(id);

    std::puts("\nMAIN: Pipeline run completed OK.");
    return 0;
  } catch (const std::exception& ex) {
    std::fprintf(stderr, "MAIN: FATAL: %s\n", ex.what());
    return 1;
  }
}
