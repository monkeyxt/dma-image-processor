// ============================================================================
// `drive.cpp` -- End-to-end driver for DMA → Processor → Archiver
//
// Usage:
//   ./dma-image-processor <npy_dir> [run_seconds] [--config config.toml]
//
//  - Loads all *.npy frames in <npy_dir> (row-major H×W, uint16).
//  - Sets up slab pool, SPSC queues, DMA source, processor, and archiver.
//  - Uses PoissonDetector to set an occupancy threshold for all ROIs.
//  - Runs for run_seconds (default 5 s) and prints stats.
//  - Configuration can be loaded from a TOML file via --config flag (optional).
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

#include <cnpy.h>

#include "slab_pool.hpp"
#include "spsc_ring.hpp"
#include "dma_source.hpp"
#include "processor.hpp"
#include "archiver.hpp"
#include "poisson.hpp"
#include "metrics.hpp"
#include "config.hpp"

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
load_npy_frames(const std::string& dir, uint32_t expected_h, uint32_t expected_w, std::size_t expected_frame_bytes) {
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
    if (H != expected_h || W != expected_w) {
      throw std::runtime_error("npy shape mismatch (expected "
                               + std::to_string(expected_h) + "x"
                               + std::to_string(expected_w) + ") in " + p.string());
    }
    if (H * W * arr.word_size != expected_frame_bytes) {
      throw std::runtime_error("npy size mismatch vs FRAME_BYTES in " + p.string());
    }

    const std::uint16_t* src = arr.data<const std::uint16_t>();
    std::vector<std::byte> buf(expected_frame_bytes);
    std::memcpy(buf.data(), src, expected_frame_bytes);
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
      "Usage: %s <npy_dir> [run_seconds] [--config config.toml]\n"
      "Example: %s data/ 5\n"
      "Example: %s data/ 5 --config configs/default.toml\n",
      argv[0], argv[0], argv[0]);
    return 1;
  }

  const std::string npy_dir = argv[1];
  double run_seconds = 5.0;
  std::string config_path;
  
  // Parse arguments: [run_seconds] [--config config.toml]
  bool run_seconds_set = false;
  for (int i = 2; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--config" || arg == "-c") {
      if (i + 1 < argc) {
        config_path = argv[++i];
      } else {
        std::fprintf(stderr, "Error: --config requires a file path\n");
        return 1;
      }
    } else if (!run_seconds_set) {
      // Try to parse as run_seconds (only if not already set)
      double parsed = std::atof(arg.c_str());
      if (parsed > 0.0) {
        run_seconds = parsed;
        run_seconds_set = true;
      } else {
        std::fprintf(stderr, 
          "MAIN: Error: Invalid run_seconds value: %s\n", arg.c_str());
        std::fprintf(stderr, 
          "MAIN: Usage: %s <npy_dir> [run_seconds] [--config config.toml]\n", 
          argv[0]);
        return 1;
      }
    } else {
      // Unknown argument
      std::fprintf(stderr, 
        "MAIN: Error: Unknown argument: %s\n", arg.c_str());
      std::fprintf(stderr, 
        "MAIN: Usage: %s <npy_dir> [run_seconds] [--config config.toml]\n", 
        argv[0]);
      return 1;
    }
  }
  
  // Default to configs/default.toml if no config specified and it exists
  if (config_path.empty()) {
    std::filesystem::path default_config = "configs/default.toml";
    if (std::filesystem::exists(default_config)) {
      config_path = default_config.string();
    }
  }
  
  Config cfg;
  if (!config_path.empty()) {
    cfg = load_config(config_path);
  } else {
    std::printf("Using default configuration (no config file specified).\n");
  }

  try {
    // ========================================================================
    // Load frames from .npy into host storage
    // ========================================================================
    auto storage = load_npy_frames(npy_dir, cfg.IMAGE_H, cfg.IMAGE_W, cfg.FRAME_BYTES());
    std::vector<std::span<const std::byte>> frames;
    frames.reserve(storage.size());
    for (auto& v : storage) {
      frames.emplace_back(v.data(), v.size());
    }
    const size_t n_frames = frames.size();

    // ========================================================================
    // Core resources: slab pool and queues
    // ========================================================================
    SlabPool pool(cfg.SLABS, cfg.FRAME_BYTES(), /*alignment*/4096);

    Ring<Desc, Config::PROC_Q_SIZE_DEFAULT> proc_q;
    Ring<Desc, Config::ARCH_Q_SIZE_DEFAULT> arch_q;

    PipelineMetrics metrics;

    // ========================================================================
    // DMA source configuration and callbacks
    // ========================================================================
    DmaConfig dcfg;
    dcfg.paced           = cfg.PACE_FRAMES;
    dcfg.frame_period_us = cfg.FRAME_PERIOD();
    dcfg.expected_w      = cfg.IMAGE_W;
    dcfg.expected_h      = cfg.IMAGE_H;
    dcfg.nth_archive     = cfg.NTH_ARCHIVE;
    dcfg.cpu_affinity    = cfg.CPU_AFFINITY;
    dcfg.thread_nice     = cfg.THREAD_NICE;
    dcfg.loop_frames     = cfg.LOOP_FRAMES;

    /// on_proc: push into proc_q; if full, drop and clear PROC bit
    /// If the `proc_q` is full, this creates back pressure on the DMA source.
    /// The DMA source will drop frames until the `proc_q` has space.
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
      = std::filesystem::path(cfg.ARCH_OUTPUT_DIR) / timestamp_str;
    
    ArchiverConfig acfg;
    acfg.output_dir      = archdir_path.string();
    acfg.file_prefix     = cfg.ARCH_OUTPUT_PREFIX;
    acfg.segment_bytes   = cfg.SEGMENT_BYTES;
    acfg.io_buffer_bytes = cfg.IO_BUFFER_BYTES;
    acfg.make_index      = cfg.MAKE_INDEX;
    acfg.use_io_uring    = cfg.USE_IO_URING;

    pipeline::PopFn arch_pop = [&](Desc& d) -> bool {
      return arch_q.pop(d);
    };

    Archiver arch(pool, acfg, arch_pop);

    // ========================================================================
    // Processor setup (ROIs + AVX worker pool)
    // ========================================================================
    
    auto rois = pipeline::build_rois_with_poisson(
        cfg.ROW_START, cfg.COL_START, cfg.ROW_END, cfg.COL_END,
        cfg.ROI_W, cfg.ROI_H,
        cfg.LAMBDA_OCC, cfg.LAMBDA_EMPTY,
        cfg.FP_TARGET, cfg.MAX_K);

    pipeline::PopFn proc_pop = [&](Desc& d) -> bool {
      return proc_q.pop(d);
    };

    // Create timestamped result file path for processor
    char result_name[128];
    std::snprintf(result_name, sizeof(result_name), "results_%s.txt", timestamp_str);
    std::filesystem::path result_path 
      = std::filesystem::path(cfg.PROC_OUTPUT_DIR) / result_name;

    /// On result callback (empty - Processor will write to file if configured)
    pipeline::ResultCallback on_result = nullptr;

    ProcessorConfig pcfg;
    pcfg.image_w        = cfg.IMAGE_W;
    pcfg.image_h        = cfg.IMAGE_H;
    pcfg.worker_threads = cfg.WORKER_THREADS;
    pcfg.print_stdout   = cfg.PRINT_STDOUT;
    pcfg.interval_print = cfg.INTERVAL_PRINT;
    pcfg.output_path    = result_path.string();  // Processor will create dir and file

    Processor proc(pool, pcfg, proc_pop, rois, on_result);

    // ========================================================================
    // Run pipeline
    // ========================================================================
    std::puts("MAIN: Starting pipeline...");
    if (cfg.PACE_FRAMES) {
      std::puts("MAIN: Pacing frames is enabled");
      std::printf("MAIN: Pacing frames to %d FPS\n", cfg.FPS);
    } else {
      std::puts("MAIN: Pacing frames is disabled");
      std::puts("MAIN: Frames will be delivered as fast as possible");
    }

    if (cfg.DRAIN_QUEUES) {
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
    if (cfg.DRAIN_QUEUES) {
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
    /// In drain mode, all frames should be processed and all slabs released.
    /// In no-drain mode, frames remaining in queues keep their slabs allocated,
    /// so we only verify pool consistency rather than expecting all slabs back.
    std::vector<std::uint32_t> ids;
    ids.reserve(pool.size());
    
    if (cfg.DRAIN_QUEUES) {
      // Drain mode: expect all slabs back in pool
      for (std::uint32_t i = 0; i < pool.size(); ++i) {
        auto id = pool.acquire();
        EXPECT_TRUE(id != UINT32_MAX);
        ids.push_back(id);
      }
      EXPECT_TRUE(pool.acquire() == UINT32_MAX);
      std::printf("MAIN: Sanity check passed: all %zu slabs back in pool\n", 
                  ids.size());
    } else {
      // No-drain mode: some slabs may be stuck in queues, just verify pool works
      // Wait briefly for any in-flight processing to complete
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
      
      uint32_t available_count = 0;
      while (true) {
        auto id = pool.acquire();
        if (id == UINT32_MAX) break;
        ids.push_back(id);
        ++available_count;
      }
      std::printf("\nMAIN: Sanity check: %u/%u slabs available (no-drain mode)\n", 
                  available_count, pool.size());
    }
    
    // Release all acquired slabs back
    for (auto id : ids) pool.release(id);

    std::puts("\nMAIN: Pipeline run completed OK.");
    return 0;
  } catch (const std::exception& ex) {
    std::fprintf(stderr, "MAIN: FATAL: %s\n", ex.what());
    return 1;
  }
}
