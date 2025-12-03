// ============================================================================
// `drive.cpp` -- End-to-end driver for DMA → Processor → Archiver
// ============================================================================
//
// Usage:
//   ./dma-image-processor <npy_dir> [run_seconds]
//
//  - Loads all *.npy frames in <npy_dir> (row-major H×W, uint16).
//  - Sets up slab pool, SPSC queues, DMA source, processor, and archiver.
//  - Uses PoissonDetector to set an occupancy threshold for all ROIs.
//  - Runs for run_seconds (default 5 s) and prints stats.
//
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
#include "poisson_detector.hpp"
#include "metrics.hpp"

#define EXPECT_TRUE(x) do{ if(!(x)){ std::fprintf(stderr,"EXPECT_TRUE failed: %s @ %s:%d\n",#x,__FILE__,__LINE__); std::abort(); } }while(0)
#define EXPECT_EQ(a,b) do{ auto _va=(a), _vb=(b); if(!((_va)==(_vb))){ std::fprintf(stderr,"EXPECT_EQ failed: %s=%lld %s=%lld @ %s:%d\n",#a,(long long)_va,#b,(long long)_vb,__FILE__,__LINE__); std::abort(); } }while(0)

namespace fs = std::filesystem;

using slab::SlabPool;
using slab::Desc;
using pipeline::DmaSource;
using pipeline::DmaConfig;
using pipeline::Processor;
using pipeline::ProcessorConfig;
using pipeline::ROI;
using pipeline::Archiver;
using pipeline::ArchiverConfig;
using pipeline::PipelineMetrics;

/// SPSC alias
template<typename T, std::size_t N>
using Ring = spsc::Ring<T, N>;

// ============================================================================
// Pipeline geometry / config
// ============================================================================
constexpr uint32_t IMAGE_W      = 300;
constexpr uint32_t IMAGE_H      = 300;
constexpr std::size_t FRAME_BYTES = std::size_t(IMAGE_W) * IMAGE_H * 2; // uint16
constexpr uint32_t SLABS        = 128;
constexpr int      FPS = 1000;  // 1 kFPS target
constexpr int      FRAME_PERIOD_US = 1000000 / FPS;
constexpr uint32_t NTH_ARCHIVE  = 10;        // every 10th frame archived

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

  std::printf("Loaded %zu frames from %s\n", storage.size(), dir.c_str());
  return storage;
}

// ============================================================================
// Build a grid of ROIs and assign Poisson thresholds
// The template parameters are the start and end coordinates of the ROI grid.
// ============================================================================
template<int ROW_START, int COL_START, int ROW_END, int COL_END>
static std::vector<ROI> build_rois_with_poisson() {

  constexpr int ROI_W = 10;
  constexpr int ROI_H = 10;

  std::vector<ROI> rois;
  for (int y = ROW_START; y + ROI_H <= ROW_END; y += ROI_H) {
    for (int x = COL_START; x + ROI_W <= COL_END; x += ROI_W) {
      //printf("ROI: x=%d y=%d w=%d h=%d\n", x, y, ROI_W, ROI_H);
      ROI r{x, y, ROI_W, ROI_H, 0};
      rois.emplace_back(r);
    }
  }
  printf("Built %zu ROIs\n", rois.size());

  const double lambda_occ   = 10.0;   // photons when occupied
  const double lambda_empty = 0.5;    // effective noise photons when empty
  const double fp_target    = 1e-3;   // max false positive rate
  const int    max_k        = 100;    // scan range

  pipeline::PoissonDetector det(lambda_occ, lambda_empty, fp_target, max_k);
  const int T = det.threshold();

  std::printf("PoissonDetector threshold T = %d "
              "(FP=%.3e, FN=%.3e)\n",
              T,
              det.false_positive_rate(),
              det.false_negative_rate());

  for (auto& r : rois) {
    r.threshold = static_cast<std::uint32_t>(T);
  }

  std::printf("Configured %zu ROIs of size %dx%d\n",
              rois.size(), ROI_W, ROI_H);
  return rois;
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

    // ========================================================================
    // Core resources: slab pool and queues
    // ========================================================================
    SlabPool pool(SLABS, FRAME_BYTES, /*alignment*/4096);

    Ring<Desc, 1024> proc_q;
    Ring<Desc, 1024> arch_q;

    PipelineMetrics metrics;

    // ========================================================================
    // DMA source configuration and callbacks
    // ========================================================================
    DmaConfig dcfg;
    dcfg.paced           = true;
    dcfg.frame_period_us = FRAME_PERIOD_US;
    dcfg.expected_w      = IMAGE_W;
    dcfg.expected_h      = IMAGE_H;
    dcfg.nth_archive     = NTH_ARCHIVE;
    dcfg.cpu_affinity    = -1;  // no pin
    dcfg.thread_nice     = 0;

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
    // Archiver setup
    // ========================================================================
    ArchiverConfig acfg;
    acfg.output_dir      = "archive_out";
    acfg.file_prefix     = "frames";
    acfg.segment_bytes   = 4ull * 1024 * 1024 * 1024; // 4 GiB segments
    acfg.io_buffer_bytes = 8ull * 1024 * 1024;
    acfg.make_index      = true;
    acfg.use_io_uring    = false; // set true on Linux with io_uring if desired

    pipeline::PopFn arch_pop = [&](Desc& d) -> bool {
      return arch_q.pop(d);
    };

    Archiver arch(pool, acfg, arch_pop);

    // ========================================================================
    // Processor setup (ROIs + AVX worker pool)
    // ========================================================================
    
    constexpr int ROW_START = 50;
    constexpr int COL_START = 50;
    constexpr int ROW_END = 250;
    constexpr int COL_END = 250;
    auto rois = build_rois_with_poisson<ROW_START, COL_START, ROW_END, COL_END>();

    pipeline::PopFn proc_pop = [&](Desc& d) -> bool {
      return proc_q.pop(d);
    };

    std::uint64_t interval{37};
    pipeline::ResultCallback on_result =
      [&](std::uint64_t seq,
          const std::vector<std::uint64_t>& sums,
          const std::vector<std::uint8_t>& occ) -> bool {
        (void)sums;
        /// (void)occ;

        if ((seq % interval) == 0ull) {
          std::printf("[frame %llu] rois=%zu\n",
                      (unsigned long long)seq, rois.size());

          // Print the sums vector
          // std::printf("sums: [");
          // for (size_t i = 0; i < sums.size(); ++i) {
          //   std::printf("%llu", static_cast<unsigned long long>(sums[i]));
          //   if (i + 1 < sums.size()) std::printf(", ");
          // }
          // std::printf("\n");
          // std::printf("]\n");

          // Print the occ vector
          //std::printf("occ:  [");
          for (size_t i = 0; i < occ.size(); ++i) {
            std::printf("%u", static_cast<unsigned int>(occ[i]));
            ///if (i + 1 < occ.size()) std::printf(", ");
          }
          std::printf("\n");
          //std::printf("]\n");
        }



        return true;
      };

    ProcessorConfig pcfg;
    pcfg.image_w        = IMAGE_W;
    pcfg.image_h        = IMAGE_H;
    pcfg.worker_threads = 4;   // adjust per CPU
    pcfg.print_stdout   = false;

    Processor proc(pool, pcfg, proc_pop, rois, on_result);

    // ========================================================================
    // Run pipeline
    // ========================================================================
    std::puts("Starting pipeline...");
    auto t0 = std::chrono::steady_clock::now();

    arch.start();
    proc.start();
    dma.start();

    std::this_thread::sleep_for(std::chrono::duration<double>(run_seconds));

    std::puts("Stopping pipeline...");
    dma.stop();
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
    //const auto  snap = metrics.snapshot();

    std::printf("\n=== Component Stats ===\n");
    std::printf("DMA:    produced=%llu  archived_offered=%llu  pool_exhaust=%llu\n",
                (unsigned long long)ds.produced.load(),
                (unsigned long long)ds.archived_offered.load(),
                (unsigned long long)ds.pool_exhaust.load());
    std::printf("PROC:   frames=%llu  stale_desc=%llu  empty_polls=%llu\n",
                (unsigned long long)ps.frames.load(),
                (unsigned long long)ps.stale_desc.load(),
                (unsigned long long)ps.empty_polls.load());
    std::printf("ARCH:   frames=%llu  bytes=%llu  rotations=%llu  stale_desc=%llu  io_errors=%llu\n",
                (unsigned long long)as.archived_frames.load(),
                (unsigned long long)as.archived_bytes.load(),
                (unsigned long long)as.rotations.load(),
                (unsigned long long)as.stale_desc.load(),
                (unsigned long long)as.io_errors.load());

    std::printf("\n=== End-to-end throughput ===\n");
    double dma_fps   = ds.produced.load() / dt;
    double proc_fps  = ps.frames.load()   / dt;
    double arch_fps  = as.archived_frames.load() / dt;
    double arch_gbps = (as.archived_bytes.load() / dt) / 1e9;
    double arch_gibps= (as.archived_bytes.load() / dt) / (1024.0*1024.0*1024.0);

    std::printf("Elapsed: %.3f s\n", dt);
    std::printf("DMA:     %.1f fps\n", dma_fps);
    std::printf("PROC:    %.1f fps\n", proc_fps);
    std::printf("ARCH:    %.1f fps | %.3f GB/s (%.3f GiB/s)\n",
                arch_fps, arch_gbps, arch_gibps);

    //std::printf("\n=== PipelineMetrics snapshot ===\n");
    //metrics.print(stdout);

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

    std::puts("\nPipeline run completed OK.");
    return 0;
  } catch (const std::exception& ex) {
    std::fprintf(stderr, "FATAL: %s\n", ex.what());
    return 1;
  }
}
