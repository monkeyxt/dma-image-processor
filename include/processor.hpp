// ============================================================================
// processor.hpp -- Processor for ROI occupancy detection
//
// This header defines the Processor class responsible for analyzing slab-backed
// image frames (e.g., from DMA input) and performing region-of-interest (ROI)
// evaluation for photon occupancy detection.
//
// Core features:
//
// - Support for rectangular ROIs, each with configurable thresholds.
// - Multi-threaded processing with a tiling worker pool, leveraging AVX2 for 
//   efficient computation.
// - Thread-affinity options for performance tuning.
// - Integration with slab memory pools, and pop function abstractions for 
//   input queue handling.
// - Callback-based result delivery with per-frame ROI sum and occupancy 
//   classification.
//
// Types and interfaces defined:
// - ROI: Structure representing a rectangular region with a threshold.
// - ProcessorConfig: Configuration options (dimensions, CPU pinning, 
//   threading, output).
// - PopFn, ResultCallback: Functional types for input dequeuing and result 
//   handling.
// - Processor: Main class orchestrating ROI evaluation and worker pool 
//   management.
//
// See implementation for details on processing pipeline and concurrency model.
// ============================================================================
#pragma once
#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "slab_pool.hpp"   // For slab::SlabPool, slab::Desc

namespace pipeline {

// ============================================================================
// `ROI` class
// Rectangle ROI in image pixel coords (row-major H×W, uint16 pixels)
// ============================================================================
struct ROI {
  int x{0}, y{0}, w{0}, h{0};
  uint32_t threshold{0};   // classify occupied if sum >= threshold
};

// ============================================================================
// `ProcessorConfig` class
// Configuration for the `Processor` class
// ============================================================================
struct ProcessorConfig {
  uint32_t image_w{4096};         // width of the image in pixels
  uint32_t image_h{256};          // height of the image in pixels
  int      cpu_affinity{-1};      // -1 = don't pin ingress thread
  bool     print_stdout{false};   // print to stdout if no callback
  unsigned worker_threads{0};     // 0 => use hardware_concurrency(), 1 => single-thread
};

/// Pop function the processor uses to get descriptors from your SPSC queue.
using PopFn = std::function<bool(slab::Desc&)>;

// Callback with per-frame classification results.
// `roi_sums.size() == occupied.size() == rois.size()`.
// Return false if you want to stop the processor after this frame.
using ResultCallback = std::function<bool(uint64_t seq,
                                          const std::vector<uint64_t>& roi_sums,
                                          const std::vector<uint8_t>& occupied)>;

// ============================================================================
// Processor class
// AVX2-based processor with a tiled worker pool over ROIs.
// ============================================================================
class Processor {
public:
  /// Default constructor for the Processor class
  /// @param pool The slab memory pool to use
  /// @param cfg The processor configuration
  /// @param pop_fn The function to use to pop descriptors from the slab memory
  /// @param rois The ROIs to process
  /// @param on_result The function to use to process the results
  Processor(slab::SlabPool& pool, ProcessorConfig cfg, PopFn pop_fn,
            std::vector<ROI> rois, ResultCallback on_result = {});
  ~Processor();

  Processor(const Processor&) = delete;
  Processor& operator=(const Processor&) = delete;

  void start();
  void stop();

  struct Stats {
    std::atomic<uint64_t> frames{0};
    std::atomic<uint64_t> stale_desc{0};
    std::atomic<uint64_t> empty_polls{0};
  };
  const Stats& stats() const noexcept;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace pipeline
