// ============================================================================
// metrics.hpp -- simple metrics for DMA → Processor → Archiver pipeline
// ============================================================================
#pragma once
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>

namespace pipeline {

// ============================================================================
// `PipelineMetricsSnapshot` struct
// Snapshot of metrics at a point in time, with derived rates.
// ============================================================================
struct PipelineMetricsSnapshot {
  // timestamps (nanoseconds since steady_clock::epoch)
  std::uint64_t start_ns{0};
  std::uint64_t now_ns{0};
  double        elapsed_sec{0.0};

  // raw counters
  std::uint64_t dma_frames{0};
  std::uint64_t dma_arch_offered{0};
  std::uint64_t dma_pool_exhaust{0};

  std::uint64_t proc_frames{0};
  std::uint64_t proc_stale_desc{0};

  std::uint64_t arch_frames{0};
  std::uint64_t arch_bytes{0};
  std::uint64_t arch_rotations{0};
  std::uint64_t arch_stale_desc{0};
  std::uint64_t arch_io_errors{0};

  std::uint64_t drops_proc_queue{0};
  std::uint64_t drops_arch_queue{0};

  // derived rates (per second)
  double dma_fps{0.0};
  double proc_fps{0.0};
  double arch_fps{0.0};
  double arch_gbps{0.0};  // bytes/sec / 1e9
  double arch_gibps{0.0}; // bytes/sec / 1024^3
};

// ============================================================================
// `PipelineMetrics` class
// Lightweight metrics collector for the pipeline.
// Thread-safe: all increments are atomic; snapshot is mostly lock-free vs. 
// monotonic clocks.
// ============================================================================
class PipelineMetrics {
public:
  PipelineMetrics();

  // Mark events (all atomic, safe from multiple threads)
  void mark_dma_frame(std::uint64_t n = 1);
  void mark_dma_arch_offered(std::uint64_t n = 1);
  void mark_dma_pool_exhaust(std::uint64_t n = 1);

  void mark_proc_frame(std::uint64_t n = 1);
  void mark_proc_stale_desc(std::uint64_t n = 1);

  void mark_arch_frame(std::uint64_t n = 1);
  void mark_arch_bytes(std::uint64_t bytes);
  void mark_arch_rotation(std::uint64_t n = 1);
  void mark_arch_stale_desc(std::uint64_t n = 1);
  void mark_arch_io_error(std::uint64_t n = 1);

  void mark_drop_proc_queue(std::uint64_t n = 1);
  void mark_drop_arch_queue(std::uint64_t n = 1);

  // Reset all counters and timers (careful if other threads are reading)
  void reset();

  // Take a snapshot and compute derived rates.
  PipelineMetricsSnapshot snapshot() const;

  // Pretty-print snapshot to a FILE* (stdout by default).
  void print(std::FILE* out = stdout) const;

private:
  using clock = std::chrono::steady_clock;

  clock::time_point start_;

  std::atomic<std::uint64_t> dma_frames_;
  std::atomic<std::uint64_t> dma_arch_offered_;
  std::atomic<std::uint64_t> dma_pool_exhaust_;

  std::atomic<std::uint64_t> proc_frames_;
  std::atomic<std::uint64_t> proc_stale_desc_;

  std::atomic<std::uint64_t> arch_frames_;
  std::atomic<std::uint64_t> arch_bytes_;
  std::atomic<std::uint64_t> arch_rotations_;
  std::atomic<std::uint64_t> arch_stale_desc_;
  std::atomic<std::uint64_t> arch_io_errors_;

  std::atomic<std::uint64_t> drops_proc_queue_;
  std::atomic<std::uint64_t> drops_arch_queue_;
};

} // namespace pipe
