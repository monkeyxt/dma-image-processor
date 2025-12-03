// ============================================================================
// metrics.cpp -- implementation of PipelineMetrics
//
// PipelineMetrics tracks the following counters for pipeline statistics:
//
// dma_frames_          : Number of frames delivered by the DMA source.
// dma_arch_offered_    : Number of frames that DMA offered to the archiver queue.
// dma_pool_exhaust_    : Number of times the DMA source could not acquire a free slab from the pool.
// proc_frames_         : Number of frames processed by the processor.
// proc_stale_desc_     : Number of stale descriptor references seen by the processor.
// arch_frames_         : Number of frames written by the archiver.
// arch_bytes_          : Total bytes written by the archiver.
// arch_rotations_      : Number of output segment rotations/events in the archiver.
// arch_stale_desc_     : Number of stale descriptors encountered by the archiver.
// arch_io_errors_      : Number of I/O errors during archiving.
// drops_proc_queue_    : Number of frames dropped due to a full processor queue.
// drops_arch_queue_    : Number of frames dropped due to a full archiver queue.
// ============================================================================
#include "metrics.hpp"

#include <iomanip>
#include <iostream>

namespace pipeline {

PipelineMetrics::PipelineMetrics()
  : start_(clock::now()),
    dma_frames_(0),
    dma_arch_offered_(0),
    dma_pool_exhaust_(0),
    proc_frames_(0),
    proc_stale_desc_(0),
    proc_empty_polls_(0),
    arch_frames_(0),
    arch_bytes_(0),
    arch_rotations_(0),
    arch_stale_desc_(0),
    arch_io_errors_(0),
    drops_proc_queue_(0),
    drops_arch_queue_(0)
{}

void PipelineMetrics::mark_dma_frame(std::uint64_t n) {
  dma_frames_.fetch_add(n, std::memory_order_relaxed);
}
void PipelineMetrics::mark_dma_arch_offered(std::uint64_t n) {
  dma_arch_offered_.fetch_add(n, std::memory_order_relaxed);
}
void PipelineMetrics::mark_dma_pool_exhaust(std::uint64_t n) {
  dma_pool_exhaust_.fetch_add(n, std::memory_order_relaxed);
}

void PipelineMetrics::mark_proc_frame(std::uint64_t n) {
  proc_frames_.fetch_add(n, std::memory_order_relaxed);
}
void PipelineMetrics::mark_proc_stale_desc(std::uint64_t n) {
  proc_stale_desc_.fetch_add(n, std::memory_order_relaxed);
}
void PipelineMetrics::mark_proc_empty_polls(std::uint64_t n) {
  proc_empty_polls_.fetch_add(n, std::memory_order_relaxed);
}

void PipelineMetrics::mark_arch_frame(std::uint64_t n) {
  arch_frames_.fetch_add(n, std::memory_order_relaxed);
}
void PipelineMetrics::mark_arch_bytes(std::uint64_t bytes) {
  arch_bytes_.fetch_add(bytes, std::memory_order_relaxed);
}
void PipelineMetrics::mark_arch_rotation(std::uint64_t n) {
  arch_rotations_.fetch_add(n, std::memory_order_relaxed);
}
void PipelineMetrics::mark_arch_stale_desc(std::uint64_t n) {
  arch_stale_desc_.fetch_add(n, std::memory_order_relaxed);
}
void PipelineMetrics::mark_arch_io_error(std::uint64_t n) {
  arch_io_errors_.fetch_add(n, std::memory_order_relaxed);
}

void PipelineMetrics::mark_drop_proc_queue(std::uint64_t n) {
  drops_proc_queue_.fetch_add(n, std::memory_order_relaxed);
}
void PipelineMetrics::mark_drop_arch_queue(std::uint64_t n) {
  drops_arch_queue_.fetch_add(n, std::memory_order_relaxed);
}

void PipelineMetrics::reset() {
  start_ = clock::now();
  dma_frames_.store(0, std::memory_order_relaxed);
  dma_arch_offered_.store(0, std::memory_order_relaxed);
  dma_pool_exhaust_.store(0, std::memory_order_relaxed);

  proc_frames_.store(0, std::memory_order_relaxed);
  proc_stale_desc_.store(0, std::memory_order_relaxed);
  proc_empty_polls_.store(0, std::memory_order_relaxed);

  arch_frames_.store(0, std::memory_order_relaxed);
  arch_bytes_.store(0, std::memory_order_relaxed);
  arch_rotations_.store(0, std::memory_order_relaxed);
  arch_stale_desc_.store(0, std::memory_order_relaxed);
  arch_io_errors_.store(0, std::memory_order_relaxed);

  drops_proc_queue_.store(0, std::memory_order_relaxed);
  drops_arch_queue_.store(0, std::memory_order_relaxed);
}

PipelineMetricsSnapshot PipelineMetrics::snapshot() const {
  PipelineMetricsSnapshot s{};

  const auto start = start_;
  const auto now   = clock::now();
  const auto ns0   = std::chrono::time_point_cast<std::chrono::nanoseconds>(start);
  const auto ns1   = std::chrono::time_point_cast<std::chrono::nanoseconds>(now);
  s.start_ns       = static_cast<std::uint64_t>(ns0.time_since_epoch().count());
  s.now_ns         = static_cast<std::uint64_t>(ns1.time_since_epoch().count());
  s.elapsed_sec    = std::chrono::duration<double>(now - start).count();
  if (s.elapsed_sec <= 0.0) s.elapsed_sec = 1e-9; // avoid div-by-zero

  // Load counters
  s.dma_frames        = dma_frames_.load(std::memory_order_relaxed);
  s.dma_arch_offered  = dma_arch_offered_.load(std::memory_order_relaxed);
  s.dma_pool_exhaust  = dma_pool_exhaust_.load(std::memory_order_relaxed);

  s.proc_frames       = proc_frames_.load(std::memory_order_relaxed);
  s.proc_stale_desc   = proc_stale_desc_.load(std::memory_order_relaxed);
  s.proc_empty_polls  = proc_empty_polls_.load(std::memory_order_relaxed);

  s.arch_frames       = arch_frames_.load(std::memory_order_relaxed);
  s.arch_bytes        = arch_bytes_.load(std::memory_order_relaxed);
  s.arch_rotations    = arch_rotations_.load(std::memory_order_relaxed);
  s.arch_stale_desc   = arch_stale_desc_.load(std::memory_order_relaxed);
  s.arch_io_errors    = arch_io_errors_.load(std::memory_order_relaxed);

  s.drops_proc_queue  = drops_proc_queue_.load(std::memory_order_relaxed);
  s.drops_arch_queue  = drops_arch_queue_.load(std::memory_order_relaxed);

  // Derived rates
  const double dt = s.elapsed_sec;
  s.dma_fps   = s.dma_frames / dt;
  s.proc_fps  = s.proc_frames / dt;
  s.arch_fps  = s.arch_frames / dt;
  s.arch_gbps = (s.arch_bytes / dt) / 1e9;
  s.arch_gibps = (s.arch_bytes / dt) / (1024.0 * 1024.0 * 1024.0);

  return s;
}

void PipelineMetrics::print(std::FILE* out) const {
  PipelineMetricsSnapshot s = snapshot();

  std::fprintf(out,
    "\n=== Component Stats ===\n"
    "DMA:    produced=%llu  archived_offered=%llu  pool_exhaust=%llu\n"
    "PROC:   frames=%llu  stale_desc=%llu  empty_polls=%llu\n"
    "ARCH:   frames=%llu  bytes=%llu  rotations=%llu  stale_desc=%llu  io_errors=%llu\n"
    "\n=== End-to-end throughput ===\n"
    "Elapsed: %.3f s\n"
    "DMA:     %.1f fps\n"
    "PROC:    %.1f fps\n"
    "ARCH:    %.1f fps | %.3f GB/s (%.3f GiB/s)\n",
    (unsigned long long)s.dma_frames,
    (unsigned long long)s.dma_arch_offered,
    (unsigned long long)s.dma_pool_exhaust,
    (unsigned long long)s.proc_frames,
    (unsigned long long)s.proc_stale_desc,
    (unsigned long long)s.proc_empty_polls,
    (unsigned long long)s.arch_frames,
    (unsigned long long)s.arch_bytes,
    (unsigned long long)s.arch_rotations,
    (unsigned long long)s.arch_stale_desc,
    (unsigned long long)s.arch_io_errors,
    s.elapsed_sec,
    s.dma_fps,
    s.proc_fps,
    s.arch_fps,
    s.arch_gbps,
    s.arch_gibps
  );
}

} // namespace pipe
