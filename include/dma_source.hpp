// ============================================================================
// dma_source.hpp -- DMA Source for DMA Image Processor
//
// This header contains the definition of the DmaSource class, which is used to
// source frames from a DMA-capable source and publish them to a slab pool. 
// ============================================================================
#pragma once
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <span>
#include <thread>
#include <vector>

#include "slab_pool.hpp"   // For slab::SlabPool, slab::Desc
#include "cnpy.hpp"        // For cnpy::NpyArray

using cnpy::NpyArray;

namespace pipeline {

// ============================================================================
// `DmaConfig` struct
// Configuration for the `DmaSource` class
// ============================================================================
struct DmaConfig {
  bool     paced           { true };     // sleep_until(period) if true
  int      frame_period_us { 1000 };     // 1 kFPS
  int      cpu_affinity    { -1 };       // -1 = no pinning
  int      thread_nice     { 0 };
  uint32_t nth_archive     { 0 };        // 0 = disable arch publishing
  uint32_t expected_w      { 4096 };
  uint32_t expected_h      { 256  };
  bool     loop_frames     { true };     // if true, loop through frames; if false, stop at end
};

using ProcCallback = std::function<void(const slab::Desc&)>;
using ArchCallback = std::function<void(const slab::Desc&)>;

// ============================================================================
// `DmaSource` class
// Source frames from a DMA-capable source and publish them to a slab pool.
// ============================================================================
class DmaSource {
public:
  /// Constructor for the DmaSource class.
  /// @param pool The slab pool to use.
  /// @param cfg The configuration for the DmaSource.
  /// @param proc_cb The callback to use for processing the frames.
  /// @param arch_cb The callback to use for archiving the frames.
  DmaSource(slab::SlabPool& pool, DmaConfig cfg,
            ProcCallback proc_cb, ArchCallback arch_cb = {});
  ~DmaSource();

  DmaSource(const DmaSource&) = delete;
  DmaSource& operator=(const DmaSource&) = delete;

  /// Set the frames from a vector of NpyArrays.
  /// @param arrays The vector of NpyArrays to set the frames from.
  void set_frames_from_npy(const std::vector<NpyArray>& arrays);

  /// Set the frames from a vector of spans of bytes.
  /// @param frames The vector of spans of bytes to set the frames from.
  void set_frames(std::vector<std::span<const std::byte>> frames);

  // Control
  void start();
  void stop();

  /// Set the paced mode.
  /// @param paced Whether to pace the frames.
  /// @param period_us The period in microseconds to pace the frames.
  void set_paced(bool paced, int period_us);

  /// Set the nth archive.
  /// @param n the nth archive to archive.
  void set_nth_archive(uint32_t n);

  /// Set whether to loop through frames.
  /// @param loop If true, continuously loop through frames; if false, stop at end.
  void set_loop_frames(bool loop);

  struct Stats {
    std::atomic<uint64_t> produced{0};
    std::atomic<uint64_t> archived_offered{0};
    std::atomic<uint64_t> pool_exhaust{0};
  };
  const Stats& stats() const noexcept;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace pipeline
