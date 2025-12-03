// ============================================================================
// archiver.hpp -- Archiver for DMA Image Processor
//
// This header defines the Archiver class, responsible for persisting a
// continuous stream of slab-backed image frames provided by a slab pool to
// disk storage as a sequence of files ("segments"). 
//
// Core features:
//
// - Support for efficient and robust frame archival with file rotation at 
//   configurable segment sizes.
// - Optional I/O backends (e.g., stdio, Linux io_uring, O_DIRECT for 
//   unbuffered I/O).
// - Optional index file writing to record mapping from sequence numbers to 
//   file offsets.
//
// Types and interfaces defined:
// - ArchiverConfig: Configuration options (output directory, file prefix, 
//   segment size, I/O buffer size, CPU affinity, make index, use io_uring, 
//   direct I/O).
// - PopFn: Functional type for popping descriptors from the slab memory pool.
// - Archiver: Main class orchestrating frame archival and I/O operations.
// - Stats: Statistics for the Archiver class.
// ============================================================================
#pragma once
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <thread>

#include "slab_pool.hpp"   // slab::SlabPool, slab::Desc
#include "archiver.hpp"

namespace pipeline {

// ============================================================================
// ArchiverConfig: configuration for the Archiver class
// ============================================================================
struct ArchiverConfig {
  std::string output_dir      = "archive";
  std::string file_prefix     = "frames";
  std::size_t segment_bytes   = 4ull * 1024 * 1024 * 1024; // rotate at ~4 GiB
  std::size_t io_buffer_bytes = 8ull * 1024 * 1024;        // stdio buffer (fallback)
  int         cpu_affinity    = -1;                        // -1 = no pin
  bool        make_index      = true;                      // write a simple (seq,offset) text index

  // io_uring / direct I/O (Linux-only; ignored on other OSes)
  bool        use_io_uring    = false;   // enable io_uring backend
  bool        direct_io       = false;   // open with O_DIRECT (requires aligned writes)
  unsigned    uring_qd        = 64;      // SQ/CQ depth
  unsigned    max_inflight    = 32;      // cap in-flight write requests
  std::size_t block_bytes     = 4096;    // alignment & size multiple when using O_DIRECT
};

/// Pop function the archiver uses to get work from your SPSC queue.
using PopFn = std::function<bool(slab::Desc&)>;

// ============================================================================
// Archiver: pops descriptors, writes slab bytes to rotating files, 
// releases slabs.
// ============================================================================
class Archiver {
public:
  /// Constructor for the Archiver class
  /// @param pool The slab memory pool to use
  /// @param cfg The archiver configuration
  /// @param pop_fn The function to use to pop descriptors from the slab memory
  Archiver(slab::SlabPool& pool, ArchiverConfig cfg, PopFn pop_fn);
  ~Archiver();

  Archiver(const Archiver&) = delete;
  Archiver& operator=(const Archiver&) = delete;

  void start();
  void stop();

  struct Stats {
    std::atomic<uint64_t> archived_frames{0};
    std::atomic<uint64_t> archived_bytes{0};
    std::atomic<uint64_t> rotations{0};
    std::atomic<uint64_t> stale_desc{0};   // gen mismatch
    std::atomic<uint64_t> io_errors{0};
    std::atomic<uint64_t> empty_polls{0};
    std::atomic<uint64_t> uring_submits{0};
    std::atomic<uint64_t> uring_completions{0};
    std::atomic<uint64_t> uring_sq_full{0};
  };
  const Stats& stats() const noexcept;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace pipe
