// ============================================================================
// dma_source.cpp -- implementation of the DmaSource class
// ============================================================================
#include "dma_source.hpp"
#include <cstring>
#include <stdexcept>
#include <span>
#include <cstddef>

#if defined(__linux__)
  #include <pthread.h>
  #include <sched.h>
#endif

using cnpy::NpyArray;

namespace pipeline {

// ============================================================================
// Impl: implementation of the DmaSource class
// ============================================================================
struct DmaSource::Impl {
  slab::SlabPool& pool;
  DmaConfig cfg;
  ProcCallback  proc_cb;
  ArchCallback  arch_cb;

  std::vector<std::vector<std::byte>> storage;     // N × frame_bytes
  std::vector<std::span<const std::byte>> frames;  // views into storage

  /// Thread for the dma source loop
  std::thread th;
  std::atomic<bool> run{false};

  /// Sequence number for the current frame
  uint64_t seq{0};
  /// Statistics for the dma source
  Stats stats;

  /// Thread settings: configure thread behavior for the DMA source. Adjustments 
  /// here help improve real-time performance or isolate resources for the 
  /// DmaSource loop thread.
  void tune_thread() {
#if defined(__linux__)
    if (cfg.cpu_affinity >= 0) {
      cpu_set_t set; CPU_ZERO(&set); CPU_SET(cfg.cpu_affinity, &set);
      pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
    }
#endif
  }

  /// This is the main loop for the dma source. It acquires slabs from the pool,
  /// copies the frames into the slabs, sets the pending bits, and calls the 
  /// callbacks.
  void loop() {
    using clock = std::chrono::steady_clock;
    auto next = clock::now();
    const auto period = std::chrono::microseconds(cfg.frame_period_us);

    if (frames.empty()) return;

    std::size_t i = 0;
    const std::size_t fb = pool.frame_bytes();

    while (run.load(std::memory_order_relaxed)) {
      uint32_t id = pool.acquire();
      if (id == UINT32_MAX) {
        stats.pool_exhaust.fetch_add(1, std::memory_order_relaxed);
        std::this_thread::yield();
        if (cfg.paced) { next += period; std::this_thread::sleep_until(next); }
        continue;
      }

      auto& b = pool.get(id);

      /// This is the mock DMA write into the slab.
      std::memcpy(b.data, frames[i].data(), fb);

      /// Who needs this frame?
      uint8_t mask = slab::SlabPool::BIT_PROC;
      const bool do_arch = (cfg.nth_archive > 0) && arch_cb 
                            && ((seq % cfg.nth_archive) == 0);
      if (do_arch) mask |= slab::SlabPool::BIT_ARCH;

      /// Publish the frame to the slab pool.
      b.hdr.pending.store(mask, std::memory_order_release);
      slab::Desc d{ id, b.hdr.gen, seq };
      if (proc_cb) { 
        proc_cb(d); 
        stats.produced.fetch_add(1, std::memory_order_relaxed); 
      }
      if (do_arch) { 
        arch_cb(d); 
        stats.archived_offered.fetch_add(1, std::memory_order_relaxed); 
      }

      /// Next frame
      i = (i + 1) % frames.size();
      ++seq;

      /// Pause for the next frame if paced is true.
      if (cfg.paced) { next += period; std::this_thread::sleep_until(next); }
    }
  }

  /// Set the frames from a vector of NpyArrays.
  void set_frames_from_npy(const std::vector<NpyArray>& arrays) {
    const std::size_t fb = pool.frame_bytes();
    const uint32_t W = cfg.expected_w, H = cfg.expected_h;

    storage.clear();
    frames.clear();
    storage.reserve(arrays.size());
    frames.reserve(arrays.size());

    for (const auto& arr : arrays) {
      if (arr.word_size != 2) {
        throw std::runtime_error("DmaSource: npy word_size != 2 (uint16 required)");
      }
      if (arr.shape.size() != 2) {
        throw std::runtime_error("DmaSource: npy must be 2D");
      }
      if (arr.shape[0] != H || arr.shape[1] != W) {
        throw std::runtime_error("DmaSource: npy shape mismatch (expected HxW row-major)");
      }

      // allocate destination and copy
      std::vector<std::byte> buf(fb);
      std::memcpy(buf.data(), 
                  reinterpret_cast<const void*>(arr.data<const uint16_t>()), 
                  fb);
      storage.emplace_back(std::move(buf));
    }

    for (auto& v : storage) frames.emplace_back(v.data(), v.size());
  }
};

DmaSource::DmaSource(slab::SlabPool& pool, DmaConfig cfg,
                     ProcCallback proc_cb, ArchCallback arch_cb)
: impl_(std::unique_ptr<Impl>(new Impl{
    pool,
    std::move(cfg),
    std::move(proc_cb),
    std::move(arch_cb),
    {},  // storage
    {},  // frames
    {},  // th
    false,  // run
    0,  // seq
    {}  // stats
})) {}

DmaSource::~DmaSource() { stop(); }

void DmaSource::set_frames_from_npy(const std::vector<NpyArray>& arrays) {
  impl_->set_frames_from_npy(arrays);
}

void DmaSource::set_frames(std::vector<std::span<const std::byte>> frames) {
  const auto fb = impl_->pool.frame_bytes();
  for (auto s : frames) {
    if (s.size() != fb) {
      throw std::invalid_argument("DmaSource::set_frames: span size != pool.frame_bytes()");
    }
  }
  impl_->storage.clear();     // external memory; we keep only views
  impl_->frames = std::move(frames);
}

void DmaSource::start() {
  if (impl_->run.exchange(true)) return;
  impl_->th = std::thread([this]{ impl_->tune_thread(); impl_->loop(); });
}

void DmaSource::stop() {
  if (!impl_->run.exchange(false)) return;
  if (impl_->th.joinable()) impl_->th.join();
}

void DmaSource::set_paced(bool paced, int period_us) {
  impl_->cfg.paced = paced;
  if (period_us > 0) impl_->cfg.frame_period_us = period_us;
}

void DmaSource::set_nth_archive(uint32_t n) {
  impl_->cfg.nth_archive = n;
}

const DmaSource::Stats& DmaSource::stats() const noexcept { 
  return impl_->stats; 
}

} // namespace pipeline
