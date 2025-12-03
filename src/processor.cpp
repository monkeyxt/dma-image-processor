// ============================================================================
// processor.cpp -- implementation of the Processor class
// ============================================================================
#include "processor.hpp"

#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <condition_variable>
#include <mutex>

#if defined(__linux__)
  #include <pthread.h>
#endif

#include <cstddef>
#if defined(__AVX2__)
  #include <immintrin.h>
#endif

namespace pipeline {

// ============================================================================
// Fallback ROI sum helpers. This is used when AVX2/AVX512 is not available.
// The code are written in a way such that only 8-byte aligned memory is 
// accessed to avoid potential performance issues.
// ============================================================================
static inline uint64_t sum_roi_scalar(const uint16_t* img,
                                      int img_w,
                                      int x, int y, int w, int h) {
  uint64_t sum = 0;
  const uint16_t* row = img + y * img_w + x;
  for (int r = 0; r < h; ++r) {
    const uint16_t* p = row;
    int c = 0;
    for (; c + 7 < w; c += 8) {
      sum += p[0] + p[1] + p[2] + p[3] + p[4] + p[5] + p[6] + p[7];
      p += 8;
    }
    for (; c < w; ++c) sum += *p++;
    row += img_w;
  }
  return sum;
}

#if defined(__AVX2__)
static inline uint64_t horizontal_sum_epi32(__m256i v) {
  __m128i vlow  = _mm256_castsi256_si128(v);
  __m128i vhigh = _mm256_extracti128_si256(v, 1);
  __m128i vsum  = _mm_add_epi32(vlow, vhigh);          // 4 lanes
  vsum = _mm_hadd_epi32(vsum, vsum);                   // 2 lanes
  vsum = _mm_hadd_epi32(vsum, vsum);                   // 1 lane
  return static_cast<uint64_t>(_mm_cvtsi128_si32(vsum));
}

static inline uint64_t sum_roi_avx2(const uint16_t* img,
                                    int img_w,
                                    int x, int y, int w, int h) {
  const uint16_t* row = img + y * img_w + x;
  uint64_t total = 0;
  for (int r = 0; r < h; ++r) {
    const uint16_t* p = row;
    int remaining = w;

    __m256i acc = _mm256_setzero_si256();

    // process 16 pixels per iteration
    while (remaining >= 16) {
      __m256i v16 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p));
      __m256i lo  = _mm256_cvtepu16_epi32(_mm256_castsi256_si128(v16));
      __m256i hi  = _mm256_cvtepu16_epi32(_mm256_extracti128_si256(v16, 1));
      acc = _mm256_add_epi32(acc, lo);
      acc = _mm256_add_epi32(acc, hi);
      p += 16;
      remaining -= 16;
    }

    total += horizontal_sum_epi32(acc);

    while (remaining >= 8) {
      __m128i v8  = _mm_loadu_si128(reinterpret_cast<const __m128i*>(p));
      __m256i w32 = _mm256_cvtepu16_epi32(v8);
      total += horizontal_sum_epi32(w32);
      p += 8;
      remaining -= 8;
    }
    for (; remaining > 0; --remaining) total += *p++;

    row += img_w;
  }
  return total;
}
#endif

static inline uint64_t sum_roi_u16(const uint16_t* img,
                                   int img_w,
                                   int x, int y, int w, int h) {
#if defined(__AVX2__)
  return sum_roi_avx2(img, img_w, x, y, w, h);
#else
  return sum_roi_scalar(img, img_w, x, y, w, h);
#endif
}

// ============================================================================
// `Impl` class
// Contains the internal implementation details for the `Processor` class,
// including worker threads, synchronization, and per-frame state.
// ============================================================================
struct Processor::Impl {
  /// Constructor for the `Impl` class
  /// @param pool_ The slab memory pool to use
  /// @param cfg_ The processor configuration
  /// @param pop_fn_ The function to use to pop descriptors from the slab memory
  /// @param rois_ The ROIs to process
  /// @param on_result_ The function to use to process the results
    Impl(slab::SlabPool& pool_, ProcessorConfig cfg_, PopFn pop_fn_,
         std::vector<ROI> rois_, ResultCallback on_result_)
     : pool(pool_), cfg(std::move(cfg_)), pop_fn(std::move(pop_fn_)), 
       rois(std::move(rois_)), on_result(std::move(on_result_)){}

  /// The slab memory pool to use
  slab::SlabPool&  pool;
  ProcessorConfig  cfg;
  PopFn            pop_fn;
  std::vector<ROI> rois;
  ResultCallback   on_result;

  /// Ingress + worker threads
  std::thread              ingress_th;
  std::vector<std::thread> workers;
  std::atomic<bool>        run{false};

  /// Shared per-frame state (one frame in flight)
  slab::Desc               cur_desc{};
  const uint16_t*          cur_img{nullptr};
  std::vector<uint64_t>    roi_sums;
  std::vector<uint8_t>     roi_occ;

  std::atomic<size_t>      next_roi{0};     // next ROI to process
  std::atomic<unsigned>    workers_left{0}; // workers left to finish frame
  std::atomic<uint64_t>    frame_epoch{0};  // increments for each new frame

  std::mutex               frame_mtx;         // frame_cv waits for frame done
  std::condition_variable  frame_cv;          // ingress waits for frame done
  std::mutex               start_mtx;         // start_cv waits for new frame
  std::condition_variable  start_cv;          // workers wait for new frame

  Stats                    stats_;

  void pin_thread() {
#if defined(__linux__)
    if (cfg.cpu_affinity >= 0) {
      cpu_set_t set; CPU_ZERO(&set); CPU_SET(cfg.cpu_affinity, &set);
      pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
    }
#endif
  }

  /// Release the slab for processor path
  void release_proc(const slab::Desc& d) {
    auto& b = pool.get(d.id);
    if (d.gen != b.hdr.gen) return;
    auto prev = b.hdr.pending.fetch_and(uint8_t(~slab::SlabPool::BIT_PROC),
                                        std::memory_order_acq_rel);
    if ((prev & ~slab::SlabPool::BIT_PROC) == 0) {
      ++b.hdr.gen;
      pool.release(d.id);
    }
  }

  /// Worker loop for the `Processor` class
  /// @param worker_id The ID of the worker
  void worker_loop(unsigned worker_id) {
    (void)worker_id;
    uint64_t seen_epoch = 0;

    while (run.load(std::memory_order_relaxed)) {
      // Wait for a new frame epoch
      {
        std::unique_lock lk(start_mtx);
        start_cv.wait(lk, [&]{
          return !run.load(std::memory_order_relaxed) ||
                 frame_epoch.load(std::memory_order_acquire) > seen_epoch;
        });
      }
      if (!run.load(std::memory_order_relaxed)) break;

      seen_epoch = frame_epoch.load(std::memory_order_acquire);
      const uint16_t* img = cur_img;
      const int img_w = static_cast<int>(cfg.image_w);

      // Work-stealing over ROI indices
      for (;;) {
        size_t idx = next_roi.fetch_add(1, std::memory_order_relaxed);
        if (idx >= rois.size()) break;

        const ROI& r = rois[idx];
        uint64_t s = sum_roi_u16(img, img_w, r.x, r.y, r.w, r.h);
        roi_sums[idx] = s;
        roi_occ[idx]  = (s >= r.threshold) ? 1u : 0u;
      }

      // Signal completion for this worker
      if (workers_left.fetch_sub(1, std::memory_order_acq_rel) == 1u) {
        std::lock_guard lk(frame_mtx);
        frame_cv.notify_one();
      }
    }
  }

  /// Ingress loop for the `Processor` class
  void ingress_loop() {
    pin_thread();

    const int W = static_cast<int>(cfg.image_w);
    const int H = static_cast<int>(cfg.image_h);
    const std::size_t fb = pool.frame_bytes();
    if (fb != std::size_t(W) * std::size_t(H) * 2u) {
      throw std::runtime_error("Processor: frame_bytes does not match image");
    }

    roi_sums.assign(rois.size(), 0);
    roi_occ.assign(rois.size(), 0);

    slab::Desc d{};
    while (run.load(std::memory_order_relaxed)) {
      if (!pop_fn || !pop_fn(d)) {
        stats_.empty_polls.fetch_add(1, std::memory_order_relaxed);
        std::this_thread::yield();
        continue;
      }

      auto& buf = pool.get(d.id);
      if (d.gen != buf.hdr.gen) {
        stats_.stale_desc.fetch_add(1, std::memory_order_relaxed);
        release_proc(d);
        continue;
      }

      // Set current frame state
      cur_desc = d;
      cur_img  = reinterpret_cast<const uint16_t*>(buf.data);
      next_roi.store(0, std::memory_order_relaxed);
      workers_left.store(static_cast<unsigned>(workers.size()), 
                         std::memory_order_release);

      /// Kick workers with new epoch
      frame_epoch.fetch_add(1, std::memory_order_acq_rel);
      {
        std::lock_guard lk(start_mtx);
        start_cv.notify_all();
      }

      /// Wait for all workers to finish this frame
      {
        std::unique_lock lk(frame_mtx);
        frame_cv.wait(lk, [&]{
          return workers_left.load(std::memory_order_acquire) == 0 ||
                 !run.load(std::memory_order_relaxed);
        });
      }
      if (!run.load(std::memory_order_relaxed)) break;

      /// Results ready: callback or print
      bool keep_running = true;
      if (on_result) {
        keep_running = on_result(d.seq, roi_sums, roi_occ);
      } else if (cfg.print_stdout) {
        uint64_t occ_count = 0;
        for (auto v : roi_occ) occ_count += v;
        std::printf("seq=%llu rois=%zu occupied=%llu\n",
                    (unsigned long long)d.seq,
                    rois.size(),
                    (unsigned long long)occ_count);
      }

      /// Release the slab for processor path
      release_proc(d);
      stats_.frames.fetch_add(1, std::memory_order_relaxed);

      if (!keep_running) break;
    }
  }
};

// ============================================================================
// `Processor` class
// AVX2-based processor with a tiled worker pool over ROIs.
// ============================================================================
Processor::Processor(slab::SlabPool& pool,
    ProcessorConfig cfg,
    PopFn pop_fn,
    std::vector<ROI> rois,
    ResultCallback on_result)
: impl_(std::make_unique<Impl>(pool,
              std::move(cfg),
              std::move(pop_fn),
              std::move(rois),
              std::move(on_result))) {}

Processor::~Processor() { stop(); }

void Processor::start() {
  if (impl_->run.exchange(true)) return;

  unsigned n = impl_->cfg.worker_threads;
  if (n == 0) {
    n = std::max(1u, std::thread::hardware_concurrency());
  }
  impl_->workers.clear();
  impl_->workers.reserve(n);
  for (unsigned i = 0; i < n; ++i) {
    impl_->workers.emplace_back([this, i]{ impl_->worker_loop(i); });
  }
  impl_->ingress_th = std::thread([this]{ impl_->ingress_loop(); });
}

void Processor::stop() {
  if (!impl_->run.exchange(false)) return;
  {
    std::lock_guard lk(impl_->start_mtx);
    impl_->start_cv.notify_all();
  }
  {
    std::lock_guard lk(impl_->frame_mtx);
    impl_->frame_cv.notify_all();
  }
  if (impl_->ingress_th.joinable()) impl_->ingress_th.join();
  for (auto& t : impl_->workers) {
    if (t.joinable()) t.join();
  }
  impl_->workers.clear();
}

const Processor::Stats& Processor::stats() const noexcept {
  return impl_->stats_;
}

} // namespace pipe
