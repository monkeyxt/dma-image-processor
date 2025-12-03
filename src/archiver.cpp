// ============================================================================
// archiver.cpp -- implementation of the Archiver class
// ============================================================================
#include <cstdio>
#include <filesystem>
#include <stdexcept>
#include <vector>
#include <cstring>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include "archiver.hpp"


#if defined(__linux__) && !defined(PIPE_DISABLE_URING)
  #if __has_include(<liburing.h>)
    #include <liburing.h>
    #define PIPE_HAS_URING 1
  #else
    #define PIPE_HAS_URING 0
  #endif
#else
  #define PIPE_HAS_URING 0
#endif

#if defined(__linux__)
  #include <pthread.h>
#endif

namespace fs = std::filesystem;

namespace pipeline {

/// Constructs the file path for a segment binary file given the configuration 
/// and segment index. The resulting filename is of the form 
/// "<output_dir>/<file_prefix>_<seg_idx>.bin", where seg_idx is zero-padded 
/// to 6 digits.
static std::string make_segment_name(const ArchiverConfig& cfg, 
                                     uint64_t seg_idx) {
  char buf[512];
  std::snprintf(buf, sizeof(buf), "%s_%06llu.bin",
                cfg.file_prefix.c_str(),
                static_cast<unsigned long long>(seg_idx));
  return (fs::path(cfg.output_dir) / buf).string();
}

/// Constructs the file path for an index file corresponding to a segment, 
/// given the configuration and segment index. The resulting filename is of 
/// the form "<output_dir>/<file_prefix>_<seg_idx>.idx", where seg_idx is 
/// zero-padded to 6 digits.
static std::string make_index_name(const ArchiverConfig& cfg, 
                                   uint64_t seg_idx) {
  char buf[512];
  std::snprintf(buf, sizeof(buf), "%s_%06llu.idx",
                cfg.file_prefix.c_str(),
                static_cast<unsigned long long>(seg_idx));
  return (fs::path(cfg.output_dir) / buf).string();
}

// ============================================================================
// Impl: implementation of the Archiver class
// ============================================================================
struct Archiver::Impl {
  slab::SlabPool&  pool;
  ArchiverConfig   cfg;
  PopFn            pop_fn;

  /// Thread for the archiver loop
  std::thread       th;
  std::atomic<bool> run{false};

  /// Statistics for the archiver
  Stats             stats_;

  /// Segment index for the current segment
  uint64_t          seg_idx{0};
  /// Number of bytes written to the current segment
  std::size_t       seg_bytes_written{0};

  /// File pointer for the current segment binary file
  std::FILE*        seg_fp{nullptr};
  /// File pointer for the current segment index file
  std::FILE*        idx_fp{nullptr};
  /// Buffer for the stdio backend
  std::vector<char> stdio_buf;

  /// io_uring queue for the uring backend
#if PIPE_HAS_URING
  /// io_uring queue for the uring backend
  io_uring          ring{};
  /// File descriptor for the current segment binary file
  int               fd_data{-1};
  int               fd_idx {-1};
  unsigned          inflight{0};
#endif

  /// Ensure the output directory exists
  void ensure_dir() {
    std::error_code ec;
    if (!fs::exists(cfg.output_dir, ec)) {
      fs::create_directories(cfg.output_dir, ec);
      if (ec) {
        throw std::runtime_error(
          "Archiver: failed to create output dir: " + ec.message());
      }
    }
  }

  /// Pin the thread to a specific CPU
  void pin_thread() {
#if defined(__linux__)
    if (cfg.cpu_affinity >= 0) {
      cpu_set_t set; CPU_ZERO(&set); CPU_SET(cfg.cpu_affinity, &set);
      pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
    }
#endif
  }

  /// Open the current segment binary file for writing, and the index file if 
  /// requested. This is the fallback function for writing data to the current 
  /// segment binary file without io_uring.
  bool open_segment_stdio() {
    close_segment_stdio();
    const auto path = make_segment_name(cfg, seg_idx);
    seg_fp = std::fopen(path.c_str(), "wb");
    if (!seg_fp) return false;
    if (cfg.io_buffer_bytes > 0) {
      stdio_buf.resize(cfg.io_buffer_bytes);
      std::setvbuf(seg_fp, stdio_buf.data(), _IOFBF, stdio_buf.size());
    }
    if (cfg.make_index) {
      const auto ipath = make_index_name(cfg, seg_idx);
      idx_fp = std::fopen(ipath.c_str(), "wb");
      if (!idx_fp) { std::fclose(seg_fp); seg_fp=nullptr; return false; }
      std::setvbuf(idx_fp, nullptr, _IOLBF, 0);
    }
    seg_bytes_written = 0;
    stats_.rotations.fetch_add(1, std::memory_order_relaxed);
    return true;
  }

  /// Close the current segment binary file and index file without io_uring.
  /// This is the close function for writing data to the current segment binary 
  /// file without io_uring.
  void close_segment_stdio() {
    if (seg_fp) { std::fflush(seg_fp); std::fclose(seg_fp); seg_fp=nullptr; }
    if (idx_fp) { std::fflush(idx_fp); std::fclose(idx_fp); idx_fp=nullptr; }
  }

#if PIPE_HAS_URING
  /// Open the current segment binary file for writing, and the index file if 
  /// requested with io_uring. This is the open function for writing data to the 
  /// current segment binary file with io_uring.
  bool open_segment_uring() {
    close_segment_uring();
    const auto path = make_segment_name(cfg, seg_idx);

    int flags = O_CREAT | O_WRONLY | O_TRUNC;
    int mode  = 0644;
    if (cfg.direct_io) flags |= O_DIRECT;

    fd_data = ::open(path.c_str(), flags, mode);
    if (fd_data < 0) return false;

    if (cfg.make_index) {
      const auto ipath = make_index_name(cfg, seg_idx);
      fd_idx = ::open(ipath.c_str(), O_CREAT | O_WRONLY | O_TRUNC, 0644);
      if (fd_idx < 0) { ::close(fd_data); fd_data=-1; return false; }
    }

    if (io_uring_queue_init(cfg.uring_qd, &ring, 0) != 0) {
      if (fd_idx >= 0) ::close(fd_idx);
      ::close(fd_data); fd_data = -1; fd_idx = -1;
      return false;
    }

    seg_bytes_written = 0;
    inflight = 0;
    stats_.rotations.fetch_add(1, std::memory_order_relaxed);
    return true;
  }

  /// Close the current segment binary file and index file with io_uring.
  /// This is the close function for writing data to the current segment binary 
  /// file with io_uring.
  void close_segment_uring() {
    /// Drain completions
    if (fd_data >= 0) {
      io_uring_cqe* cqe;
      while (inflight > 0) {
        if (io_uring_wait_cqe(&ring, &cqe) == 0) {
          if (cqe->res < 0) {
            stats_.io_errors.fetch_add(1, std::memory_order_relaxed);
          }
          io_uring_cqe_seen(&ring, cqe);
          --inflight;
          stats_.uring_completions.fetch_add(1, std::memory_order_relaxed);
        }
      }
    }
    if (ring.ring_fd > 0) {
      io_uring_queue_exit(&ring);
      memset(&ring, 0, sizeof(ring));
    }
    if (fd_idx >= 0) { ::close(fd_idx); fd_idx = -1; }
    if (fd_data >= 0) { ::close(fd_data); fd_data = -1; }
  }

  /// Enqueue a write; waits for a completion if queue/inflight are full
  /// This is the main function for writing data to the current segment binary 
  /// file with io_uring.
  /// @param data       The data to write
  /// @param len        The length of the data to write
  /// @param offset     The offset to write the data to
  /// @return           True if the write was successful, false otherwise
  bool uring_write(const void* data, std::size_t len, off_t offset) {
    /// enforce O_DIRECT alignment if requested
    if (cfg.direct_io) {
      const std::size_t bl = cfg.block_bytes ? cfg.block_bytes : 4096;
      if ((reinterpret_cast<uintptr_t>(data) % bl) != 0 
          || (len % bl) != 0 || (offset % bl) != 0) {
        /// caller must ensure slab_pool uses aligned allocations sized to 
        /// block_bytes
        stats_.io_errors.fetch_add(1, std::memory_order_relaxed);
        return false;
      }
    }

    for (;;) {

      /// Inflight here refers to the number of I/O write operations that have 
      /// been submitted to io_uring but have not yet completed. If the number 
      /// of inflight requests reaches the configured maximum, we wait for at 
      /// least one I/O completion before submitting another write.
      if (inflight >= cfg.max_inflight) {
        /// Too many in-flight requests: wait for at least one to complete 
        /// before continuing
        io_uring_cqe* cqe;
        if (io_uring_wait_cqe(&ring, &cqe) == 0) {
          if (cqe->res < 0) {
            stats_.io_errors.fetch_add(1, std::memory_order_relaxed);
          }
          io_uring_cqe_seen(&ring, cqe);
          --inflight;
          stats_.uring_completions.fetch_add(1, std::memory_order_relaxed);
        }
        continue;
      }
      /// Get a new submission entry from the io_uring queue.
      io_uring_sqe* sqe = io_uring_get_sqe(&ring);
      if (!sqe) {
        stats_.uring_sq_full.fetch_add(1, std::memory_order_relaxed);
        io_uring_submit(&ring);
        continue;
      }

      io_uring_prep_write(sqe, fd_data, data, static_cast<unsigned>(len), 
                          offset);
      /// Submit the write request.
      int ret = io_uring_submit(&ring);
      if (ret < 0) {
        stats_.io_errors.fetch_add(1, std::memory_order_relaxed);
        return false;
      }
      ++inflight;
      stats_.uring_submits.fetch_add(1, std::memory_order_relaxed);
      return true;
    }
  }

  /// Write the index file with io_uring.
  bool uring_write_index(uint64_t seq, uint64_t off) {
    if (!cfg.make_index) return true;
    char line[64];
    int  n = std::snprintf(line, sizeof(line), "%llu %llu\n",
                           (unsigned long long)seq, (unsigned long long)off);
    ssize_t wr = ::write(fd_idx, line, n);
    return wr == n;
  }
#endif

  /// Main loop for the archiver
  void loop() {
    pin_thread();
    const bool use_uring = (cfg.use_io_uring && PIPE_HAS_URING);

    /// Open first segment lazily on first write
    slab::Desc d{};
    while (run.load(std::memory_order_relaxed)) {
      if (!pop_fn || !pop_fn(d)) {
        stats_.empty_polls.fetch_add(1, std::memory_order_relaxed);
        std::this_thread::yield();
        continue;
      }

      auto& b = pool.get(d.id);
      const std::size_t fb = pool.frame_bytes();

      if (d.gen != b.hdr.gen) {
        stats_.stale_desc.fetch_add(1, std::memory_order_relaxed);
        release_arch(d);
        continue;
      }

      if (!use_uring) {
        if (!seg_fp || (seg_bytes_written + fb > cfg.segment_bytes)) {
          if (!open_segment_stdio()) { 
            stats_.io_errors.fetch_add(1, std::memory_order_relaxed); 
            release_arch(d); continue; 
          }
          ++seg_idx;
        }
        const auto offset_before = seg_bytes_written;
        const auto n = std::fwrite(b.data, 1, fb, seg_fp);
        if (n != fb) {
          stats_.io_errors.fetch_add(1, std::memory_order_relaxed);
          close_segment_stdio();
        } else {
          seg_bytes_written += n;
          stats_.archived_frames.fetch_add(1, std::memory_order_relaxed);
          stats_.archived_bytes .fetch_add(n, std::memory_order_relaxed);
          if (idx_fp && cfg.make_index) {
            std::fprintf(idx_fp, "%llu %llu\n",
                         (unsigned long long)d.seq,
                         (unsigned long long)offset_before);
          }
        }
      } else {
#if PIPE_HAS_URING
        /// rotate if needed
        if (fd_data < 0 || (seg_bytes_written + fb > cfg.segment_bytes)) {
          if (!open_segment_uring()) { 
            stats_.io_errors.fetch_add(1, std::memory_order_relaxed); 
                                       release_arch(d); 
            continue; 
          }
          ++seg_idx;
        }
        const off_t off = static_cast<off_t>(seg_bytes_written);
        if (!uring_write(b.data, fb, off)) {
          close_segment_uring();
        } else {
          seg_bytes_written += fb;
          stats_.archived_frames.fetch_add(1, std::memory_order_relaxed);
          stats_.archived_bytes .fetch_add(fb, std::memory_order_relaxed);
          if (!uring_write_index(d.seq, off)) {
            stats_.io_errors.fetch_add(1, std::memory_order_relaxed);
          }
        }
#else
        /// should not happen; compiled without uring support
        (void)use_uring;
#endif
      }
      /// Release ARCH bit (and maybe return slab) regardless of I/O outcome
      release_arch(d);
    }
    if (!use_uring) {
      close_segment_stdio();
    } else {
#if PIPE_HAS_URING
      close_segment_uring();
#endif
    }
  }

  /// Clear ARCH bit and maybe return to pool
  void release_arch(const slab::Desc& d) {
    auto& buf = pool.get(d.id);
    if (d.gen != buf.hdr.gen) return;
    auto prev = buf.hdr.pending.fetch_and(uint8_t(~slab::SlabPool::BIT_ARCH),
                                          std::memory_order_acq_rel);
    if ((prev & ~slab::SlabPool::BIT_ARCH) == 0) {
      ++buf.hdr.gen;
      pool.release(d.id);
    }
  }
};

// ============================================================================'
// `Archiver` class
// ============================================================================
Archiver::Archiver(slab::SlabPool& pool, ArchiverConfig cfg, PopFn pop_fn)
: impl_(std::unique_ptr<Impl>(new Impl{
    pool, std::move(cfg), std::move(pop_fn), {}, false, {}, 0,  0,  nullptr,
    nullptr, {}
#if PIPE_HAS_URING
    , {}, -1, -1, 0  // ring, fd_data, fd_idx, inflight
#endif
})) {}

Archiver::~Archiver() { stop(); }

void Archiver::start() {
  if (impl_->run.exchange(true)) return;
  
  // Ensure output directory exists (may create timestamped subdirectory)
  impl_->ensure_dir();
  
  // Announce Archiver configuration when starting
  std::printf("ARCH: Running with config:\n");

  std::printf("  segment_bytes=%llu (%.2f GiB)\n", 
              (unsigned long long)impl_->cfg.segment_bytes,
              impl_->cfg.segment_bytes / (1024.0 * 1024.0 * 1024.0));
  std::printf("  io_buffer_bytes=%llu (%.2f MiB)\n",
              (unsigned long long)impl_->cfg.io_buffer_bytes,
              impl_->cfg.io_buffer_bytes / (1024.0 * 1024.0));
  std::printf("  make_index=%s\n", impl_->cfg.make_index ? "true" : "false");
  std::printf("  use_io_uring=%s\n", impl_->cfg.use_io_uring ? "true" : "false");
  if (impl_->cfg.use_io_uring) {
    std::printf("  uring_qd=%u  max_inflight=%u\n", 
                impl_->cfg.uring_qd, impl_->cfg.max_inflight);
  }
  std::printf("  direct_io=%s\n", impl_->cfg.direct_io ? "true" : "false");
  std::printf("  output_dir=%s\n", impl_->cfg.output_dir.c_str());
  std::printf("  file_prefix=%s\n", impl_->cfg.file_prefix.c_str());
  
  impl_->th = std::thread([this]{ impl_->loop(); });
}

void Archiver::stop() {
  if (!impl_->run.exchange(false)) return;
  if (impl_->th.joinable()) impl_->th.join();
}

const Archiver::Stats& Archiver::stats() const noexcept {
  return impl_->stats_;
}

} // namespace pipeline
