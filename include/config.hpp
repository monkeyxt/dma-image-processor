// ============================================================================
// config.hpp -- Configuration structure for DMA Image Processor
//
// This header defines the Config struct that holds all configuration values
// for the pipeline, which can be loaded from a TOML configuration file.
// ============================================================================
#pragma once
#include <cstddef>
#include <cstdint>
#include <string>
#include <stdexcept>
#include <type_traits>

// ============================================================================
// Configuration structure
// ============================================================================
struct Config {
    // ========================================================================
    // Pipeline geometry / config
    // ========================================================================
    uint32_t IMAGE_W        { 300 };
    uint32_t IMAGE_H        { 300 };
    uint32_t SLABS          { 128 };
    int      FPS            { 1000 };      // 1 kFPS target
    uint32_t NTH_ARCHIVE    { 10 };        // every 10th frame archived
    bool     DRAIN_QUEUES   { false };     // drain queues after processing
    
    // ========================================================================
    // ROI grid configuration
    // ========================================================================
    std::size_t  ROW_START     { 50 };     // start row of ROI grid
    std::size_t  COL_START     { 50 };     // start column of ROI grid
    std::size_t  ROW_END       { 250 };    // end row of ROI grid
    std::size_t  COL_END       { 250 };    // end column of ROI grid
    std::size_t  ROI_W         { 10 };     // ROI width
    std::size_t  ROI_H         { 10 };     // ROI height
    double       LAMBDA_OCC    { 10.0 };   // Poisson distribution for occupied ROI
    double       LAMBDA_EMPTY  { 0.5 };    // Poisson distribution for empty ROI
    double       FP_TARGET     { 1e-3 };   // False positive rate
    int          MAX_K         { 100 };    // Maximum scan range
    
    // ========================================================================
    // SPSC queue configuration
    // Note: These values are used as template parameters for Ring queues.
    // Template parameters must be compile-time constants, so these defaults
    // are used.
    // ========================================================================
    static constexpr uint32_t PROC_Q_SIZE_DEFAULT { 4096 };
    static constexpr uint32_t ARCH_Q_SIZE_DEFAULT { 4096 };

    // ========================================================================
    // DMA source configuration
    // The DMA source will pace the frames to the rate. If the rate is not paced
    // then the frames will be delivered as fast as possible.
    // ========================================================================
    bool      PACE_FRAMES   { true };
    int       CPU_AFFINITY  { -1 };     // no pinning
    int       THREAD_NICE   { 0 };      // no niceness
    bool      LOOP_FRAMES   { true };   // loop through frames
    
    // ========================================================================
    // Processor configuration
    // ========================================================================
    uint32_t  WORKER_THREADS { 4 };   // adjust per CPU
    bool      PRINT_STDOUT   { false };
    uint64_t  INTERVAL_PRINT { 37 };  // random prime number for interval printing
    std::string PROC_OUTPUT_DIR    { "PROC_out" };
    
    // ========================================================================
    // Archiver configuration
    // ========================================================================
    uint64_t    SEGMENT_BYTES   { 4ull * 1024 * 1024 * 1024 }; // 4 GiB segments
    uint64_t    IO_BUFFER_BYTES { 8ull * 1024 * 1024 };
    bool        MAKE_INDEX      { true };
    bool        USE_IO_URING    { false };   // set true on Linux with io_uring if desired
    bool        DIRECT_IO       { false };   // open with O_DIRECT (requires aligned writes)
    unsigned    URING_QD        { 64 };      // SQ/CQ depth
    unsigned    MAX_INFLIGHT    { 32 };      // cap in-flight write requests
    std::size_t BLOCK_BYTES     { 4096 };    // alignment & size multiple when using O_DIRECT
    
    std::string ARCH_OUTPUT_DIR    { "ARCH_out" };
    std::string ARCH_OUTPUT_PREFIX = "frames";
    
    // ========================================================================
    // Derived/computed values
    // ========================================================================
    std::size_t FRAME_BYTES() const { return std::size_t(IMAGE_W) * IMAGE_H * 2; }
    int FRAME_PERIOD() const { return 1000000 / FPS; }

};

// ============================================================================
// Load configuration from TOML file
// ============================================================================
Config load_config(const std::string& config_path);


