// ============================================================================
// config.cpp -- Configuration loading implementation
//
// This file implements the load_config function that reads configuration
// values from a TOML file and returns a Config struct.
// ============================================================================
#include "config.hpp"
#include <toml++/toml.hpp>
#include <cstdio>
#include <stdexcept>

// ============================================================================
// Load configuration from TOML file
// ============================================================================
Config load_config(const std::string& config_path) {
    Config cfg;
    
    try {
        auto tbl = toml::parse_file(config_path);
        
        // Main config
        if (auto v = tbl["main"]["IMAGE_W"].value<uint32_t>()) cfg.IMAGE_W = *v;
        if (auto v = tbl["main"]["IMAGE_H"].value<uint32_t>()) cfg.IMAGE_H = *v;
        if (auto v = tbl["main"]["SLABS"].value<uint32_t>()) cfg.SLABS = *v;
        if (auto v = tbl["main"]["FPS"].value<int>()) cfg.FPS = *v;
        if (auto v = tbl["main"]["NTH_ARCHIVE"].value<uint32_t>()) cfg.NTH_ARCHIVE = *v;
        if (auto v = tbl["main"]["DRAIN_QUEUES"].value<bool>()) cfg.DRAIN_QUEUES = *v;
        
        // Poisson/ROI config
        if (auto v = tbl["pois"]["ROW_START"].value<std::size_t>()) cfg.ROW_START = *v;
        if (auto v = tbl["pois"]["COL_START"].value<std::size_t>()) cfg.COL_START = *v;
        if (auto v = tbl["pois"]["ROW_END"].value<std::size_t>()) cfg.ROW_END = *v;
        if (auto v = tbl["pois"]["COL_END"].value<std::size_t>()) cfg.COL_END = *v;
        if (auto v = tbl["pois"]["ROI_W"].value<std::size_t>()) cfg.ROI_W = *v;
        if (auto v = tbl["pois"]["ROI_H"].value<std::size_t>()) cfg.ROI_H = *v;
        if (auto v = tbl["pois"]["LAMBDA_OCC"].value<double>()) cfg.LAMBDA_OCC = *v;
        if (auto v = tbl["pois"]["LAMBDA_EMPTY"].value<double>()) cfg.LAMBDA_EMPTY = *v;
        if (auto v = tbl["pois"]["FP_TARGET"].value<double>()) cfg.FP_TARGET = *v;
        if (auto v = tbl["pois"]["MAX_K"].value<int>()) cfg.MAX_K = *v;
        
        // DMA config
        if (auto v = tbl["dma"]["PACE_FRAMES"].value<bool>()) cfg.PACE_FRAMES = *v;
        if (auto v = tbl["dma"]["CPU_AFFINITY"].value<int>()) cfg.CPU_AFFINITY = *v;
        if (auto v = tbl["dma"]["THREAD_NICE"].value<int>()) cfg.THREAD_NICE = *v;
        if (auto v = tbl["dma"]["LOOP_FRAMES"].value<bool>()) cfg.LOOP_FRAMES = *v;
        
        // Processor config
        if (auto v = tbl["proc"]["WORKER_THREADS"].value<uint32_t>()) cfg.WORKER_THREADS = *v;
        if (auto v = tbl["proc"]["PRINT_STDOUT"].value<bool>()) cfg.PRINT_STDOUT = *v;
        if (auto v = tbl["proc"]["INTERVAL_PRINT"].value<uint64_t>()) cfg.INTERVAL_PRINT = *v;
        if (auto v = tbl["proc"]["PROC_OUTPUT_DIR"].value<std::string>()) cfg.PROC_OUTPUT_DIR = *v;
        
        // Archiver config
        if (auto v = tbl["arch"]["SEGMENT_BYTES"].value<uint64_t>()) cfg.SEGMENT_BYTES = *v;
        if (auto v = tbl["arch"]["IO_BUFFER_BYTES"].value<uint64_t>()) cfg.IO_BUFFER_BYTES = *v;
        if (auto v = tbl["arch"]["MAKE_INDEX"].value<bool>()) cfg.MAKE_INDEX = *v;
        if (auto v = tbl["arch"]["USE_IO_URING"].value<bool>()) cfg.USE_IO_URING = *v;
        if (auto v = tbl["arch"]["DIRECT_IO"].value<bool>()) cfg.DIRECT_IO = *v;
        if (auto v = tbl["arch"]["URING_QD"].value<unsigned>()) cfg.URING_QD = *v;
        if (auto v = tbl["arch"]["MAX_INFLIGHT"].value<unsigned>()) cfg.MAX_INFLIGHT = *v;
        if (auto v = tbl["arch"]["BLOCK_BYTES"].value<std::size_t>()) cfg.BLOCK_BYTES = *v;
        if (auto v = tbl["arch"]["ARCH_OUTPUT_DIR"].value<std::string>()) cfg.ARCH_OUTPUT_DIR = *v;
        if (auto v = tbl["arch"]["ARCH_OUTPUT_PREFIX"].value<std::string>()) cfg.ARCH_OUTPUT_PREFIX = *v;
        
        std::printf("Loaded configuration from: %s\n", config_path.c_str());
    } catch (const toml::parse_error& err) {
        std::fprintf(stderr, "Error parsing config file '%s': %s\n", 
                     config_path.c_str(), err.description().data());
        std::fprintf(stderr, "Using default configuration values.\n");
    } catch (const std::exception& err) {
        std::fprintf(stderr, "Error loading config file '%s': %s\n", 
                     config_path.c_str(), err.what());
        std::fprintf(stderr, "Using default configuration values.\n");
    }
    
    return cfg;
}

