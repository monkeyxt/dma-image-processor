# DMA-Image-Processor

`DMA-Image-Processor` is a high-performance C++ tool designed to efficiently process and archive streams of image data from DMA-capable sources. It features a memory slab pool, multi-threaded processing pipeline, and optional indexed archiving. 


## Requirements
```
c++20
clang >= 14.0
cmake >= 3.15
```

## Build & Run
```
cmake .
make
./dma-image-processor <npy_dir> [run_seconds]
```

## Architecture
```
┌─────────────┐    ┌──────┐┌──────┐  ┌────┐  ┌────┐
│dma:slab_pool│    │proc_q││arch_q│  │proc│  │arch│
└──────┬──────┘    └──┬───┘└──┬───┘  └─┬──┘  └─┬──┘
       │              │       │        │       │   
       │ProcCallback()│       │        │       │   
       │─────────────>│       │        │       │   
       │              │       │        │       │   
       │ paced @1kfps │       │        │       │   
       │─────────────>│       │        │       │   
       │              │       │        │       │   
       │    ArchCallback()    │        │       │   
       │─────────────────────>│        │       │   
       │              │       │        │       │   
       │    paced @200fps     │        │       │   
       │─────────────────────>│        │       │   
       │              │       │        │       │   
       │              │PopFn proc_pop()│       │   
       │              │───────────────>│       │   
       │              │       │        │       │   
       │              │       │PopFn arch_pop()│   
       │              │       │───────────────>│   
       │              │       │        │       │   
       │        release_proc()│        │       │   
       │<──────────────────────────────│       │   
       │              │       │        │       │   
       │            release_arch()     │       │   
       │<──────────────────────────────────────│   
┌──────┴──────┐    ┌──┴───┐┌──┴───┐  ┌─┴──┐  ┌─┴──┐
│dma:slab_pool│    │proc_q││arch_q│  │proc│  │arch│
└─────────────┘    └──────┘└──────┘  └────┘  └────┘

```

The system is designed as a high-throughput, lock-free, multi-threaded frame pipeline for DMA-capable imaging applications. Its architecture is modular, separating responsibility between memory management, frame sequencing, processing, and archiving.

### DMA Source
- Ingests frames from external sources (hardware, files, memory)
- Supports "paced" mode to control frame rate
- Handles both processing and archiving paths via callbacks

### Slab Pool
- Manages a pool of fixed-size, aligned frame buffers ("slabs")
- Provides lock-free allocation and reclamation
- Allows multiple consumers to share slabs without blocking

### Processor
- Processes data from a lock-free `spsc_ring` queue
- Optimized for modern CPUs with SIMD instructions (AVX2/AVX512)
- Supports efficient multithreading

### Archiver
- Archives frames from a lock-free `spsc_ring` queue
- Supports Linux zero-copy `io_uring` interface for asynchronous, low-overhead disk writes
- Configurable to archive every Nth frame for reduced I/O and storage needs
