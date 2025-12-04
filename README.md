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
./dma-image-processor <npy_dir> [run_seconds] [--config config.toml]
```

For `io_uring` to work, the following packages must be installed
```
sudo apt-get update
sudo apt-get install -y build-essential cmake liburing-dev
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
