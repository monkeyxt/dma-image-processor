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
mkdir -p build
cd build
cmake ..
make
./bin/dma-image-processor <npy_dir> [run_seconds] [--config config.toml]
```
The `<npy_dir>` specifies where the input images are. The `--config` flag specifies the location of the config file. If no config file is specified, default configurations will be used.
See the `config/README.md` about what each config parameter does and also `config/default.toml` which gives the default params for the program. 

To build without tests:
```
cmake .. -DBUILD_TESTS=OFF
make
```

For `io_uring` to work, the following packages must be installed
```
sudo apt-get update
sudo apt-get install -y build-essential cmake liburing-dev
```
## Scripts
There a couple of useful scripts in the `utils` folder. 
- `decode.py`: this python script decodes the output `.bin` files into `.npy` files. 
- `image.py`: turns `.npy` into image files.
- `npydiff.py`: diffs two `.npy` files.

## Architecture
```
 ┌──────────────┐     ┌─────────────────┐┌─────────────────┐┌─────────┐┌────────┐
 │dma(slab_pool)│     │proc_q(spsc_ring)││arch_q(spsc_ring)││processor││archiver│
 └──────┬───────┘     └────────┬────────┘└────────┬────────┘└────┬────┘└───┬────┘
        │                      │                  │              │         │     
        │    ProcCallback()    │                  │              │         │     
        │─────────────────────>│                  │              │         │     
        │                      │                  │              │         │     
        │paced @1kfps (default)│                  │              │         │     
        │─────────────────────>│                  │              │         │     
        │                      │                  │              │         │     
        │             ArchCallback()              │              │         │     
        │────────────────────────────────────────>│              │         │     
        │                      │                  │              │         │     
        │         paced @200fps (default)         │              │         │     
        │────────────────────────────────────────>│              │         │     
        │                      │                  │              │         │     
        │                      │      process -- proc_pop()      │         │     
        │                      │────────────────────────────────>│         │     
        │                      │                  │              │         │     
        │                      │                  │  write -- arch_pop()   │     
        │                      │                  │───────────────────────>│     
        │                      │                  │              │         │     
        │             release slab -- release_proc()             │         │     
        │<───────────────────────────────────────────────────────│         │     
        │                      │                  │              │         │     
        │                  release slab -- release_arch()        │         │     
        │<─────────────────────────────────────────────────────────────────│     
 ┌──────┴───────┐     ┌────────┴────────┐┌────────┴────────┐┌────┴────┐┌───┴────┐
 │dma(slab_pool)│     │proc_q(spsc_ring)││arch_q(spsc_ring)││processor││archiver│
 └──────────────┘     └─────────────────┘└─────────────────┘└─────────┘└────────┘
```
## Metrics

The pipeline collects detailed runtime metrics to allow monitoring and analysis of system performance and throughput. One can find the metrics code in `src/metrics.cpp` (class `PipelineMetrics`). Below is a summary table of the key metrics available:

| Metric                  | Component   | Description                                                                 |
|-------------------------|-------------|-----------------------------------------------------------------------------|
| `dma_frames`            | DMA         | Number of frames ingested (produced) by DMA thread                          |
| `dma_arch_offered`      | DMA         | Frames passed from DMA to archiving                                         |
| `dma_pool_exhaust`      | DMA         | Times slab pool was exhausted (potential backpressure on DMA)               |
| `proc_frames`           | Processor   | Number of frames processed                                                  |
| `proc_stale_desc`       | Processor   | Number of stale descriptors dropped at processor                            |
| `proc_empty_polls`      | Processor   | Empty polls on processor queue                                              |
| `drops_proc_queue`      | Processor   | Frames dropped due to full processor queue                                  |
| `arch_frames`           | Archiver    | Number of frames successfully archived                                      |
| `arch_bytes`            | Archiver    | Total bytes written to archive(s)                                           |
| `arch_rotations`        | Archiver    | Number of archive file rotations (due to segment size)                      |
| `arch_stale_desc`       | Archiver    | Stale descriptors dropped at archiver                                       |
| `arch_io_errors`        | Archiver    | I/O errors encountered during archive writes                                |
| `drops_arch_queue`      | Archiver    | Frames dropped due to full archiver queue                                   |
| `elapsed_sec`           | Overall     | Total elapsed time for metrics snapshot (seconds)                           |
| `dma_fps`               | DMA         | DMA throughput (frames per second)                                          |
| `proc_fps`              | Processor   | Processor throughput (frames per second)                                    |
| `arch_fps`              | Archiver    | Archiver throughput (frames per second)                                     |
| `arch_gbps`             | Archiver    | Archiver write throughput (gigabytes per second, GB/s)                      |
| `arch_gibps`            | Archiver    | Archiver write throughput (gibibytes per second, GiB/s)                     |

### Example Output

```
=== Component Stats ===
DMA:    produced=1000000  archived_offered=998450  pool_exhaust=12
PROC:   frames=998400  stale_desc=5  empty_polls=100 dropped_proc_queue=2
ARCH:   frames=998350  bytes=179910000  rotations=2  stale_desc=1  io_errors=0 dropped_arch_queue=0

=== End-to-end Throughput ===
Elapsed: 10.000 s
DMA:     100000.0 fps
PROC:    99840.0 fps
ARCH:    99835.0 fps | 17.991 GB/s (16.756 GiB/s)
```
