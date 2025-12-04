# Config Parameters

## Overview

The configuration files for `DMA-Image-Processor` use TOML format and are divided into several sections. Each section controls a logical aspect of the system, such as DMA source, processing pipeline, or output archiving. Below are the available sections and their commonly available parameters.


## `[main]` : Main Application Parameters

| Parameter       | Type    | Description                                                        |
|-----------------|---------|--------------------------------------------------------------------|
| IMAGE_W         | int     | Image width in pixels.                                             |
| IMAGE_H         | int     | Image height in pixels.                                            |
| FRAME_BYTES     | int     | Total bytes per frame (should match IMAGE_W x IMAGE_H x pixel size).|
| SLABS           | int     | Number of slabs in the DMA slab pool. Controls in-flight frames.   |
| FPS             | int     | Source frame rate in frames per second.                            |
| NTH_ARCHIVE     | int     | Archive every N-th frame.                                          |
| DRAIN_QUEUES    | bool    | Drain queues when stopping (true/false).                           |



## `[pois]` : ROI (Region of Interest) Configuration

| Parameter      | Type    | Description                                                    |
|----------------|---------|----------------------------------------------------------------|
| ROW_START      | int     | First ROI: starting row.                                       |
| COL_START      | int     | First ROI: starting column.                                    |
| ROW_END        | int     | Final ROI: ending row (inclusive).                             |
| COL_END        | int     | Final ROI: ending column (inclusive).                          |
| ROI_W          | int     | ROI width in pixels.                                           |
| ROI_H          | int     | ROI height in pixels.                                          |
| LAMBDA_OCC     | float   | Poisson lambda for "occupied" hypothesis.                      |
| LAMBDA_EMPTY   | float   | Poisson lambda for "empty" hypothesis.                         |
| FP_TARGET      | float   | Targeted false positive rate; used for detection calibration.  |
| MAX_K          | int     | Max candidates to scan over                                    |

The Poisson threshold is computed as the following. Consider two hypothesis event: $H_0$ : site empty, and $H_1$ : site occupied. For the two events, we have $N \sim \mathrm{Poisson}\left(\lambda_{\text 1}\right)$, and $N \sim \mathrm{Poisson}\left(\lambda_{\text 2}\right)$ where $\lambda_1$ is the `LAMBDA_EMPTY` and $\lambda_2$ is the `LAMBDA_OCC`. We have the following decision rule
$$
\delta_T(N)= \begin{cases}\text { occupied } & \text { if } N \geq T, \\ \text { empty } & \text { if } N<T .\end{cases}
$$
We want to minimize the two sided error $P = P_{\text{FP}} + = P_{\text{FN}}$ where
$$
P_{\mathrm{FP}}(T)=\mathbb{P}\left(N \geq T \mid H_0\right)=\sum_{n=T}^{\infty} e^{-\lambda_{\text {1}}} \frac{\lambda_{\text {1}}^n}{n!} .
$$
and 
$$
P_{\mathrm{FN}}(T)=\mathbb{P}\left(N<T \mid H_1\right)=\sum_{n=0}^{T-1} e^{-\lambda_{\mathrm{2}}} \frac{\lambda_{\mathrm{2}}^n}{n!}.
$$
The threshold can be precomputed before the pipeline starts.



## `[dma]` : DMA & Ingress Thread Parameters

| Parameter       | Type    | Description                                            |
|-----------------|---------|--------------------------------------------------------|
| PACE_FRAMES     | bool    | Whether to strictly pace frame ingestion.              |
| CPU_AFFINITY    | int     | Pin DMA thread to a CPU (-1 for no pinning).           |
| THREAD_NICE     | int     | niceness (priority) of thread (0 = default).           |
| LOOP_FRAMES     | bool    | Loop source frames when end is reached.                |



## `[proc]` : Processing Parameters

| Parameter         | Type    | Description                                        |
|-------------------|---------|----------------------------------------------------|
| WORKER_THREADS    | int     | Number of worker threads for processing.           |
| PRINT_STDOUT      | bool    | Print processing results to stdout.                |
| INTERVAL_PRINT    | int     | Print every Nth frame processed (if PRINT_STDOUT). |
| PROC_OUTPUT_DIR   | str     | Output directory for processor results.            |



## `[arch]` : Archiving & Output Parameters

| Parameter          | Type    | Description                                              |
|--------------------|---------|----------------------------------------------------------|
| SEGMENT_BYTES      | int     | Bytes per archive segment file.                          |
| IO_BUFFER_BYTES    | int     | Size of I/O buffer (bytes).                             |
| MAKE_INDEX         | bool    | Whether to generate index files.                         |
| USE_IO_URING       | bool    | Use Linux io_uring for asynchronous IO.                  |
| DIRECT_IO          | bool    | Use O_DIRECT for file writes (if supported).            |
| URING_QD           | int     | io_uring queue depth.                                   |
| MAX_INFLIGHT       | int     | Max archive ops in flight.                              |
| BLOCK_BYTES        | int     | Archive file system block size in bytes.                |

---

## Example

```toml
[main]
IMAGE_W = 300
IMAGE_H = 300
FRAME_BYTES = 180000
SLABS = 128
FPS = 1000
NTH_ARCHIVE = 10
DRAIN_QUEUES = false

[pois]
ROW_START = 50
COL_START = 50
ROW_END = 250
COL_END = 250
ROI_W = 10
ROI_H = 10
LAMBDA_OCC = 10.0
LAMBDA_EMPTY = 0.5
FP_TARGET = 1e-3
MAX_K = 100

[dma]
PACE_FRAMES = true
CPU_AFFINITY = -1
THREAD_NICE = 0
LOOP_FRAMES = true

[proc]
WORKER_THREADS = 4
PRINT_STDOUT = false
INTERVAL_PRINT = 37
PROC_OUTPUT_DIR = "PROC_out"

[arch]
SEGMENT_BYTES = 4294967296
IO_BUFFER_BYTES = 8388608
MAKE_INDEX = true
USE_IO_URING = false
DIRECT_IO = false
URING_QD = 64
MAX_INFLIGHT = 32
BLOCK_BYTES = 4096
...
```
