# KSNP: GPU Chemotactic Particles Simulator

> **Short description:** High-performance CUDA C++ simulation of chemotactic active searchers with scent deposition, pairwise interactions, and stochastic resetting mechanisms.

## Summary

KSNP is a high-performance simulation framework designed to model the collective dynamics of active matter, specifically focusing on "searcher" particles that interact with a dynamic chemical environment. The simulator couples agent-based Langevin dynamics with a grid-based representation of a scalar field (scent), allowing particles to modify their environment through deposition and navigate it via chemotaxis.

The core physics engine accounts for pairwise particle interactions (e.g., soft repulsion) accelerated by spatial hashing, complex boundary conditions, and varying chemosensitivity models (e.g., gradient vs. log-gradient sensing). A distinguishing feature of KSNP-v7.1 is its extensive support for stochastic resetting mechanisms, allowing researchers to investigate how timed or spatial resets affect search efficiency in complex landscapes populated with active targets. Built on CUDA C++ and OpenMP, KSNP-v7.1 supports multi-stream execution, enabling massive parallel parameter sweeps on a single GPU.

## Features

- Highly-parallel CUDA implementation of the simulation core (includes both spatial-hashing and simple interaction kernel variants).
- Hybrid OpenMP/CUDA architecture enabling multi-stream execution for high-throughput parameter sweeps.
- Fully configurable simulation setup via standard INI files (geometry, sensing models, resetting dynamics, and target lists).
- Comprehensive statistical data export (cumulative concentration matrices, regular snapshots, and event logs) designed for post-processing.
- Integrated asynchronous kernel profiling for precise performance monitoring and tuning.

## Prerequisites

- Linux-based OS
- NVIDIA GPU with CUDA support.
- CUDA Toolkit 11.5 or higher
- nvcc available on PATH or set CUDA_HOME.
- C/C++ compiler compatible with CUDA.
- Python 3.8+ for scripts (not required for simulations)

**Environment variables:**
```bash
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```

## Build / Compilation

```bash
cd src/
make -f makefile-sm80
# or
make -f makefile-sm90
```

The simulation configuration (logging, profiling, and threading limits) is controlled via preprocessor macros in `config.cpp`. Before compiling, ensure these are set according to your needs:

*   `RELEASE_MODE`: Default flag for standard high-performance execution.
*   `KERNELTIME`: Uncomment to enable the asynchronous kernel timer. This adds detailed kernel execution statistics to the output log but may slightly impact performance.
*   `DEBUG0` / `DEBUG1` / `DEBUG2`: Uncomment for increasing levels of verbose console output (useful for debugging logic or initialization).
*   `MAX_STREAMS_PER_GPU`: Sets the hard limit on the number of concurrent CUDA streams managed by OpenMP.

## Running the simulation

The executable requires 5 command-line arguments to define the simulation environment and output destinations.

**Syntax:**
```bash
./KSNP-v7.1-sm80 <task.ini> <gpu_id> <num_streams> <seed> <output_prefix>
```

**Typical execution example:**
```bash    
./KSNP-v7.1-sm80 N1024_tauSC0125_tsim1Kx1.ini 0 8 0 N1024_tauSC0125_tsim1Kx1 > N1024_tauSC0125_tsim1Kx1.out 2>&1 & 
```

In this example:
N1024_tauSC0125_tsim1Kx1.ini: uses input task configuration from this file.
0: Runs on GPU device ID 0.
8: Uses 8 CUDA streams.
0: Uses 0 as the random seed (i.e. randomized at execution, plus internally offset by the stream ID).
Output files will be prefixed with N1024_tauSC0125_tsim1Kx1 (e.g., _log.txt, _PCcumulative.txt).
Standard console output is redirected to .out, as well as error output. Runs in background.
  

## Directory layout

```
/ (repository root)
├── bin/                 # compiled binaries
├── examples/            # runnable examples
├── include/             # additional libraries
├── scripts/             # Python scripts and requirements
└── src/                 # CUDA/C++ source and makefiles
```

### `src/` contents (brief description)

- `makefile-sm80` — Makefile targeting compute capability SM 8.0.
- `makefile-sm90` — Makefile targeting compute capability SM 9.0.
- `config.cu` — TODO: confirm exact role.
- `gpudata.cu` — TODO: confirm.
- `kernel_blockSC.cu` — TODO: confirm exact semantics.
- `kernel_init.cu`
- `kernel_leaveScentMark.cu`
- `kernel_moveParticles.cu`
- `kernel_pairwiseForcesHashing.cu`
- `kernel_pairwiseForcesSimple.cu`
- `kernel_postParticles.cu`
- `kernel_renormSC.cu`
- `kernel_updateTargets.cu`
- `kernelTimerAsync.cu`
- `KSNP-v7.1-main.cu` — TODO: clarify whether main or legacy.
- `logdata.cu`
- `runner.cu`
- `setup.cu`
- `simdata.cu`
- `simulator.cu` — TODO: confirm main executable.
- `solver.cu`

## Examples

- `examples/` — a set of examples for chemotactic particles simulations.

**TODO**: list specific example files, run durations, and expected outputs.


## Scripts / Postprocessing

**TODO**

Run visualization:
```bash
python scripts/visualize.py --input examples/minimal/output --save figures/
```

## Citation 

**TODO** Add citation and DOI
See CITATION.cff file.


## License

License: MIT. See LICENSE file.


## Contact

Author: Vladimir Rudyak
E-mail: vurdizm@gmail.com
