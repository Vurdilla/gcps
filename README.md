# BIG UPDATE TO VERSION 7.10 IS HERE 🍭

## New features

### Particles:

**Additional mechanics**
The particle velocity now is not fixed to the single value V0 for all the simulation time. Each particle now has the default velocity V0, and it's current velocity equals to V0 by default, but it can be affected by events during the simulations - same as beta. In the same manner, Searcher->VDecayTime in setup determines the characteristic time of exponential relaxation of the current velocity of each particle to it's V0.

**Population of particles**
Before, all the particles were identical in their parameters (velocity, rotational diffusion, chemosensitivity, chemosecretion). Now, you still can set a single value as before. And in addition, you can assign a distribution of values for the following parameters:
- default velocity V0 (by Searcher->V0_distr)
- rotational diffusion coefficient (by Searcher->rotationalDiffusion_distr)
- default chemosecretion beta0 (by Searcher->beta0_distr)
- default rotational and translational chemosensitivities (by Searcher->chiRot_distr and Searcher->chiTrans_distr)
- chemosensitivity noise level (by Searcher->SC0_distr)

Each distribution can be one of the following types (in Searcher section of the setup):
- single value (distr = 0)
- bimodal (distr = 1)
- uniform (distr = 2)
- log-uniform (distr = 3)
- gaussian (distr = 4)
- log-gaussian (distr = 5)

Then the parameters of the distributions are determined by the the following additional parameter in Searcher section of the setup:
min, max, bias, mean, sigma
For example, for V0 distribution is will be V0_min, V0_max, V0_bias, V0_mean, V0_sigma. The meaning of each parameter is defined by distribution type, check the generator code `kernel_init.cu`.

### System geometry:
**Periodic boundary conditions are available now**
You can choose between three system geometries now (via System->boundaryType):
- circle (boundaryType = 0)
- square (boundaryType = 1)
- square with periodic boundary conditions (boundaryType = 2)

In addition to that, you can determine where the particles will start at time 0 (via System->initialParticlePos):
- center of arena (initialParticlePos = 0)
- uniform distribution across arena area (initialParticlePos = 1)

### Time reset
In special cases, I needed to reset all the particles at once (globally) in the middle of simulations. For this, the parameter TimedReset->globalResetTime was introduced, with options:
- no global resetting (globalResetTime = -1)
- global resetting at a fixed time t (globalResetTime = t)
Note that the globalResetTime value is in simulation time units, not timesteps.

### Targets:
Now target hit can affect particles velocity in addition to beta (via Boundary->VBoundary and Boundary->VBoundary in setup).

Now several types of particle reactions to a target hit are supported (via TargetX->targetResetType in setup):
- particle position to the center + random direction
- particle  direction to the center + one timestep towards center
- particle reverse direction + one timestep backward
- permeatable target, replacing particles velocity and beta with targets values while the particle is inside the target

### Boundary:
Now boundary hit can affect particles velocity in addition to beta (via Target->VTargetMin and Target->VTargetMax in setup).

### Scenarios added:
Scenarios are the global changes independent on the current particle state, that you can imply in your simulations. Simulation can contain multiple scenarios at once (or no scenarios at all). In each scenario, you chose what particle parameter it will set globally to all particles, to each value and and what time. You can make changes to one of these particle parameters (via scenarioParticlesX->parameter):
- particle_V0 (parameter = 1)
- particle_beta0 (parameter = 2)
- particle_chiT (parameter = 3)
- particle_chiR (parameter = 4)
- particle_c0 (parameter = 5)
- particle_DR (parameter = 6)

The set of time values (in time steps) and parameter values to be set are defined in setup in scenarioParticlesX->timesteppoints and scenarioParticlesX->values (both can be one of multiple values, for example, "1.0, 1.1, 1.2").

### Output:
**Trajectory output in LAMMPS trj format**
The trajectory output is switched on/off via Output->printTrjs in the setup.
You can set the min amd max times for trj as general timestep cutoffs for trajectory output. Inside these limits, the trajectory is written in blocks, starting every Output->trjBlockEverySteps, and each block of length Output->trjBlockDurationSteps. The output frequency is controlled by Output->trjBlockFreqSteps (in time steps). as well as output frequency and block pattern. Additional parameteres of the particles can be written to trj (V0, V, beta0, beta, chiT, chiR, c0, DR via Output->trjParticleProperties).

**Radial density distribution**
- additional to the full cumulative particle density matrix, radial distribution is written now as well. The granilarity of bins over radius is controlled via Output->cumulativeRnbins.

**Individual particle hits versus time**
The table showing how many *different* particles hitted the target. Controlled by Output->HitCountReg_window and Output->HitCountReg_minStep.



# GPU Chemotactic Particles Simulator (GCPS)

> **Short description:** High-performance CUDA C++ simulation of chemotactic active searchers with scent deposition, pairwise interactions, and stochastic resetting mechanisms.

## Summary

GCPS is a high-performance simulation framework designed to model the collective dynamics of active matter, specifically focusing on "searcher" particles that interact with a dynamic chemical environment. The simulator couples agent-based Langevin dynamics with a grid-based representation of a scalar field (scent), allowing particles to modify their environment through deposition and navigate it via chemotaxis.

The core physics engine accounts for pairwise particle interactions (e.g., soft repulsion) accelerated by spatial hashing, complex boundary conditions, and varying chemosensitivity models (e.g., gradient vs. log-gradient sensing). A distinguishing feature of GCPS-v7.10 is its extensive support for stochastic resetting mechanisms, allowing researchers to investigate how timed or spatial resets affect search efficiency in complex landscapes populated with active targets. Built on CUDA C++ and OpenMP, GCPS-v7.10 supports multi-stream execution, enabling massive parallel parameter sweeps on a single GPU.

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
./GCPS-v7.10-sm80 <task.ini> <gpu_id> <num_streams> <seed> <output_prefix>
```

**Typical execution example:**
```bash    
./GCPS-v7.10-sm80 N1024_tauSC0125_tsim1Kx1.ini 0 8 0 N1024_tauSC0125_tsim1Kx1 > N1024_tauSC0125_tsim1Kx1.out 2>&1 & 
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
- `GCPS-v7.10-main.cu` — TODO: clarify whether main or legacy.
- `logdata.cu`
- `runner.cu`
- `setup.cu`
- `simdata.cu`
- `simulator.cu` — TODO: confirm main executable.
- `solver.cu`

## Examples

- `examples/` — a set of examples for chemotactic particles simulations.

**TODO**: list specific example files, run durations, and expected outputs.


## Model flexibility

Here are several parts of the code, where physics is explicitly separated from the technicalities of CUDA, and thus can be easily modified within the existing simulations workflow:
- Initial state of the particles: kernel_init.cu, lines 63--92 (positions), line 100 (rotation);
- The equations for the pairwise forces: kernel_pairwiseForcesHashing.cu, lines 89--92;
- The shape of the chemosecretion profile: kernel_leaveScentMark.cu, lines 26--47 (for non-PBC) and lines 91--92 (for PBC);
- Boundary shape: kernel_moveParticles.cu, lines 272--284;
- Resetting rate distribution: kernel_moveParticles.cu, lines 344--354 (during the simulations); kernel_init.cu, lines 356--377 (at the beginning of simulations);
- The equations for the action of chemotactic field and its gradient: kernel_moveParticles.cu, lines 101--116 (current velocity), lines 161--168 (internal angle);
- On-the-fly trajectory analysis (computed on CPU host in parallel to GPU computations, no CUDA knowledge needed): simdata.cu (defines measured data); runner.cu, lines 172--388 (computes measured data); simulator.cu, lines 140--413 (writes measured data to the log file).


## Citation 

CIte this work when using the code:
Efficient open-source GPU implementation for multi-agent autochemotactic 2D modeling, Vladimir Yu. Rudyak, Yael Roichman, Computer Physics Communications, 2026
DOI: 10.1016/j.cpc.2026.110285
https://doi.org/10.1016/j.cpc.2026.110285
https://www.sciencedirect.com/science/article/pii/S0010465526002675
See CITATION.cff file.


## License

License: MIT. See LICENSE file.


## Contact

Author: Vladimir Rudyak
E-mail: vurdizm@gmail.com
