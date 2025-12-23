#ifndef VC3_PHYS_KSNP_SOLVER
#define VC3_PHYS_KSNP_SOLVER

#include <math.h>
#include <time.h>
#include "omp.h"
#include <iostream>

#include "config.cu"

#include "../include/types.cu"
#include "../include/cumath/cuvector2D.cu"
#include "../include/cumath/cusizevector2D.cu"
#include "../include/math/spatial/vector.cu"
#include "../include/math/containers/matrix.cu"
#include "../include/math/randgen/uniform_ran2.cu"
#include "../include/cuda_error_hadling.h"
#include "../include/general/timer_ms_cuda.cu"

#include "setup.cu"
#include "logdata.cu"
#include "gpudata.cu"
#include "kernelTimerAsync.cu"
#include "kernel_init.cu"
#include "kernel_pairwiseForcesSimple.cu"
#include "kernel_pairwiseForcesHashing.cu"
#include "kernel_moveParticles.cu"
#include "kernel_leaveScentMark.cu"
#include "kernel_renormSC.cu"
#include "kernel_postParticles.cu"
#include "kernel_updateTargets.cu"
#include "kernel_blockSC.cu"


namespace vc3_phys
{

class KSNPSolver
{

public:
    /** Constructor
    **/
    KSNPSolver(cudaStream_t& stream, setupParameters setup, int randseed=0, int seedoffset=1) noexcept;

    /** Run one step
    **/
    int runOneStep(cudaStream_t& stream) noexcept;

    /** Get SC matrix **/
    void getSCmatrix(cudaStream_t& stream, vc3_math::Matrix<flt2> *SC, int SCblockSize, bool resize = false) noexcept;

    /** Get particle data **/
    void getParticleData(cudaStream_t& stream, std::vector<vc3_cumath::planar::cuvector> *pos, std::vector<vc3_cumath::planar::cusizevector> *posLattice) noexcept;

    /** Get current status **/
    void getCurrentStatus(cudaStream_t& stream, gpu_variables* status) noexcept;

    /** Print GPU history data **/
    void printGPUHistoryData(std::ostream &out) noexcept;

    /** Destructor
    **/
    ~KSNPSolver();


protected:

    /** Generate a set of targets based on the current setup
    **/
    int generate_targets() noexcept;

    /** Calculate pairwise particle-particle interactions
        LJ with cutoff
    **/
    int ppInteractions(cudaStream_t& stream) noexcept;

    /** Produce set of procedures to move particles across the arena
    **/
    int moveParticles(cudaStream_t& stream) noexcept;

    /** Leave scent marks
    **/
    int leaveScentMarks(cudaStream_t& stream) noexcept;

    /** Post particles procedures
    **/
    int postParticles(cudaStream_t& stream) noexcept;

    /** Store particle history on GPU
    **/
    int storeHistory(cudaStream_t& stream) noexcept;


    // Host data
    setupParameters _setup;
    int _myIter;
    int _myStream;
    int _randseed;
    long long int _currentStep;
    flt2 _currentTime;
    int _currentGPUHistoryOffset;

    // Additional host data
    int _threadsPerBlock;
    flt2 _dl; // lattice step
    flt2 _SCmark_cutoff; // scent mark cutoff
    flt2 _r2max;
    int _SCmark_dl_cutoff; // number of lattice steps in scent mark cutoff
    long long int _stepsSinceScentDivideFactorRenorm;
    gpu_particle_stats _cpu_particles_stats; // length of gpu history + 1, as the first element stores the last pre-history state
    gpu_targets _cputargets;
    logData _timing;
    vc3_general::timer_ms_cuda _solverTimer;
    vc3_cumath::planar::cuvector* particle_PPforce;
    vc3_cumath::planar::cuvector* particle_PPforceSpatialHashing;

    // GPU data
    gpu_setup_variables _gpusetup_variables;
    gpu_precomputes _gpuprecomputes;
    gpu_targets _gputargets, * __gputargets;
    gpu_variables _gpuvariables, *__gpuvariables;
    gpu_matrixes _gpumatrixes, *__gpumatrixes;
    gpu_particles _gpuparticles, *__gpuparticles;
    gpu_ppinteractions _gpuppinteractions, * __gpuppinteractions;
    gpu_particle_stats _gpuparticle_stats, *__gpuparticle_stats;

};

}// namespace vc3_phys


vc3_phys::KSNPSolver::KSNPSolver(cudaStream_t& stream, vc3_phys::setupParameters setup, int randseed, int iter) noexcept
{
    // 0. Define default kernel thread number on GPU
    // ==================
    _threadsPerBlock = 512;

    // 1. Setup
    // ==================
    _setup = setup;
    _myIter = iter;
    _myStream = omp_get_thread_num();
    _randseed = randseed;
    if (_randseed <= 0)
    {
        // Generate random seed
        long ltime = time(NULL);
        unsigned stime = (unsigned)ltime / 2;
        srand(stime);
        _randseed = rand() % 35648125 + omp_get_thread_num();
    }
    int seedoffset = iter;
    _randseed += seedoffset * MAX_STREAMS_PER_GPU;
    std::cout << "\n randseed = " << _randseed << "\n";

    // 2. CPU data
    // ==================
    _currentStep = 0;
    _currentTime = 0.00;
    _currentGPUHistoryOffset = 0;
    _dl = _setup.boxSize / flt2(_setup.latticeSize - 1);
    _SCmark_cutoff = _setup.Rscent * _setup.Rscent_cutoffMultiplier;
    _r2max = _SCmark_cutoff * _SCmark_cutoff;
    _SCmark_dl_cutoff = _SCmark_cutoff / _dl;
    if (_SCmark_dl_cutoff * _dl < _SCmark_cutoff) _SCmark_dl_cutoff++;
    _stepsSinceScentDivideFactorRenorm = 0;
    int cpu_history_size = _setup.nParticles * (_setup.gpuHistoryLength + 1);
#ifdef DEBUG0
    std::cout << "\KSNPSolver::KSNPSolver() - allocating _cpu_particles_stats" << cpu_history_size << "\n";
    std::cout.flush();
#endif
    SAFE_CALL(cudaMallocHost(&_cpu_particles_stats.particle_pos_history, sizeof(vc3_cumath::planar::cuvector) * cpu_history_size));
    SAFE_CALL(cudaMallocHost(&_cpu_particles_stats.particle_angle_history, sizeof(flt2) * cpu_history_size));
    SAFE_CALL(cudaMallocHost(&_cpu_particles_stats.particle_flag_boundaryHit_history, sizeof(bool) * cpu_history_size));
    SAFE_CALL(cudaMallocHost(&_cpu_particles_stats.particle_flag_targetHit_history, sizeof(int) * cpu_history_size));
    SAFE_CALL(cudaMallocHost(&_cpu_particles_stats.particle_flag_timedReset_history, sizeof(bool) * cpu_history_size));
    _timing.initialize();

    SAFE_CALL(cudaMallocHost(&particle_PPforce, sizeof(vc3_cumath::planar::cuvector) * _setup.nParticles));
    SAFE_CALL(cudaMallocHost(&particle_PPforceSpatialHashing, sizeof(vc3_cumath::planar::cuvector) * _setup.nParticles));

    // 3. GPU setup variables
    // ==================
    // Set values on host
#ifdef DEBUG0
    std::cout << "\KSNPSolver::KSNPSolver() - _gpusetup_variables.set(_setup)\n";
    std::cout.flush();
#endif
    _gpusetup_variables.set(_setup);
    // Copy variables to the device
    SAFE_CALL(cudaMemcpyToSymbolAsync(__gpusetup_variables, &_gpusetup_variables, sizeof(gpu_setup_variables), 0, cudaMemcpyHostToDevice, stream));

    // 4. GPU precomputes
    // ==================
    // Set values on host
#ifdef DEBUG0
    std::cout << "\KSNPSolver::KSNPSolver() - _gpuprecomputes.set(_setup)\n";
    std::cout.flush();
#endif
    _gpuprecomputes.set(_setup);
    // Copy variables to the device
    SAFE_CALL(cudaMemcpyToSymbolAsync(__gpuprecomputes, &_gpuprecomputes, sizeof(gpu_precomputes), 0, cudaMemcpyHostToDevice, stream));

    // 5. GPU targets
    // ==================
    // // Generate random targets on host
#ifdef DEBUG0
    std::cout << "\KSNPSolver::KSNPSolver() - generate_targets()\n";
    std::cout.flush();
#endif
    int target_err = generate_targets();
    if (target_err > 0)
    {
        std::cout << "\n!!! Target generation error: " << target_err << " !!!\n";
    }
    // Allocate memory on GPU
#ifdef DEBUG0
    std::cout << "\KSNPSolver::KSNPSolver() - allocating _gputargets" << _gpusetup_variables.nTargets << "\n";
    std::cout.flush();
#endif
    SAFE_CALL(cudaMalloc((void**)&__gputargets, sizeof(gpu_targets)));
    SAFE_CALL(cudaMalloc((void**)&_gputargets.target_active, sizeof(bool) * _gpusetup_variables.nTargets));
    SAFE_CALL(cudaMalloc((void**)&_gputargets.target_pos, sizeof(vc3_cumath::planar::cuvector) * _gpusetup_variables.nTargets));
    SAFE_CALL(cudaMalloc((void**)&_gputargets.target_radius, sizeof(flt2) * _gpusetup_variables.nTargets));
    SAFE_CALL(cudaMalloc((void**)&_gputargets.target_radius2, sizeof(flt2) * _gpusetup_variables.nTargets));
    SAFE_CALL(cudaMalloc((void**)&_gputargets.target_hit_max, sizeof(int) * _gpusetup_variables.nTargets));
    SAFE_CALL(cudaMalloc((void**)&_gputargets.target_hit_count, sizeof(int) * _gpusetup_variables.nTargets));
    SAFE_CALL(cudaMalloc((void**)&_gputargets.target_appearTime, sizeof(flt2) * _gpusetup_variables.nTargets));
    SAFE_CALL(cudaMalloc((void**)&_gputargets.target_terminationType, sizeof(int) * _gpusetup_variables.nTargets));
    SAFE_CALL(cudaMalloc((void**)&_gputargets.target_terminationTime, sizeof(flt2) * _gpusetup_variables.nTargets));
    SAFE_CALL(cudaMalloc((void**)&_gputargets.target_resetType, sizeof(int) * _gpusetup_variables.nTargets));
    SAFE_CALL(cudaMalloc((void**)&_gputargets.target_betaMult, sizeof(flt2) * _gpusetup_variables.nTargets));
    SAFE_CALL(cudaMalloc((void**)&_gputargets.target_resetSRtime, sizeof(int) * _gpusetup_variables.nTargets));
    SAFE_CALL(cudaMalloc((void**)&_gputargets.target_homeType, sizeof(int) * _gpusetup_variables.nTargets));
    // Copy variables to the device
    SAFE_CALL(cudaMemcpyAsync(__gputargets, &_gputargets, sizeof(gpu_targets), cudaMemcpyHostToDevice, stream));
    SAFE_CALL(cudaMemcpyAsync(_gputargets.target_active, _cputargets.target_active, sizeof(bool) * _gpusetup_variables.nTargets, cudaMemcpyHostToDevice, stream));
    SAFE_CALL(cudaMemcpyAsync(_gputargets.target_pos, _cputargets.target_pos, sizeof(vc3_cumath::planar::cuvector) * _gpusetup_variables.nTargets, cudaMemcpyHostToDevice, stream));
    SAFE_CALL(cudaMemcpyAsync(_gputargets.target_radius, _cputargets.target_radius, sizeof(flt2) * _gpusetup_variables.nTargets, cudaMemcpyHostToDevice, stream));
    SAFE_CALL(cudaMemcpyAsync(_gputargets.target_radius2, _cputargets.target_radius2, sizeof(flt2) * _gpusetup_variables.nTargets, cudaMemcpyHostToDevice, stream));
    SAFE_CALL(cudaMemcpyAsync(_gputargets.target_hit_max, _cputargets.target_hit_max, sizeof(int) * _gpusetup_variables.nTargets, cudaMemcpyHostToDevice, stream));
    SAFE_CALL(cudaMemcpyAsync(_gputargets.target_hit_count, _cputargets.target_hit_count, sizeof(int) * _gpusetup_variables.nTargets, cudaMemcpyHostToDevice, stream));
    SAFE_CALL(cudaMemcpyAsync(_gputargets.target_appearTime, _cputargets.target_appearTime, sizeof(flt2) * _gpusetup_variables.nTargets, cudaMemcpyHostToDevice, stream));
    SAFE_CALL(cudaMemcpyAsync(_gputargets.target_terminationType, _cputargets.target_terminationType, sizeof(int) * _gpusetup_variables.nTargets, cudaMemcpyHostToDevice, stream));
    SAFE_CALL(cudaMemcpyAsync(_gputargets.target_terminationTime, _cputargets.target_terminationTime, sizeof(flt2) * _gpusetup_variables.nTargets, cudaMemcpyHostToDevice, stream));
    SAFE_CALL(cudaMemcpyAsync(_gputargets.target_resetType, _cputargets.target_resetType, sizeof(int) * _gpusetup_variables.nTargets, cudaMemcpyHostToDevice, stream));
    SAFE_CALL(cudaMemcpyAsync(_gputargets.target_betaMult, _cputargets.target_betaMult, sizeof(flt2) * _gpusetup_variables.nTargets, cudaMemcpyHostToDevice, stream));
    SAFE_CALL(cudaMemcpyAsync(_gputargets.target_resetSRtime, _cputargets.target_resetSRtime, sizeof(int) * _gpusetup_variables.nTargets, cudaMemcpyHostToDevice, stream));
    SAFE_CALL(cudaMemcpyAsync(_gputargets.target_homeType, _cputargets.target_homeType, sizeof(int) * _gpusetup_variables.nTargets, cudaMemcpyHostToDevice, stream));
    cudaDeviceSynchronize();

    // 6. GPU variables
    // ==================
    // // Set values on host
    _gpuvariables.currentStep = 0;
    _gpuvariables.currentTime = 0.00;
    _gpuvariables.scentDivideFactor = 1.00;
    _gpuvariables.stepsSinceScentDivideFactorRenorm = 0;
    _gpuvariables.historyOffset = 0;
    _gpuvariables.error_vnan = 0;
    _gpuvariables.error_posnan = 0;
    _gpuvariables.error_omeganan = 0;
    _gpuvariables.error_bhposnan = 0;
    // Allocate memory on GPU
    SAFE_CALL(cudaMalloc((void**)&__gpuvariables, sizeof(gpu_variables)));
    // Copy variables to the device
    SAFE_CALL(cudaMemcpyAsync(__gpuvariables, &_gpuvariables, sizeof(gpu_variables), cudaMemcpyHostToDevice, stream));
    cudaDeviceSynchronize();

    // 7. GPU matrixes
    // ==================
    // Allocate memory on GPU
#ifdef DEBUG0
    std::cout << "\KSNPSolver::KSNPSolver() - allocating _gpumatrixes " << _setup.latticeSize << " x " << _setup.latticeSize << "\n";
    std::cout.flush();
#endif
    SAFE_CALL(cudaMalloc((void**)&__gpumatrixes, sizeof(gpu_matrixes)));
    long long int matrixSizeBytes = (long long int)sizeof(flt2) * _setup.latticeSize * _setup.latticeSize;
    SAFE_CALL(cudaMalloc((void**)&_gpumatrixes.SC, matrixSizeBytes));
    long long int sc_blockedBytes = (long long int)sizeof(flt2) * _gpuprecomputes.SCblockLatticeSize * _gpuprecomputes.SCblockLatticeSize;
    SAFE_CALL(cudaMalloc((void**)&_gpumatrixes.SCblocked, sc_blockedBytes));
#ifdef DEBUG0
    std::cout << "\KSNPSolver::KSNPSolver() - allocating _gpumatrixes " << _setup.latticeSize << " x " << _setup.latticeSize << " - done\n";
    std::cout.flush();
#endif
    // Copy pointers to the device
    SAFE_CALL(cudaMemcpyAsync(__gpumatrixes, &_gpumatrixes, sizeof(gpu_matrixes), cudaMemcpyHostToDevice, stream));
    // Initialize GPU objects
#ifdef DEBUG0
    std::cout << "\KSNPSolver::KSNPSolver() - __kernel_init_matrixes\n";
    std::cout.flush();
#endif
    dim3 blockDim(16, 16);
    dim3 gridDim((_setup.latticeSize + blockDim.x - 1) / blockDim.x, (_setup.latticeSize + blockDim.y - 1) / blockDim.y);
#ifdef KERNELTIME
    kernelTimerAsync::getInstance().record("__kernel_init_matrixes_2D",
        __kernel_init_matrixes_2D, gridDim, blockDim, 0, stream,
        __gpumatrixes);
#else
    SAFE_KERNEL_CALL((__kernel_init_matrixes_2D << < gridDim, blockDim, 0, stream >> > (__gpumatrixes)));
#endif
#ifdef DEBUG0
    std::cout << "\KSNPSolver::KSNPSolver() - __kernel_init_matrixes - done\n";
    std::cout.flush();
#endif

    // 8. GPU particles
    // ==================
    // Allocate memory on GPU
#ifdef DEBUG0
    std::cout << "\KSNPSolver::KSNPSolver() - allocating _gpuparticles " << _setup.nParticles << "\n";
    std::cout.flush();
#endif
    SAFE_CALL(cudaMalloc((void**)&__gpuparticles, sizeof(gpu_particles)));
    SAFE_CALL(cudaMalloc((void**)&_gpuparticles.particle_pos, sizeof(vc3_cumath::planar::cuvector) * _setup.nParticles));
    SAFE_CALL(cudaMalloc((void**)&_gpuparticles.particle_posLattice, sizeof(vc3_cumath::planar::cusizevector) * _setup.nParticles));
    SAFE_CALL(cudaMalloc((void**)&_gpuparticles.particle_dpos, sizeof(vc3_cumath::planar::cuvector) * _setup.nParticles));
    SAFE_CALL(cudaMalloc((void**)&_gpuparticles.particle_angle, sizeof(flt2) * _setup.nParticles));
    SAFE_CALL(cudaMalloc((void**)&_gpuparticles.particle_rot, sizeof(vc3_cumath::planar::cuvector) * _setup.nParticles));
    SAFE_CALL(cudaMalloc((void**)&_gpuparticles.particle_SC, sizeof(flt2) * _setup.nParticles));
    SAFE_CALL(cudaMalloc((void**)&_gpuparticles.particle_GSC, sizeof(vc3_cumath::planar::cuvector) * _setup.nParticles));
    SAFE_CALL(cudaMalloc((void**)&_gpuparticles.particle_PPforce, sizeof(vc3_cumath::planar::cuvector) * _setup.nParticles));
    SAFE_CALL(cudaMalloc((void**)&_gpuparticles.particle_PPforceSpatialHashing, sizeof(vc3_cumath::planar::cuvector) * _setup.nParticles));
    SAFE_CALL(cudaMalloc((void**)&_gpuparticles.particle_beta0, sizeof(flt2) * _setup.nParticles));
    SAFE_CALL(cudaMalloc((void**)&_gpuparticles.particle_beta, sizeof(flt2) * _setup.nParticles));
    SAFE_CALL(cudaMalloc((void**)&_gpuparticles.particle_chiT, sizeof(flt2) * _setup.nParticles));
    SAFE_CALL(cudaMalloc((void**)&_gpuparticles.particle_chiR, sizeof(flt2) * _setup.nParticles));
    SAFE_CALL(cudaMalloc((void**)&_gpuparticles.particle_c0, sizeof(flt2) * _setup.nParticles));
    SAFE_CALL(cudaMalloc((void**)&_gpuparticles.particle_DR, sizeof(flt2) * _setup.nParticles));
    SAFE_CALL(cudaMalloc((void**)&_gpuparticles.particle_NRT, sizeof(flt2) * _setup.nParticles));
    SAFE_CALL(cudaMalloc((void**)&_gpuparticles.particle_flag_boundaryHit, sizeof(bool) * _setup.nParticles));
    SAFE_CALL(cudaMalloc((void**)&_gpuparticles.particle_flag_targetHit, sizeof(int) * _setup.nParticles));
    SAFE_CALL(cudaMalloc((void**)&_gpuparticles.particle_flag_timedReset, sizeof(bool) * _setup.nParticles));
    SAFE_CALL(cudaMalloc((void**)&_gpuparticles.particle_curandstate, sizeof(curandState) * _setup.nParticles));
    SAFE_CALL(cudaMalloc((void**)&_gpuparticles.particle_ID, sizeof(int) * _setup.nParticles));
    SAFE_CALL(cudaMalloc((void**)&_gpuparticles.particle_next_in_cell, sizeof(int) * _setup.nParticles));
    SAFE_CALL(cudaMalloc((void**)&_gpuparticles.particle_cell_coord, sizeof(vc3_cumath::planar::cusizevector) * _setup.nParticles));
    // Copy pointers to the device
    SAFE_CALL(cudaMemcpyAsync(__gpuparticles, &_gpuparticles, sizeof(gpu_particles), cudaMemcpyHostToDevice, stream));
    // Initialize GPU objects
        // Allocate memory on GPU
#ifdef DEBUG0
    std::cout << "\KSNPSolver::KSNPSolver() - allocating _gpuparticles " << _setup.nParticles << " - init\n";
    std::cout.flush();
#endif
    dim3 Nblocks_init_particles((_setup.nParticles + _threadsPerBlock - 1) / _threadsPerBlock);
    dim3 blockDim_init_particles(_threadsPerBlock);
#ifdef KERNELTIME
    kernelTimerAsync::getInstance().record("__kernel_init_particles",
        __kernel_init_particles, Nblocks_init_particles, blockDim_init_particles, 0, stream,
        __gpuparticles, _randseed);
#else
    SAFE_KERNEL_CALL(( __kernel_init_particles<<< Nblocks_init_particles, _threadsPerBlock, 0, stream >>>(__gpuparticles, _randseed)) );
#endif
    // Copy current state from GPU to the last state in current history on CPU
    // - it will make it the first pre-history state after first full history copy
    int initoffset = _currentGPUHistoryOffset * _setup.nParticles;
    SAFE_CALL(cudaMemcpy(_cpu_particles_stats.particle_pos_history + initoffset, _gpuparticles.particle_pos,
        sizeof(vc3_cumath::planar::cuvector) * _setup.nParticles, cudaMemcpyDeviceToHost));
    SAFE_CALL(cudaMemcpy(_cpu_particles_stats.particle_angle_history + initoffset, _gpuparticles.particle_angle,
        sizeof(flt2) * _setup.nParticles, cudaMemcpyDeviceToHost));
    SAFE_CALL(cudaMemcpy(_cpu_particles_stats.particle_flag_boundaryHit_history + initoffset, _gpuparticles.particle_flag_boundaryHit,
        sizeof(bool) * _setup.nParticles, cudaMemcpyDeviceToHost));
    SAFE_CALL(cudaMemcpy(_cpu_particles_stats.particle_flag_targetHit_history + initoffset, _gpuparticles.particle_flag_targetHit,
        sizeof(int) * _setup.nParticles, cudaMemcpyDeviceToHost));
    SAFE_CALL(cudaMemcpy(_cpu_particles_stats.particle_flag_timedReset_history + initoffset, _gpuparticles.particle_flag_timedReset,
        sizeof(bool) * _setup.nParticles, cudaMemcpyDeviceToHost));
    // Allocate memory on GPU
#ifdef DEBUG0
    std::cout << "\KSNPSolver::KSNPSolver() - allocating _gpuparticles " << _setup.nParticles << " - done\n";
    std::cout.flush();
#endif

    // 9. GPU pp interactions
    // ==================
    // Allocate memory on GPU
#ifdef DEBUG0
    std::cout << "\KSNPSolver::KSNPSolver() - allocating _gpu_ppinteractions\n";
    std::cout << "\KSNPSolver::KSNPSolver() - PPhash_table_size = " << _gpuprecomputes.PPhash_table_size << "\n";
    std::cout.flush();
#endif
    SAFE_CALL(cudaMalloc((void**)&__gpuppinteractions, sizeof(gpu_ppinteractions)));
    SAFE_CALL(cudaMalloc((void**)&_gpuppinteractions.grid_cell_heads, sizeof(int)* _gpuprecomputes.PPhash_table_size));
    // Copy pointers to the device
    SAFE_CALL(cudaMemcpyAsync(__gpuppinteractions, &_gpuppinteractions, sizeof(gpu_ppinteractions), cudaMemcpyHostToDevice, stream));

    // 10. GPU particle stats
    // ==================
    // Allocate memory on GPU
#ifdef DEBUG0
    std::cout << "\KSNPSolver::KSNPSolver() - allocating _gpuparticle_stats\n";
    std::cout.flush();
#endif
    SAFE_CALL(cudaMalloc((void**)&__gpuparticle_stats, sizeof(gpu_particle_stats)));
    SAFE_CALL(cudaMalloc((void**)&_gpuparticle_stats.particle_pos_history, sizeof(vc3_cumath::planar::cuvector) * _setup.nParticles * _setup.gpuHistoryLength));
    SAFE_CALL(cudaMalloc((void**)&_gpuparticle_stats.particle_angle_history, sizeof(flt2) * _setup.nParticles * _setup.gpuHistoryLength));
    SAFE_CALL(cudaMalloc((void**)&_gpuparticle_stats.particle_flag_boundaryHit_history, sizeof(bool) * _setup.nParticles * _setup.gpuHistoryLength));
    SAFE_CALL(cudaMalloc((void**)&_gpuparticle_stats.particle_flag_targetHit_history, sizeof(int) * _setup.nParticles * _setup.gpuHistoryLength));
    SAFE_CALL(cudaMalloc((void**)&_gpuparticle_stats.particle_flag_timedReset_history, sizeof(bool) * _setup.nParticles * _setup.gpuHistoryLength));
    // Copy pointers to the device
    SAFE_CALL(cudaMemcpyAsync(__gpuparticle_stats, &_gpuparticle_stats, sizeof(gpu_particle_stats), cudaMemcpyHostToDevice, stream));

    cudaDeviceSynchronize();
    std::cout.flush();
}

int vc3_phys::KSNPSolver::runOneStep(cudaStream_t& stream) noexcept
{
#ifdef DEBUG0
    std::cout << " " << _currentStep;
    std::cout.flush();
#endif
#ifdef DEBUG1
    std::cout << "KSNPSolver::runOneStep():: step = " << _currentStep << "\n";
    std::cout.flush();
#endif
    int err = 0;
#ifdef KERNELTIME
    kernelTimerAsync::getInstance().start_step(stream);
#endif
    _solverTimer.start();
#ifdef DEBUG0
    std::cout << "M";
    std::cout.flush();
#endif
#ifdef DEBUG1
    std::cout << "KSNPSolver::runOneStep()::ppInteractions()\n";
    std::cout.flush();
#endif
    err += ppInteractions(stream);
    _solverTimer.lap();
#ifdef DEBUG0
    std::cout << "m";
    std::cout.flush();
#endif
#ifdef DEBUG1
    std::cout << "KSNPSolver::runOneStep()::moveParticles()\n";
    std::cout.flush();
#endif
    err += moveParticles(stream);
    _timing.timeSolver_moveParticles += _solverTimer.lap();
#ifdef DEBUG0
    std::cout << "l";
    std::cout.flush();
#endif
#ifdef DEBUG1
    std::cout << "KSNPSolver::runOneStep()::leaveScentMarks()\n";
        std::cout.flush();
#endif
    err += leaveScentMarks(stream);
    _timing.timeSolver_leaveScentMarks += _solverTimer.lap();
#ifdef DEBUG0
    std::cout << "p";
    std::cout.flush();
#endif
#ifdef DEBUG1
    std::cout << "KSNPSolver::runOneStep()::postParticles()\n";
        std::cout.flush();
#endif
    err += postParticles(stream);
    _timing.timeSolver_postParticles += _solverTimer.lap();
#ifdef DEBUG0
    std::cout << "s";
    std::cout.flush();
#endif
#ifdef DEBUG1
    std::cout << "KSNPSolver::runOneStep()::storeHistory()\n";
        std::cout.flush();
#endif
    err += storeHistory(stream);
    _timing.timeSolver_storeHistory += _solverTimer.lap();
#ifdef DEBUG0
    std::cout << "+";
    std::cout.flush();
#endif
#ifdef DEBUG1
    std::cout << "KSNPSolver::runOneStep(): done\n";
        std::cout.flush();
#endif
#ifdef KERNELTIME
        kernelTimerAsync::getInstance().end_step();
#endif
    return err;
}

void vc3_phys::KSNPSolver::getSCmatrix(cudaStream_t& stream, vc3_math::Matrix<flt2>* SC, int SCblockSize, bool resize) noexcept
{
#ifdef DEBUG0
    std::cout << "KSNPSolver::getSCmatrix()\n";
    std::cout.flush();
#endif
    cudaStreamSynchronize(stream);
#ifdef DEBUG0
    std::cout << "Resizing host matrix\n";
    std::cout.flush();
#endif
    if (resize) SC->resize(_gpuprecomputes.SCblockLatticeSize, _gpuprecomputes.SCblockLatticeSize);
#ifdef DEBUG0
    std::cout << "Launching SC block kernel\n";
    std::cout.flush();
#endif
    // Launch the coarsening kernel
    dim3 blockDim(16, 16);
    dim3 gridDim((_gpuprecomputes.SCblockLatticeSize + blockDim.x - 1) / blockDim.x, (_gpuprecomputes.SCblockLatticeSize + blockDim.y - 1) / blockDim.y);
#ifdef KERNELTIME
    kernelTimerAsync::getInstance().record("__kernel_blockSC",
        __kernel_blockSC, gridDim, blockDim, 0, stream,
        __gpumatrixes);
#else
    SAFE_KERNEL_CALL(( __kernel_blockSC<<< gridDim, blockDim, 0, stream >>>(__gpumatrixes)) );
#endif
#ifdef DEBUG0
    std::cout << "Copying data from device\n";
    std::cout.flush();
#endif
    flt2* scblocktemp;
    const long long int coarsened_bytes = (long long int)_gpuprecomputes.SCblockLatticeSize * _gpuprecomputes.SCblockLatticeSize * sizeof(flt2);
    SAFE_CALL(cudaMallocHost((void**)&scblocktemp, coarsened_bytes));
    SAFE_CALL(cudaMemcpyAsync(scblocktemp, _gpumatrixes.SCblocked, coarsened_bytes, cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);
#ifdef DEBUG0
    std::cout << "Pushing data from array to matrix\n";
    std::cout.flush();
#endif
    for (int nx = 0; nx < _gpuprecomputes.SCblockLatticeSize; nx++)
        for (int ny = 0; ny < _gpuprecomputes.SCblockLatticeSize; ny++)
            (*SC)(nx, ny) = scblocktemp[nx * _gpuprecomputes.SCblockLatticeSize + ny];
#ifdef DEBUG0
    std::cout << "Deleting array\n";
    std::cout.flush();
#endif
    SAFE_CALL(cudaFreeHost(scblocktemp));
#ifdef DEBUG0
    std::cout << "KSNPSolver::getSCmatrix() done\n";
    std::cout.flush();
#endif
}

void vc3_phys::KSNPSolver::getParticleData(cudaStream_t& stream, std::vector<vc3_cumath::planar::cuvector>* pos, std::vector<vc3_cumath::planar::cusizevector>* posLattice) noexcept
{
#ifdef DEBUG0
    std::cout << "KSNPSolver::getParticleData()\n";
    std::cout.flush();
#endif
    cudaDeviceSynchronize();

    //std::cout << "Resizing host vectors\n";
    pos->resize(_setup.nParticles);
    posLattice->resize(_setup.nParticles);
    //std::cout << "Allocating host arrays\n";
    vc3_cumath::planar::cuvector* postemp;
    vc3_cumath::planar::cusizevector* posLatticetemp;
    cudaMallocHost(&postemp, sizeof(vc3_cumath::planar::cuvector) * _setup.nParticles);
    cudaMallocHost(&posLatticetemp, sizeof(vc3_cumath::planar::cusizevector) * _setup.nParticles);
    //std::cout << "Copying data from device\n";
    SAFE_CALL(cudaMemcpy(postemp, _gpuparticles.particle_pos, sizeof(vc3_cumath::planar::cuvector) * _setup.nParticles, cudaMemcpyDeviceToHost));
    SAFE_CALL(cudaMemcpy(posLatticetemp, _gpuparticles.particle_posLattice, sizeof(vc3_cumath::planar::cusizevector) * _setup.nParticles, cudaMemcpyDeviceToHost));
    //std::cout << "Pushing data from arrays to vectors\n";
    for (int np = 0; np < _setup.nParticles; np++)
    {
        (*pos)[np] = postemp[np];
        (*posLattice)[np] = posLatticetemp[np];
    }
    //std::cout << "Deleting arrays\n";
    cudaFreeHost(postemp);
    cudaFreeHost(posLatticetemp);
#ifdef DEBUG0
    std::cout << "KSNPSolver::getParticleData() done\n";
    std::cout.flush();
#endif
}

void vc3_phys::KSNPSolver::getCurrentStatus(cudaStream_t& stream, vc3_phys::gpu_variables* status) noexcept
{
    cudaDeviceSynchronize();
    SAFE_CALL(cudaMemcpy(status, __gpuvariables, sizeof(vc3_phys::gpu_variables), cudaMemcpyDeviceToHost));
}

void vc3_phys::KSNPSolver::printGPUHistoryData(std::ostream& out) noexcept
{
    if (_currentGPUHistoryOffset != 0) return;
    for (int q = 0; q < _setup.nParticles; q++) out << "\tP" << q + 1 << "x\tP" << q + 1 << "y";
    out << "\n";
    for (int q = 0; q < _setup.gpuHistoryLength; q++)
    {
        out << q + 1;
        for (int w = 0; w < _setup.nParticles; w++)
        {
            int offset = (q + 1) * _setup.nParticles + w;
            out << "\t" << _cpu_particles_stats.particle_pos_history[offset].x 
                << "\t" << _cpu_particles_stats.particle_pos_history[offset].y;
        }
        out << "\n";
    }
}

vc3_phys::KSNPSolver::~KSNPSolver()
{
#ifdef DEBUG0
    std::cout << "KSNPSolver::~KSNPSolver()\n";
    std::cout.flush();
#endif
    // 1. Free memory allocated on GPU
    // =========================================
    SAFE_CALL(cudaFree(__gpuvariables));

    SAFE_CALL(cudaFree(__gputargets));
    SAFE_CALL(cudaFree(_gputargets.target_active));
    SAFE_CALL(cudaFree(_gputargets.target_pos));
    SAFE_CALL(cudaFree(_gputargets.target_radius));
    SAFE_CALL(cudaFree(_gputargets.target_radius2));
    SAFE_CALL(cudaFree(_gputargets.target_hit_max));
    SAFE_CALL(cudaFree(_gputargets.target_hit_count));
    SAFE_CALL(cudaFree(_gputargets.target_appearTime));
    SAFE_CALL(cudaFree(_gputargets.target_terminationType));
    SAFE_CALL(cudaFree(_gputargets.target_terminationTime));
    SAFE_CALL(cudaFree(_gputargets.target_resetType));
    SAFE_CALL(cudaFree(_gputargets.target_betaMult));
    SAFE_CALL(cudaFree(_gputargets.target_resetSRtime));
    SAFE_CALL(cudaFree(_gputargets.target_homeType));

    SAFE_CALL(cudaFree(__gpumatrixes));
    SAFE_CALL(cudaFree(_gpumatrixes.SC));
    SAFE_CALL(cudaFree(_gpumatrixes.SCblocked));

    SAFE_CALL(cudaFree(__gpuparticles));
    SAFE_CALL(cudaFree(_gpuparticles.particle_pos));
    SAFE_CALL(cudaFree(_gpuparticles.particle_posLattice));
    SAFE_CALL(cudaFree(_gpuparticles.particle_dpos));
    SAFE_CALL(cudaFree(_gpuparticles.particle_angle));
    SAFE_CALL(cudaFree(_gpuparticles.particle_rot));
    SAFE_CALL(cudaFree(_gpuparticles.particle_SC));
    SAFE_CALL(cudaFree(_gpuparticles.particle_GSC));
    SAFE_CALL(cudaFree(_gpuparticles.particle_PPforce));
    SAFE_CALL(cudaFree(_gpuparticles.particle_PPforceSpatialHashing));
    SAFE_CALL(cudaFree(_gpuparticles.particle_beta0));
    SAFE_CALL(cudaFree(_gpuparticles.particle_beta));
    SAFE_CALL(cudaFree(_gpuparticles.particle_chiT));
    SAFE_CALL(cudaFree(_gpuparticles.particle_chiR));
    SAFE_CALL(cudaFree(_gpuparticles.particle_c0));
    SAFE_CALL(cudaFree(_gpuparticles.particle_DR));
    SAFE_CALL(cudaFree(_gpuparticles.particle_NRT));
    SAFE_CALL(cudaFree(_gpuparticles.particle_flag_boundaryHit));
    SAFE_CALL(cudaFree(_gpuparticles.particle_flag_targetHit));
    SAFE_CALL(cudaFree(_gpuparticles.particle_flag_timedReset));
    SAFE_CALL(cudaFree(_gpuparticles.particle_curandstate));
    SAFE_CALL(cudaFree(_gpuparticles.particle_ID));
    SAFE_CALL(cudaFree(_gpuparticles.particle_next_in_cell));
    SAFE_CALL(cudaFree(_gpuparticles.particle_cell_coord));

    SAFE_CALL(cudaFree(__gpuppinteractions));
    SAFE_CALL(cudaFree(_gpuppinteractions.grid_cell_heads));

    SAFE_CALL(cudaFree(__gpuparticle_stats));
    SAFE_CALL(cudaFree(_gpuparticle_stats.particle_pos_history));
    SAFE_CALL(cudaFree(_gpuparticle_stats.particle_angle_history));
    SAFE_CALL(cudaFree(_gpuparticle_stats.particle_flag_boundaryHit_history));
    SAFE_CALL(cudaFree(_gpuparticle_stats.particle_flag_targetHit_history));
    SAFE_CALL(cudaFree(_gpuparticle_stats.particle_flag_timedReset_history));

    // 2. Free memory allocated on host
    // =========================================
    delete[] _cputargets.target_active;
    delete[] _cputargets.target_pos;
    delete[] _cputargets.target_radius;
    delete[] _cputargets.target_radius2;
    delete[] _cputargets.target_hit_max;
    delete[] _cputargets.target_hit_count;
    delete[] _cputargets.target_appearTime;
    delete[] _cputargets.target_terminationType;
    delete[] _cputargets.target_terminationTime;
    delete[] _cputargets.target_resetType;
    delete[] _cputargets.target_betaMult;
    delete[] _cputargets.target_resetSRtime;
    delete[] _cputargets.target_homeType;
    SAFE_CALL(cudaFreeHost(_cpu_particles_stats.particle_pos_history));
    SAFE_CALL(cudaFreeHost(_cpu_particles_stats.particle_angle_history));
    SAFE_CALL(cudaFreeHost(_cpu_particles_stats.particle_flag_boundaryHit_history));
    SAFE_CALL(cudaFreeHost(_cpu_particles_stats.particle_flag_targetHit_history));
    SAFE_CALL(cudaFreeHost(_cpu_particles_stats.particle_flag_timedReset_history));
    SAFE_CALL(cudaFreeHost(particle_PPforce));
    SAFE_CALL(cudaFreeHost(particle_PPforceSpatialHashing));

#ifdef DEBUG0
    std::cout << "KSNPSolver::~KSNPSolver() done\n";
    std::cout.flush();
#endif
}

int vc3_phys::KSNPSolver::generate_targets() noexcept
{
    std::cout << "\nKSNPSolver::generate_targets()\n";
    // Check number of targets to be generated
    int nt = _setup.targetList.size();
    if (nt == 0) nt = 1;
    std::cout << "KSNPSolver::generate_targets: Number of targets: " << nt << " (" << _setup.targetList.size() << ")\n";

    // Check consistency of number of target
    if (_gpusetup_variables.nTargets != nt) return 1;
    std::cout << "KSNPSolver::generate_targets: Number of targets consistent\n";

    // Allocate CPU memory
    _cputargets.target_active = new bool[nt];
    _cputargets.target_pos = new vc3_cumath::planar::cuvector[nt];
    _cputargets.target_radius = new flt2[nt];
    _cputargets.target_radius2 = new flt2[nt];
    _cputargets.target_hit_max = new int[nt];
    _cputargets.target_hit_count = new int[nt];
    _cputargets.target_appearTime = new flt2[nt];
    _cputargets.target_terminationType = new int[nt];
    _cputargets.target_terminationTime = new flt2[nt];
    _cputargets.target_resetType = new int[nt];
    _cputargets.target_betaMult = new flt2[nt];
    _cputargets.target_resetSRtime = new int[nt];
    _cputargets.target_homeType = new int[nt];
    std::cout << "KSNPSolver::generate_targets: Memory allocated\n";

    // Generate targets 
    if(_setup.targetList.size() > 0)
    { // Generate regular targets according to setup
        std::cout << "KSNPSolver::generate_targets: generating targets from setup\n";

        vc3_math::randgen_uniform_ran2 rng(_randseed);
        for (int t = 0; t < nt; t++)
        {
            std::cout << "KSNPSolver::generate_targets: generating target " << t + 1 << " / " << nt << "\n";
            flt2 d = _setup.targetList[t].targetDistanceMin + rng.segment_generate() * (_setup.targetList[t].targetDistanceMax - _setup.targetList[t].targetDistanceMin);
            flt2 a = _setup.targetList[t].targetAngleMin + rng.segment_generate() * (_setup.targetList[t].targetAngleMax - _setup.targetList[t].targetAngleMin);
            _cputargets.target_pos[t].x = _setup.boxSize * 0.50 + d * cos(a / 180.00 * vc3_math::Pi);
            _cputargets.target_pos[t].y = _setup.boxSize * 0.50 + d * sin(a / 180.00 * vc3_math::Pi);
            _cputargets.target_radius[t] = _setup.targetList[t].targetRadiusMin + rng.segment_generate() * (_setup.targetList[t].targetRadiusMax - _setup.targetList[t].targetRadiusMin);
            _cputargets.target_radius2[t] = _cputargets.target_radius[t] * _cputargets.target_radius[t];
            _cputargets.target_hit_max[t] = _setup.targetList[t].targetWeightMin + rng.segment_generate() * (_setup.targetList[t].targetWeightMax - _setup.targetList[t].targetWeightMin);
            _cputargets.target_hit_count[t] = 0;
            if (_setup.targetList[t].targetAppearType != 0) 
                return 2;
            _cputargets.target_appearTime[t] = _setup.targetList[t].targetAppearTime;
            if (_setup.targetList[t].targetAppearDelay > 0.0)
                return 3;
            _cputargets.target_terminationType[t] = _setup.targetList[t].targetTerminateType;
            _cputargets.target_terminationTime[t] = _setup.targetList[t].targetTerminateTime;
            _cputargets.target_resetType[t] = _setup.targetList[t].targetResetType;
            _cputargets.target_betaMult[t] = _setup.targetList[t].betaTargetMultMin + rng.segment_generate() * (_setup.targetList[t].betaTargetMultMax - _setup.targetList[t].betaTargetMultMin);
            _cputargets.target_resetSRtime[t] = _setup.targetList[t].targetResetSRtime;
            _cputargets.target_homeType[t] = _setup.targetList[t].targetHomePotential;

            if (_cputargets.target_appearTime[t] <= 0.00) _cputargets.target_active[t] = true;
            else _cputargets.target_active[t] = false;

            std::cout << "KSNPSolver::generate_targets: target " << t + 1 << " / " << nt 
                << " -- pos.x="<< _cputargets.target_pos[t].x << ", pos.y=" << _cputargets.target_pos[t].y
                << ", r="<< _cputargets.target_radius[t] << ", r2=" << _cputargets.target_radius2[t]
                << ", RT="<< _cputargets.target_resetType[t] << ", active=" << _cputargets.target_active[t]
                << "\n";

            std::cout << "KSNPSolver::generate_targets: success at target " << t + 1 << " / " << nt << "\n";
        }
    }
    else
    { // Generate imaginary target to avoid zero array size problem
        std::cout << "KSNPSolver::generate_targets: generating imaginary target\n";
        _cputargets.target_pos[0].x = 0.00;
        _cputargets.target_pos[0].y = 0.00;
        _cputargets.target_radius[0] = 0.00;
        _cputargets.target_radius2[0] = 0.00;
        _cputargets.target_hit_max[0] = 0.00;
        _cputargets.target_hit_count[0] = 0.00;
        _cputargets.target_appearTime[0] = -1.00;
        _cputargets.target_terminationType[0] = 0;
        _cputargets.target_terminationTime[0] = -1.00;
        _cputargets.target_resetType[0] = 0;
        _cputargets.target_betaMult[0] = 0.00;
        _cputargets.target_resetSRtime[0] = 0;
        _cputargets.target_homeType[0] = 0;
        _cputargets.target_active[0] = false;
    }

    std::cout << "KSNPSolver::generate_targets: done\n";
    std::cout.flush();
    return 0;
}

int vc3_phys::KSNPSolver::ppInteractions(cudaStream_t& stream) noexcept
{

    /** Spatial hashing calculations, ~N
    * Parallel execution
        LIMITED TO 65535 PARTICLES
    **/
    if (_setup.PPepsilon > 0.00)
    {
        // Reset the hash table. We use cudaMemset for this as it's often faster for simple initialization
        SAFE_CALL(cudaMemsetAsync(_gpuppinteractions.grid_cell_heads, -1, sizeof(int) * _gpuprecomputes.PPhash_table_size, stream));
        // Build the linked lists for all particles
        dim3 Nblocks_PPforcesSH((_setup.nParticles + _threadsPerBlock - 1) / _threadsPerBlock);
        dim3 blockDim_PPforcesSH(_threadsPerBlock);
#ifdef KERNELTIME
        kernelTimerAsync::getInstance().record("__kernel_PPbuildHashLists",
            __kernel_PPbuildHashLists, Nblocks_PPforcesSH, blockDim_PPforcesSH, 0, stream,
            __gpuparticles, __gpuppinteractions);
#else
        SAFE_KERNEL_CALL((__kernel_PPbuildHashLists<<< Nblocks_PPforcesSH, _threadsPerBlock, 0, stream >>>(__gpuparticles, __gpuppinteractions) ));
#endif
        // Calculate forces using the hash grid. This kernel now contains all fixes
#ifdef KERNELTIME
        kernelTimerAsync::getInstance().record("__kernel_PPforcesHashGrid",
            __kernel_PPforcesHashGrid, Nblocks_PPforcesSH, blockDim_PPforcesSH, 0, stream,
            __gpuparticles, __gpuppinteractions);
#else
        SAFE_KERNEL_CALL(( __kernel_PPforcesHashGrid<<< Nblocks_PPforcesSH, _threadsPerBlock, 0, stream >>>(__gpuparticles, __gpuppinteractions) ));
#endif
    }

    return 0;
}

int vc3_phys::KSNPSolver::moveParticles(cudaStream_t& stream) noexcept
{
    /** Produce set of procedures to move particles across the arena
    **/
    dim3 Nblocks_moveParticles((_setup.nParticles + _threadsPerBlock - 1) / _threadsPerBlock);
#ifdef KERNELTIME
    dim3 blockDim_moveParticles(_threadsPerBlock);
    kernelTimerAsync::getInstance().record("__kernel_moveParticles",
        __kernel_moveParticles, Nblocks_moveParticles, blockDim_moveParticles, 0, stream,
        __gpuvariables, __gputargets, __gpumatrixes, __gpuparticles);
#else
    SAFE_KERNEL_CALL((__kernel_moveParticles << < Nblocks_moveParticles, _threadsPerBlock, 0, stream >> >
        (__gpuvariables, __gputargets, __gpumatrixes, __gpuparticles) ));
#endif
    return 0;
}

int vc3_phys::KSNPSolver::leaveScentMarks(cudaStream_t& stream) noexcept
{
    /* Parallel execution with atomicAdd 
    LIMITED TO 65535 PARTICLES */
    int windowSize2 = (2 * _SCmark_dl_cutoff + 1) * (2 * _SCmark_dl_cutoff + 1);
    dim3 Nblocks_leaveScentMarks( (windowSize2 + _threadsPerBlock - 1) / _threadsPerBlock, _setup.nParticles);
#ifdef KERNELTIME
    dim3 blockDim_leaveScentMarks(_threadsPerBlock);
    kernelTimerAsync::getInstance().record("__kernel_leaveScentMarks",
        __kernel_leaveScentMarks, Nblocks_leaveScentMarks, blockDim_leaveScentMarks, 0, stream,
        __gpuvariables, __gpumatrixes, __gpuparticles);
#else
    SAFE_KERNEL_CALL(( __kernel_leaveScentMarks<<< Nblocks_leaveScentMarks, _threadsPerBlock, 0, stream >>>
        (__gpuvariables, __gpumatrixes, __gpuparticles) ));  
#endif

    return 0;
}

int vc3_phys::KSNPSolver::postParticles(cudaStream_t& stream) noexcept
{
    /** Post particles procedures
    **/

    // Increment time and steps and run scent concentration remormalization if needed
    // It must be run first, as gpuvariables->scentDivideFactor will be reset to 1 in __kernel_postParticles
    _stepsSinceScentDivideFactorRenorm++;
    if (_stepsSinceScentDivideFactorRenorm >= _setup.scentDecayRescalingNsteps)
    {
        //decayScentConcentration(); // LxL
        dim3 blockDim(16, 16);
        dim3 gridDim((_setup.latticeSize + blockDim.x - 1) / blockDim.x, (_setup.latticeSize + blockDim.y - 1) / blockDim.y);
#ifdef KERNELTIME
        kernelTimerAsync::getInstance().record("__kernel_renormSC_2D",
            __kernel_renormSC_2D, gridDim, blockDim, 0, stream,
            __gpuvariables, __gpumatrixes);
#else
        SAFE_KERNEL_CALL(( __kernel_renormSC_2D<<< gridDim, blockDim, 0, stream >>>(__gpuvariables, __gpumatrixes) ));
#endif
        _stepsSinceScentDivideFactorRenorm = 0;
    }

    // Run non-parallel tasks
    //dim3 Nblocks_postParticles(1);
#ifdef KERNELTIME
    dim3 blockDim(1);
    dim3 gridDim(1);
    kernelTimerAsync::getInstance().record("__kernel_postParticles",
        __kernel_postParticles, blockDim, gridDim, 0, stream,
        __gpuvariables, __gpumatrixes, __gpuparticles);
#else
    SAFE_KERNEL_CALL(( __kernel_postParticles<<< 1, 1, 0, stream >>>
        (__gpuvariables, __gpumatrixes, __gpuparticles) ));
#endif
    // Increment CPU step and time synchronious to GPU
    _currentStep++;
    _currentTime += _setup.timeStep;

    // Update targets
    dim3 Nblocks_updateTargets((_gpusetup_variables.nTargets + _threadsPerBlock - 1) / _threadsPerBlock);
#ifdef KERNELTIME
    dim3 blockDim_updateTargets(_threadsPerBlock);
    kernelTimerAsync::getInstance().record("__kernel_updateTargets",
        __kernel_updateTargets, Nblocks_updateTargets, blockDim_updateTargets, 0, stream,
        __gpuvariables, __gputargets);
#else
    SAFE_KERNEL_CALL((__kernel_updateTargets<<< Nblocks_updateTargets, _threadsPerBlock, 0, stream >>>
        (__gpuvariables, __gputargets)));
#endif

    
    return 0;
}

int vc3_phys::KSNPSolver::storeHistory(cudaStream_t& stream) noexcept
{
    /** Store particle history
    **/

    // Increment time and steps and run scent concentration remormalization if needed
    // It must be run first, as gpuvariables->scentDivideFactor will be reset to 1 in __kernel_postParticles
    // Run particle history copying via cudaMemcpyAsync
    int offset = _setup.nParticles * _currentGPUHistoryOffset;
    SAFE_CALL(cudaMemcpyAsync(
        _gpuparticle_stats.particle_pos_history + offset, _gpuparticles.particle_pos,
        _setup.nParticles * sizeof(vc3_cumath::planar::cuvector), cudaMemcpyDeviceToDevice, stream));
    SAFE_CALL(cudaMemcpyAsync(
        _gpuparticle_stats.particle_angle_history + offset, _gpuparticles.particle_angle,
        _setup.nParticles * sizeof(flt2), cudaMemcpyDeviceToDevice, stream));
    SAFE_CALL(cudaMemcpyAsync(
        _gpuparticle_stats.particle_flag_boundaryHit_history + offset, _gpuparticles.particle_flag_boundaryHit,
        _setup.nParticles * sizeof(bool), cudaMemcpyDeviceToDevice, stream));
    SAFE_CALL(cudaMemcpyAsync(
        _gpuparticle_stats.particle_flag_targetHit_history + offset, _gpuparticles.particle_flag_targetHit,
        _setup.nParticles * sizeof(int), cudaMemcpyDeviceToDevice, stream));
    SAFE_CALL(cudaMemcpyAsync(
        _gpuparticle_stats.particle_flag_timedReset_history + offset, _gpuparticles.particle_flag_timedReset,
        _setup.nParticles * sizeof(bool), cudaMemcpyDeviceToDevice, stream));
    // Increment history offset
    _currentGPUHistoryOffset++;
    if (_currentGPUHistoryOffset >= _setup.gpuHistoryLength)
    {
        // Copy the last state to 0 pos
        for (int w = 0; w < _setup.nParticles; w++)
        {
            int offset = _currentGPUHistoryOffset * _setup.nParticles + w;
            _cpu_particles_stats.particle_pos_history[w] = _cpu_particles_stats.particle_pos_history[offset];
            _cpu_particles_stats.particle_angle_history[w] = _cpu_particles_stats.particle_angle_history[offset];
            _cpu_particles_stats.particle_flag_boundaryHit_history[w] = _cpu_particles_stats.particle_flag_boundaryHit_history[offset];
            _cpu_particles_stats.particle_flag_targetHit_history[w] = _cpu_particles_stats.particle_flag_targetHit_history[offset];
            _cpu_particles_stats.particle_flag_timedReset_history[w] = _cpu_particles_stats.particle_flag_timedReset_history[offset];
        }
        // Copy new history from GPU to CPU
        /*std::cout << "\nKSNPSolver::storeHistory - COPY: _currentGPUHistoryOffset = " << _currentGPUHistoryOffset
            << ", _setup.gpuHistoryLength = " << _setup.gpuHistoryLength
            << ", _setup.nParticles = " << _setup.nParticles
            << ", offset = " << offset
            << "\n";*/
        SAFE_CALL(cudaMemcpyAsync(_cpu_particles_stats.particle_pos_history + _setup.nParticles, _gpuparticle_stats.particle_pos_history,
            sizeof(vc3_cumath::planar::cuvector) * _setup.nParticles * _setup.gpuHistoryLength, cudaMemcpyDeviceToHost, stream));
        SAFE_CALL(cudaMemcpyAsync(_cpu_particles_stats.particle_angle_history + _setup.nParticles, _gpuparticle_stats.particle_angle_history,
            sizeof(flt2) * _setup.nParticles * _setup.gpuHistoryLength, cudaMemcpyDeviceToHost, stream));
        SAFE_CALL(cudaMemcpyAsync(_cpu_particles_stats.particle_flag_boundaryHit_history + _setup.nParticles, _gpuparticle_stats.particle_flag_boundaryHit_history,
            sizeof(bool) * _setup.nParticles * _setup.gpuHistoryLength, cudaMemcpyDeviceToHost, stream));
        SAFE_CALL(cudaMemcpyAsync(_cpu_particles_stats.particle_flag_targetHit_history + _setup.nParticles, _gpuparticle_stats.particle_flag_targetHit_history,
            sizeof(int) * _setup.nParticles * _setup.gpuHistoryLength, cudaMemcpyDeviceToHost, stream));
        SAFE_CALL(cudaMemcpyAsync(_cpu_particles_stats.particle_flag_timedReset_history + _setup.nParticles, _gpuparticle_stats.particle_flag_timedReset_history,
            sizeof(bool) * _setup.nParticles * _setup.gpuHistoryLength, cudaMemcpyDeviceToHost, stream));
        _currentGPUHistoryOffset = 0;

#ifdef DEBUG1
        std::cout << "storeHistory copied "<< _setup.gpuHistoryLength<<" steps to CPU\n";
        std::cout.flush();
#endif
#ifdef DEBUG2
        std::cout << "Step\tParticle\tx\ty\tTR\tBH\tTH\n";
        for (int frame = 0; frame < _setup.gpuHistoryLength + 1; frame++)
        {
            std::cout << _currentStep - _setup.gpuHistoryLength - 1 + frame << "\n";
            int offset = frame * _setup.nParticles; // the first frame is the last frevious state
            for (int p = 0; p < _setup.nParticles; p++, offset++)
            {
                std::cout << "\t" << p
                    << "\t" << _cpu_particles_stats.particle_pos_history[offset].x
                    << "\t" << _cpu_particles_stats.particle_pos_history[offset].y
                    << "\t" << _cpu_particles_stats.particle_flag_timedReset_history[offset]
                    << "\t" << _cpu_particles_stats.particle_flag_boundaryHit_history[offset]
                    << "\t" << _cpu_particles_stats.particle_flag_targetHit_history[offset]
                    << "\n";
            }
        }
        std::cout.flush();
#endif
    }

    return 0;
}

#endif // VC3_PHYS_KSNP_SOLVER
