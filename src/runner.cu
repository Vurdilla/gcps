#ifndef VC3_PHYS_KSNP_RUNNER
#define VC3_PHYS_KSNP_RUNNER

#include <iostream>
#include <sstream>
#include <fstream>
#include <iomanip>
#include <string>

#include "../include/math/containers/matrix.cu"

#include "solver.cu"
#include "simdata.cu"


namespace vc3_phys
{

class KSNPRunner: public KSNPSolver
{

public:
    /** Constructor
    **/
    KSNPRunner(cudaStream_t& stream, setupParameters setup, int randseed=0, int seedoffset=1) noexcept;

    /** Run simulations
    **/
    int runSimulation(cudaStream_t& stream) noexcept;

    /** Adds the results of this runner to a total results object.
    **/
    int addResultsTo(simData* totalResults) noexcept;

    /** Adds the timing of this runner to a total results object.
    **/
    int addTimingTo(logData* totalTiming) noexcept;

    /** Destructor
    **/
    ~KSNPRunner();


protected:
    // The results for this specific runner instance
    simData _results;
    vc3_general::timer_ms_cuda _simTimer, _stepTimer;

    // Additional variables
    std::vector<int> _previousEventType; // per particle,  1 - TR, 2 - BH, 3 - TH, 4 - HH
    // Iteration counts of events
    std::vector<long long int> _iterNTR; // Number of timed resets: 0 - total, 1 - previous TR, 2 - previous boundary, 3 - previous target, 4 - previous home
    std::vector<long long int> _iterNBoundaryHit; // Number of boundary hits: 0 - total, 1 - previous TR, 2 - previous boundary, 3 - previous target, 4 - previous home
    std::vector<long long int> _iterNTargetHit; // Number of target hits: 0 - total, 1 - previous TR, 2 - previous boundary, 3 - previous target, 4 - previous home
    std::vector<long long int> _iterNHomeHit; // Number of target hits: 0 - total, 1 - previous TR, 2 - previous boundary, 3 - previous target, 4 - previous home

    /** Collect statistics
    **/
    int collectStat() noexcept;

    /** Collect local data
    **/
    int collectLocal(cudaStream_t& stream) noexcept;

};

}// namespace vc3_phys


vc3_phys::KSNPRunner::KSNPRunner(cudaStream_t& stream, vc3_phys::setupParameters setup, int randseed, int seedoffset) noexcept:
    KSNPSolver(stream, setup, randseed, seedoffset)
{
    // Initialize the results data structure
    _results.initialize(setup);

    // Initialize additional variables
    _previousEventType.resize(setup.nParticles);
    for (int q = 0; q < setup.nParticles; q++) _previousEventType[q] = 1;
    // Iteration counts of events
    _iterNTR.resize(5);
    fill(_iterNTR.begin(), _iterNTR.end(), 0);
    _iterNBoundaryHit.resize(5);
    fill(_iterNBoundaryHit.begin(), _iterNBoundaryHit.end(), 0);
    _iterNTargetHit.resize(5);
    fill(_iterNTargetHit.begin(), _iterNTargetHit.end(), 0);
    _iterNHomeHit.resize(5);
    fill(_iterNHomeHit.begin(), _iterNHomeHit.end(), 0);
}

int vc3_phys::KSNPRunner::runSimulation(cudaStream_t& stream) noexcept
{
    _simTimer.start();
    vc3_general::timer_ms_cuda printTimer;
    printTimer.start();
    int err = 0;
    int nout = 1;
    if (KSNPSolver::_setup.nSteps > 10) nout = KSNPSolver::_setup.nSteps / 10;
    if (KSNPSolver::_setup.nSteps > 1000) nout = KSNPSolver::_setup.nSteps / 100;
    if (nout < 1) nout = 1;
    for (int step = 0; step < KSNPSolver::_setup.nSteps; step++)
    {
        _stepTimer.start();
        err += KSNPSolver::runOneStep(stream);
        KSNPSolver::_timing.timeRunner_runOneStep += _stepTimer.lap();
        err += collectLocal(stream);
        if (KSNPSolver::_currentGPUHistoryOffset == 0)
        {
            cudaStreamSynchronize(stream);
            err += collectStat();
            KSNPSolver::_timing.timeRunner_collectStat += _stepTimer.lap();
        }
        //KSNPSolver::printGPUHistoryData(gputrjfile);

        if (KSNPSolver::_myStream == 0 && step % nout == 0)
        {
            std::cout << "Step\t" << KSNPSolver::_currentStep << "\tsim time\t"
                << KSNPSolver::_currentTime << "\trun time\t" << printTimer.lap() / 1000.00 << "\n";
            std::cout.flush();
            //std::cout << "Step "  << step + 1 << " / " << KSNPSolver::_setup.nSteps << " - err " << err << "\n";
            //std::cout.flush();
        }
    }
    
    // Increment number of computer iterations
    _results.iterationsComputed++;

    KSNPSolver::_timing.timeRunner_runSimulation += _simTimer.lap();

    return err;
}

int vc3_phys::KSNPRunner::collectStat() noexcept
{
#ifdef DEBUG0
    std::cout << "\nKSNPRunner::collectStat()\n";
    std::cout.flush();
#endif

    // Cumulative particle concentration
    int historySize = KSNPSolver::_setup.nParticles * KSNPSolver::_setup.gpuHistoryLength;
    /*std::cout << "\nSTAT: nParticles = " << KSNPSolver::_setup.nParticles
        << ", gpuHistoryLength = " << KSNPSolver::_setup.gpuHistoryLength
        << ", historySize = " << historySize
        << ", matrix size " << _cumulativeParticleCountMatrix.nCol() << " x " << _cumulativeParticleCountMatrix.nRow();
    std::cout.flush();*/
    for (int frame = 0; frame < KSNPSolver::_setup.gpuHistoryLength; frame++)
    {
        int offset = (frame + 1) * KSNPSolver::_setup.nParticles; // the first frame is the last frevious state
        for (int p = 0; p < KSNPSolver::_setup.nParticles; p++, offset++)
        {
            int pcx = KSNPSolver::_cpu_particles_stats.particle_pos_history[offset].x / KSNPSolver::_dl / KSNPSolver::_setup.cumulativePCblockSize;
            int pcy = KSNPSolver::_cpu_particles_stats.particle_pos_history[offset].y / KSNPSolver::_dl / KSNPSolver::_setup.cumulativePCblockSize;
            _results.cumulativeParticleCountMatrix(pcx, pcy)++;
        }
    }

    // Time regular window-cumulative particle concentration
    for (int frame = 0; frame < KSNPSolver::_setup.gpuHistoryLength; frame++)
    {       
        int hstep = KSNPSolver::_currentStep - KSNPSolver::_setup.gpuHistoryLength + frame;
        if (hstep >= KSNPSolver::_setup.PCReg_startstep)
        {
            int nt = (hstep - KSNPSolver::_setup.PCReg_startstep) / KSNPSolver::_setup.PCReg_window;
            int offset = (frame + 1) * KSNPSolver::_setup.nParticles; // the first frame is the last previous state
            for (int p = 0; p < KSNPSolver::_setup.nParticles; p++, offset++)
            {
                int pcx = KSNPSolver::_cpu_particles_stats.particle_pos_history[offset].x / KSNPSolver::_dl / KSNPSolver::_setup.PCReg_blockSize;
                int pcy = KSNPSolver::_cpu_particles_stats.particle_pos_history[offset].y / KSNPSolver::_dl / KSNPSolver::_setup.PCReg_blockSize;
                _results.timeRegParticleCountMatrix[nt](pcx, pcy)++;
            }
        }
    }

#ifdef DEBUG1
    std::cout << "Cumulative counts of events time function\n";
    std::cout.flush();
#endif
#ifdef DEBUG1
    std::cout << "Step\tParticle\tx\ty\tTR\tBH\tTH\t_iterNTR\t_iterNBoundaryHit\t_iterNTargetHit\n";
    std::cout.flush();
#endif
    // Cumulative counts of events time function
    for (int frame = 0; frame < KSNPSolver::_setup.gpuHistoryLength; frame++)
    {
        int hstep = KSNPSolver::_currentStep - KSNPSolver::_setup.gpuHistoryLength + frame;
        int offset = (frame + 1) * KSNPSolver::_setup.nParticles; // the first frame is the last previous state
#ifdef DEBUG1
        std::cout << hstep << "[data per particle below]\n";
        std::cout.flush();
#endif
        for (int p = 0; p < KSNPSolver::_setup.nParticles; p++, offset++)
        {
            // Increment counters
            if (KSNPSolver::_cpu_particles_stats.particle_flag_timedReset_history[offset])
            {
                _iterNTR[0]++;
                _iterNTR[_previousEventType[p]]++;
            }
            if (KSNPSolver::_cpu_particles_stats.particle_flag_boundaryHit_history[offset])
            {
                _iterNBoundaryHit[0]++;
                _iterNBoundaryHit[_previousEventType[p]]++;
            }
            if (KSNPSolver::_cpu_particles_stats.particle_flag_targetHit_history[offset] >= 0)
            {
                _iterNTargetHit[0]++;
                _iterNTargetHit[_previousEventType[p]]++;
            }
            /*if (KSNPSolver::_cpu_particles_stats.particle_flag_homeHit_history[offset])
            {
                _iterNHomeHit[0]++;
                _iterNHomeHit[_previousEventType[p]]++;
            }*/
            // Update previous state
            if (KSNPSolver::_cpu_particles_stats.particle_flag_timedReset_history[offset]) _previousEventType[p] = 1;
            if (KSNPSolver::_cpu_particles_stats.particle_flag_boundaryHit_history[offset]) _previousEventType[p] = 2;
            if (KSNPSolver::_cpu_particles_stats.particle_flag_targetHit_history[offset] >= 0) _previousEventType[p] = 3;
            //if (KSNPSolver::_cpu_particles_stats.particle_flag_homeHit_history[offset]) _previousEventType[p] = 4;

#ifdef DEBUG2
            std::cout << "\t" << hstep
                << "\t" << KSNPSolver::_cpu_particles_stats.particle_pos_history[offset].x
                << "\t" << KSNPSolver::_cpu_particles_stats.particle_pos_history[offset].y
                << "\t" << KSNPSolver::_cpu_particles_stats.particle_flag_timedReset_history[offset]
                << "\t" << KSNPSolver::_cpu_particles_stats.particle_flag_boundaryHit_history[offset]
                << "\t" << KSNPSolver::_cpu_particles_stats.particle_flag_targetHit_history[offset]
                << "\t" << _iterNTR[0]
                << "\t" << _iterNBoundaryHit[0]
                << "\t" << _iterNTargetHit[0]
                << "\n";
            std::cout.flush();
#endif
        } // for (int p = 0; p < KSNPSolver::_setup.nParticles; p++, offset++)

        if (hstep % KSNPSolver::_setup.HitCountReg_window == 0)
        {
            int nw = hstep / KSNPSolver::_setup.HitCountReg_window;
            for (int q = 0; q < 5; q++)
            {
                _results.timeRegCumulativeNTR[nw][q] += _iterNTR[q];
                _results.timeRegCumulativeNBoundaryHit[nw][q] += _iterNBoundaryHit[q];
                _results.timeRegCumulativeNTargetHit[nw][q] += _iterNTargetHit[q];
                _results.timeRegCumulativeNHomeHit[nw][q] += _iterNHomeHit[q];
#ifdef DEBUG2
                std::cout << "HitCountReg\t" << _results.timeRegCumulativeNTR[nw][0]
                    << "\t" << _results.timeRegCumulativeNBoundaryHit[nw][0]
                    << "\t" << _results.timeRegCumulativeNTargetHit[nw][0]
                    << "\t" << _results.timeRegCumulativeNHomeHit[nw][0]
                    << "\n";
                std::cout.flush();
#endif
            }
        }
    }


    return 0;
}

int vc3_phys::KSNPRunner::collectLocal(cudaStream_t& stream) noexcept
{
    if ((KSNPSolver::_currentStep - KSNPSolver::_setup.SCReg_startstep >= 0) &&
        ((KSNPSolver::_currentStep - KSNPSolver::_setup.SCReg_startstep) % KSNPSolver::_setup.SCReg_window == 0))
    {
        int nsc = (KSNPSolver::_currentStep - KSNPSolver::_setup.SCReg_startstep) / KSNPSolver::_setup.SCReg_window - 1;
#ifdef DEBUG0
        std::cout << "\nKSNPRunner::collectLocal: collecting SC matrix at step " << KSNPSolver::_currentStep << " to reg # " << nsc << "\n";
        std::cout.flush();
#endif
        KSNPSolver::getSCmatrix(stream, &(_results.timeRegScentMatrix[nsc]), KSNPSolver::_setup.SCReg_blockSize);
#ifdef DEBUG0
        std::cout << "\nKSNPRunner::collectLocal: collected\n";
        std::cout.flush();
#endif
    }

    return 0;

}

int vc3_phys::KSNPRunner::addResultsTo(simData* totalResults) noexcept
{
    // Delegate the addition logic to the simData struct's method.
    return _results.addTo(totalResults);
}

int vc3_phys::KSNPRunner::addTimingTo(logData* totalTiming) noexcept
{
    // Delegate the addition logic to the simData struct's method.
    return KSNPSolver::_timing.addTo(totalTiming);
}

vc3_phys::KSNPRunner::~KSNPRunner()
{
    // 1. Free memory allocated on GPU
    // =========================================
    

    // 2. Free memory allocated on host
    // =========================================
}

#endif // VC3_PHYS_KSNP_RUNNER
