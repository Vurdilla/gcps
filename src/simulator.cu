#ifndef VC3_PHYS_KSNP_SIMULATOR
#define VC3_PHYS_KSNP_SIMULATOR

#include <iostream>
#include <sstream>
#include <fstream>
#include <iomanip>
#include <string>
#include <vector>

#include "../include/math/containers/matrix.cu"

#include "runner.cu"


namespace vc3_phys
{

class KSNPSimulator
{

public:
    /** Constructor
    **/
    KSNPSimulator(setupParameters setup, int nStreams = 1, int randseed=0, int cudaDeviceID=0) noexcept;

    /** Run simulations
    **/
    int runSimulations() noexcept;

    /** Write statistics to file
    **/
    int writeStat(std::string outfname, bool append = false) noexcept;


protected:
    vc3_phys::setupParameters _setup;
    int _cudaDeviceID;
    int _nStreams;
    int _randseed;
    // All iterations statistics stored in a simData object.
    simData _totalResults;
    logData _totalTiming;
    vc3_general::timer_ms_cuda _simulatorTimer;
};

}// namespace vc3_phys


vc3_phys::KSNPSimulator::KSNPSimulator(vc3_phys::setupParameters setup, int nStreams, int randseed, int cudaDeviceID) noexcept
{
    std::cout << "\n";
    _setup = setup;
    _cudaDeviceID = cudaDeviceID;
    _randseed = randseed;
    _nStreams = min(nStreams, _setup.nIterations);

    // Initialize the results data structure
    _totalResults.initialize(setup);
    _totalTiming.initialize();

    cudaError_t cuerr = cudaSetDevice(_cudaDeviceID);
    if (cuerr == cudaSuccess)
    {
        std::cout << "Using CUDA device #" << _cudaDeviceID << "\n";
    }
    else
    {
        std::cout << "CUDA error: can not use CUDA device #" << _cudaDeviceID << "!\n";
        return;
    }

    std::cout << "Using " << _nStreams << " streams in parallel for " << _setup.nIterations << " iterations\n";
}

int vc3_phys::KSNPSimulator::runSimulations() noexcept
{
    _simulatorTimer.start();
    std::cout << "\nKSNPSimulator::runSimulations() preparing simulations\n";
    std::cout.flush();

    int num_threads = _nStreams;
    #pragma omp parallel num_threads(num_threads)
    {
        cudaStream_t stream1;
        cudaStreamCreateWithFlags(&stream1, cudaStreamNonBlocking);

        #pragma omp for schedule(static)
        for (int iter = 0; iter < _setup.nIterations; iter++)
        {
            if (iter == 0)
            {
                std::cout << "CURRENT DEVICE: " << _cudaDeviceID << "\n";
                std::cout.flush();
            }
            vc3_general::timer_ms_cuda iterTimer;
            // 6.0 Selecting a device and a stream
            int device;
            //cudaSetDevice(_cudaDeviceID);
            cudaGetDevice(&device);
            std::cout << "Run iteration # " << iter << " on GPU # " << device << "\n";
            std::cout.flush();

            // 6.3. Initialize KSNPRunner
            std::cout << "Iteration # " << iter << ": initialize KSNPRunner\n";
            std::cout.flush();
            vc3_phys::KSNPRunner runner(stream1, _setup, _randseed, iter);

            // 6.4. Run optimization
            std::cout << "Iteration # " << iter << ": run simulations\n";
            std::cout.flush();
            runner.runSimulation(stream1);
            #pragma omp critical
            {
                // Add the data from the completed iteration into the united matrix.
                runner.addResultsTo(&_totalResults);
                runner.addTimingTo(&_totalTiming);
            }
            std::cout << "Iteration # " << iter << ": done in X s\n";
            std::cout.flush();
        }

        // Each thread destroys its stream after finishing all its assigned iterations.
        cudaStreamDestroy(stream1);

    } // #pragma omp parallel num_threads(num_threads)

    _simulatorTimer.stop();
    _totalTiming.timeSimulator_runSimulations += _simulatorTimer.get_dtime(0);
    
    return 0;
}

int vc3_phys::KSNPSimulator::writeStat(std::string outfname, bool append) noexcept
{
    // Verify final count on console
    std::cout << "\nTotal iterations requested: " << _setup.nIterations
        << ", Total iterations computed and summed: " << _totalResults.iterationsComputed << std::endl;

    std::string fileName;
    std::ofstream outf;

    /** $ _log.txt **/
    // This file contains:
    // - timing
    fileName = outfname + "_log.txt";
    std::cout << "Opening output file \"" << fileName << "\"\n";
    if (append) outf.open(fileName.c_str(), std::ios::app);
    else outf.open(fileName.c_str());
    if (!outf)
    {
        std::cout << "\nCan not open file \"" << fileName << "\"";
        return 1;
    }
    outf << _totalResults.iterationsComputed << " iterations computed\n";
    outf << "Total time of simulatiuons, ms\t" << _totalTiming.timeSimulator_runSimulations << "\n";
    outf << "* Time breakdown by routines *\n";
    outf << "runSimulation() time, ms\t" << _totalTiming.timeRunner_runSimulation << "\n";
    outf << "runOneStep() time, ms\t" << _totalTiming.timeRunner_runOneStep << "\n";
    outf << "collectStat() time, ms\t" << _totalTiming.timeRunner_collectStat << "\n";
    outf << "runOneStep - moveParticles() time, ms\t" << _totalTiming.timeSolver_moveParticles << "\n";
    outf << "runOneStep - leaveScentMarks() time, ms\t" << _totalTiming.timeSolver_leaveScentMarks << "\n";
    outf << "runOneStep - postParticles() time, ms\t" << _totalTiming.timeSolver_postParticles << "\n";
    outf << "runOneStep - storeHistory() time, ms\t" << _totalTiming.timeSolver_storeHistory << "\n";
    outf << "\n";
#ifdef KERNELTIME
    outf << "Kernel timing\n";
    kernelTimerAsync::getInstance().print_summary(outf);
    outf << "\n";
#endif
    outf.close();
    /** / _log.txt **/

    /** $ _PCcumulative.txt **/
    // This file contains:
    // - Cumulative particle concentrations from all iterations
    fileName = outfname + "_PCcumulative.txt";
    std::cout << "\nWriting total statistics to file \"" << fileName << "\"\n";
    if (append) outf.open(fileName.c_str(), std::ios::app);
    else outf.open(fileName.c_str());
    if (!outf)
    {
        std::cout << "\nCan not open file \"" << fileName << "\"";
        return 1;
    }

    flt2 dl = _setup.boxSize / flt2(_setup.latticeSize - 1);

    outf << "CumulativeParticleConcentration\n"
        << "1\t1\t1\n"
        << "PCsize\t"<< _totalResults.cumulativeParticleCountMatrix.nRow()<<"\t"<< _totalResults.cumulativeParticleCountMatrix.nCol()<<"\n";

    outf << "\nrecordName" << "\n"
        << "Time\t" << flt2(_setup.nSteps) * _setup.timeStep << "\n";
    outf << "BoxSize\t" << _setup.boxSize << "\n";
    outf << "Time" << flt2(_setup.nSteps) * _setup.timeStep;
    for (int bj = 0; bj < _totalResults.cumulativeParticleCountMatrix.nCol(); bj++)
        outf << "\t" << (flt2(bj) + 0.50) * dl * flt2(_setup.cumulativePCblockSize);
    outf << "\n";

    flt2 cumulativeTime = _setup.nSteps - (_setup.cumulativeDataMinTime / _setup.timeStep);
    if (cumulativeTime <= 0) cumulativeTime = 1.0;
    flt2 cpcNorm = _setup.nIterations * cumulativeTime;

    for (int bi = 0; bi < _totalResults.cumulativeParticleCountMatrix.nRow(); bi++)
    {
        outf << (flt2(bi) + 0.50) * dl * flt2(_setup.cumulativePCblockSize);
        for (int bj = 0; bj < _totalResults.cumulativeParticleCountMatrix.nCol(); bj++)
            outf << "\t" << flt2(_totalResults.cumulativeParticleCountMatrix(bi, bj)) / cpcNorm;
        outf << "\n";
    }
    outf << "\n";
    outf.close();
    /** / _PCcumulative.txt **/

    /** $ _PCreg.txt **/
    // This file contains:
    // - Time regular particle concentrations
    fileName = outfname + "_PCreg.txt";
    std::cout << "Opening output file \"" << fileName << "\"\n";
    if (append) outf.open(fileName.c_str(), std::ios::app);
    else outf.open(fileName.c_str());
    if (!outf)
    {
        std::cout << "\nCan not open file \"" << fileName << "\"";
        return 1;
    }
    outf << "TimeRegularParticleConcentration\n"
        << _totalResults.timeRegParticleCountMatrix.size()<< "\t1\t" << _totalResults.timeRegParticleCountMatrix.size() << "\n"
        << "PCsize\t" << _totalResults.timeRegParticleCountMatrix[0].nRow() << "\t" << _totalResults.timeRegParticleCountMatrix[0].nCol() << "\n";
    outf << "\n";
    for (int q = 0; q < _totalResults.timeRegParticleCountMatrix.size(); q++)
    {
        outf << "recordName" << "\n"
            << "Time\t" << _setup.PCReg_startstep + _setup.PCReg_window * double(q + 1) << "\n";
        outf << "BoxSize\t" << _setup.boxSize << "\n";
        outf << "Time" << (_setup.PCReg_startstep + _setup.PCReg_window * _setup.timeStep) * double(q + 1);
        for (int bj = 0; bj < _totalResults.timeRegParticleCountMatrix[q].nCol(); bj++)
            outf << "\t" << (double(bj) + 0.50) * dl * double(_setup.PCReg_blockSize);
        outf << "\n";
        double pcRegNorm = _setup.nIterations * _setup.PCReg_window;
        for (int bi = 0; bi < _totalResults.timeRegParticleCountMatrix[q].nRow(); bi++)
        {
            outf << (double(bi) + 0.50) * dl * double(_setup.PCReg_blockSize);
            for (int bj = 0; bj < _totalResults.timeRegParticleCountMatrix[q].nCol(); bj++)
                outf << "\t" << double(_totalResults.timeRegParticleCountMatrix[q](bi, bj)) / pcRegNorm;
            outf << "\n";
        }
        outf << "\n";
    }
    outf.close();
    /** / _PCreg.txt **/

    /** $ _SCreg.txt **/
    // This file contains:
    // - Time regular scent concentrations
    fileName = outfname + "_SCreg.txt";
    std::cout << "Opening output file \"" << fileName << "\"\n";
    if (append) outf.open(fileName.c_str(), std::ios::app);
    else outf.open(fileName.c_str());
    if (!outf)
    {
        std::cout << "\nCan not open file \"" << fileName << "\"";
        return 1;
    }
    outf << "TimeRegularScentConcentration\n"
        << _totalResults.timeRegScentMatrix.size() << "\t1\t" << _totalResults.timeRegScentMatrix.size() << "\n"
        << "SCsize\t" << _totalResults.timeRegScentMatrix[0].nRow() << "\t" << _totalResults.timeRegScentMatrix[0].nCol() << "\n";
    outf << "\n";
    for (int q = 0; q < _totalResults.timeRegScentMatrix.size(); q++)
    {
        outf << "recordName" << "\n"
            << "Time\t" << _setup.SCReg_startstep + _setup.SCReg_window * double(q + 1) << "\n";
        outf << "BoxSize\t" << _setup.boxSize << "\n";
        outf << "Time" << (_setup.SCReg_startstep + _setup.SCReg_window * _setup.timeStep) * double(q + 1);
        for (int bj = 0; bj < _totalResults.timeRegScentMatrix[q].nCol(); bj++)
            outf << "\t" << (double(bj) + 0.50) * dl * double(_setup.SCReg_blockSize);
        outf << "\n";
        double scRegNorm = _setup.nIterations;
        for (int bi = 0; bi < _totalResults.timeRegScentMatrix[q].nRow(); bi++)
        {
            outf << (double(bi) + 0.50) * dl * double(_setup.SCReg_blockSize);
            for (int bj = 0; bj < _totalResults.timeRegScentMatrix[q].nCol(); bj++)
                outf << "\t" << double(_totalResults.timeRegScentMatrix[q](bi, bj)) / scRegNorm;
            outf << "\n";
        }
        outf << "\n";
    }
    outf.close();
    /** / _SCreg.txt **/

    /** $ _eventsReg.txt **/
    // This file contains:
    // - Cumulative particle concentrations
    fileName = outfname + "_eventsReg.txt";
    std::cout << "Opening output file \"" << fileName << "\"\n";
    if (append) outf.open(fileName.c_str(), std::ios::app);
    else outf.open(fileName.c_str());
    if (!outf)
    {
        std::cout << "\nCan not open file \"" << fileName << "\"";
        return 1;
    }
    outf << "\n" << "recordName" << "\n";
    //outf << "Total number of trjs\t" << _cumulativeNTrjs << "\n";
    //outf << "Avg number of trjs per iteration\t" << double(_cumulativeNTrjs) / double(_param.nIterations) << "\n";
    //outf << "Avg trj duration, steps\t" << double(_param.nIterations * _param.nSteps) / double(_cumulativeNTrjs) << "\n";
    //outf << "Avg trj duration, time units\t" << (double(_param.nIterations * _param.nSteps) * _param.timeStep - _param.cumulativeDataMinTime) / double(_cumulativeNTrjs) << "\n";
    outf << "Avg events per iteration\n";
    outf << "Time";
    outf << "\tTotal Timed Reset\tTimed Reset > Timed Reset\tBoundary Hit > Timed Reset\tTarget Hit > Timed Reset\tHome hit > Timed Reset";
    outf << "\tTotal Boundary Hit\tTimed Reset > Boundary Hit\tBoundary Hit > Boundary Hit\tTarget Hit > Boundary Hit\tHome hit > Boundary Hit";
    outf << "\tTotal Target Hit\tTimed Reset > Target Hit\tBoundary Hit > Target Hit\tTarget Hit > Target Hit\tHome hit > Target Hit";
    outf << "\tTotal Home hit\tTimed Reset > Home hit\tBoundary Hit > Home hit\tTarget Hit > Home hit\tHome hit > Home hit";
    outf << "\n";
    for (int q = 0; q < _totalResults.timeRegCumulativeNTR.size(); q++)
    {
        double time = double(q * _setup.HitCountReg_window) * _setup.timeStep;
        outf << time;
        outf << "\t" << double(_totalResults.timeRegCumulativeNTR[q][0]) / double(_setup.nIterations)
            << "\t" << double(_totalResults.timeRegCumulativeNTR[q][1]) / double(_setup.nIterations)
            << "\t" << double(_totalResults.timeRegCumulativeNTR[q][2]) / double(_setup.nIterations)
            << "\t" << double(_totalResults.timeRegCumulativeNTR[q][3]) / double(_setup.nIterations)
            << "\t" << double(_totalResults.timeRegCumulativeNTR[q][4]) / double(_setup.nIterations);
        outf << "\t" << double(_totalResults.timeRegCumulativeNBoundaryHit[q][0]) / double(_setup.nIterations)
            << "\t" << double(_totalResults.timeRegCumulativeNBoundaryHit[q][1]) / double(_setup.nIterations)
            << "\t" << double(_totalResults.timeRegCumulativeNBoundaryHit[q][2]) / double(_setup.nIterations)
            << "\t" << double(_totalResults.timeRegCumulativeNBoundaryHit[q][3]) / double(_setup.nIterations)
            << "\t" << double(_totalResults.timeRegCumulativeNBoundaryHit[q][4]) / double(_setup.nIterations);
        outf << "\t" << double(_totalResults.timeRegCumulativeNTargetHit[q][0]) / double(_setup.nIterations)
            << "\t" << double(_totalResults.timeRegCumulativeNTargetHit[q][1]) / double(_setup.nIterations)
            << "\t" << double(_totalResults.timeRegCumulativeNTargetHit[q][2]) / double(_setup.nIterations)
            << "\t" << double(_totalResults.timeRegCumulativeNTargetHit[q][3]) / double(_setup.nIterations)
            << "\t" << double(_totalResults.timeRegCumulativeNTargetHit[q][4]) / double(_setup.nIterations);
        outf << "\t" << double(_totalResults.timeRegCumulativeNHomeHit[q][0]) / double(_setup.nIterations)
            << "\t" << double(_totalResults.timeRegCumulativeNHomeHit[q][1]) / double(_setup.nIterations)
            << "\t" << double(_totalResults.timeRegCumulativeNHomeHit[q][2]) / double(_setup.nIterations)
            << "\t" << double(_totalResults.timeRegCumulativeNHomeHit[q][3]) / double(_setup.nIterations)
            << "\t" << double(_totalResults.timeRegCumulativeNHomeHit[q][4]) / double(_setup.nIterations);
        outf << "\n";
    }
    outf.close();
    /** / _eventsReg.txt **/

    return 0;
}

#endif // VC3_PHYS_KSNP_SIMULATOR
