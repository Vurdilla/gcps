#ifndef VC3_PHYS_KSNP_SIMDATA
#define VC3_PHYS_KSNP_SIMDATA

#include <iostream>

#include "../include/math/containers/matrix.cu"
#include "setup.cu"

namespace vc3_phys
{

struct simData
{
    // variables
    
    // Number of iterations that contributed to this data
    int iterationsComputed;

    // Cumulative particle concentration
    vc3_math::Matrix<long long int> cumulativeParticleCountMatrix;

    // Time regular window-cumulative particle concentration
    std::vector<vc3_math::Matrix<long long int>> timeRegParticleCountMatrix; // particle count over all simulations matrix, size: _param.latticeSize/PCReg_blockSize x _param.latticeSize/PCReg_blockSize

    // Time regular scent concentration
    std::vector<vc3_math::Matrix<flt2>> timeRegScentMatrix;
    
    // Cumulative counts of events time function
    std::vector<std::vector<long long int>> timeRegCumulativeNTR; // Number of timed resets: 0 - total, 1 - previous TR, 2 - previous boundary, 3 - previous target, 4 - previous home
    std::vector<std::vector<long long int>> timeRegCumulativeNBoundaryHit; // Number of boundary hits: 0 - total, 1 - previous TR, 2 - previous boundary, 3 - previous target, 4 - previous home
    std::vector<std::vector<long long int>> timeRegCumulativeNTargetHit; // Number of target hits: 0 - total, 1 - previous TR, 2 - previous boundary, 3 - previous target, 4 - previous home
    std::vector<std::vector<long long int>> timeRegCumulativeNHomeHit; // Number of target hits: 0 - total, 1 - previous TR, 2 - previous boundary, 3 - previous target, 4 - previous home


    // member functions

    /** Initialize data structures based on setup parameters **/
    int initialize(const setupParameters& setup);

    /** Add this object's data to another simData object **/
    int addTo(simData* addto) const;
};

int simData::initialize(const setupParameters& setup)
{
#ifdef DEBUG0
    std::cout << "\nsimData::initialize()\n";
    std::cout.flush();
#endif
    // Initialize the number of computed iterations to zero.
    iterationsComputed = 0;

    // Initialize Cumulative particle concentration matrix
    int cpcsize = setup.latticeSize / setup.cumulativePCblockSize;
    if (setup.latticeSize % setup.cumulativePCblockSize > 0) cpcsize++;
#ifdef DEBUG0
    std::cout << "simData::initialize() - cpcsize " << cpcsize << "\n";
        std::cout.flush();
#endif
    cumulativeParticleCountMatrix.resize(cpcsize, cpcsize);
    cumulativeParticleCountMatrix.set(0);

    // Initialize Time regular window-cumulative particle concentration
    int pcnt = (setup.nSteps - setup.PCReg_startstep) / setup.PCReg_window;
    if ((setup.nSteps - setup.PCReg_startstep) % setup.PCReg_window > 0) pcnt++;
    int pcsize = setup.latticeSize / setup.PCReg_blockSize;
    if (setup.latticeSize % setup.PCReg_blockSize > 0) pcsize++;
#ifdef DEBUG0
    std::cout << "simData::initialize() - pcnt " << pcnt << "\n";
    std::cout << "simData::initialize() - pcsize " << pcsize << "\n";
    std::cout.flush();
#endif
    timeRegParticleCountMatrix.resize(pcnt);
    for (int q = 0; q < pcnt; q++)
    {
        timeRegParticleCountMatrix[q].resize(pcsize, pcsize);
        timeRegParticleCountMatrix[q].set(0);
    }
    int scnt = (setup.nSteps - setup.SCReg_startstep) / setup.SCReg_window;
    if ((setup.nSteps - setup.SCReg_startstep) % setup.SCReg_window > 0) scnt++;
    int scsize = setup.latticeSize / setup.SCReg_blockSize;
    if (setup.latticeSize % setup.SCReg_blockSize > 0) scsize++;
#ifdef DEBUG0
    std::cout << "simData::initialize() - scnt " << scnt << "\n";
    std::cout << "simData::initialize() - scsize " << scsize << "\n";
    std::cout.flush();
#endif
    timeRegScentMatrix.resize(scnt);
    for (int q = 0; q < scnt; q++)
    {
        timeRegScentMatrix[q].resize(scsize, scsize);
        timeRegScentMatrix[q].set(0.00);
        //std::cout << "simData::initialize() - timeRegScentMatrix[" << q << "] resized to " << scsize << " x " << scsize << "\n";
        //std::cout.flush();
    }

    // Cumulative counts of events time function
    int ntrccount = setup.nSteps / setup.HitCountReg_window + 1;
#ifdef DEBUG0
    std::cout << "simData::initialize() - ntrccount " << ntrccount << "\n";
    std::cout.flush();
#endif
    timeRegCumulativeNTR.resize(ntrccount);
    timeRegCumulativeNBoundaryHit.resize(ntrccount);
    timeRegCumulativeNTargetHit.resize(ntrccount);
    timeRegCumulativeNHomeHit.resize(ntrccount);
    for (int q = 0; q < ntrccount; q++)
    {
        timeRegCumulativeNTR[q].resize(5);
        fill(timeRegCumulativeNTR[q].begin(), timeRegCumulativeNTR[q].end(), 0);
        timeRegCumulativeNBoundaryHit[q].resize(5);
        fill(timeRegCumulativeNBoundaryHit[q].begin(), timeRegCumulativeNBoundaryHit[q].end(), 0);
        timeRegCumulativeNTargetHit[q].resize(5);
        fill(timeRegCumulativeNTargetHit[q].begin(), timeRegCumulativeNTargetHit[q].end(), 0);
        timeRegCumulativeNHomeHit[q].resize(5);
        fill(timeRegCumulativeNHomeHit[q].begin(), timeRegCumulativeNHomeHit[q].end(), 0);
    }

    return 0; // Success
}

int simData::addTo(simData* addto) const
{
#ifdef DEBUG0
    std::cout << "\nsimData::addTo()\n";
    std::cout.flush();
#endif
    // Safety check: pointer should not be null
    if (addto == nullptr) {
        std::cerr << "Error: 'addto' simData pointer is null." << std::endl;
        return 1; // Return an error code
    }

    // Add the number of computed iterations.
    addto->iterationsComputed += this->iterationsComputed;

    // --- Sum cumulativeParticleCountMatrix ---
    vc3_math::Matrix<long long int>& addtoCPCM = addto->cumulativeParticleCountMatrix;
    if (addtoCPCM.nRow() != cumulativeParticleCountMatrix.nRow() || addtoCPCM.nCol() != cumulativeParticleCountMatrix.nCol()) {
        std::cerr << "Error: cumulativeParticleCountMatrix matrix dimensions do not match for summation." << std::endl;
        return 2; // Return a different error code
    }
    for (int i = 0; i < cumulativeParticleCountMatrix.nRow(); i++)
        for (int j = 0; j < cumulativeParticleCountMatrix.nCol(); j++)
            addtoCPCM(i, j) += cumulativeParticleCountMatrix(i, j);

    // --- Sum timeRegParticleCountMatrix ---
    std::vector<vc3_math::Matrix<long long int>> &addtoTRPCM = addto->timeRegParticleCountMatrix;
    if (addtoTRPCM.size() != timeRegParticleCountMatrix.size())
    {
        std::cerr << "Error: timeRegParticleCountMatrix matrix dimensions do not match for summation." << std::endl;
        return 3; // Return a different error code
    }
    for (int q = 0; q < addtoTRPCM.size(); q++)
    {
        if (addtoTRPCM[q].nRow() != timeRegParticleCountMatrix[q].nRow() || addtoTRPCM[q].nCol() != timeRegParticleCountMatrix[q].nCol()) {
            std::cerr << "Error: timeRegParticleCountMatrix matrix dimensions do not match for summation." << std::endl;
            return 3; // Return a different error code
        }
    }
    for (int q = 0; q < timeRegParticleCountMatrix.size(); q++)
    {
        for (int i = 0; i < timeRegParticleCountMatrix[q].nRow(); i++)
            for (int j = 0; j < timeRegParticleCountMatrix[q].nCol(); j++)
                addtoTRPCM[q](i, j) += timeRegParticleCountMatrix[q](i, j);
    }

    // --- Sum timeRegScentMatrix ---
    std::vector<vc3_math::Matrix<flt2>>& addtoTRSM = addto->timeRegScentMatrix;
    if (addtoTRSM.size() != timeRegScentMatrix.size())
    {
        std::cerr << "Error: timeRegScentMatrix matrix dimensions do not match for summation." << std::endl;
        return 3; // Return a different error code
    }
    for (int q = 0; q < addtoTRSM.size(); q++)
    {
        if (addtoTRSM[q].nRow() != timeRegScentMatrix[q].nRow() || addtoTRSM[q].nCol() != timeRegScentMatrix[q].nCol()) {
            std::cerr << "Error: timeRegScentMatrix matrix dimensions do not match for summation." << std::endl;
            return 3; // Return a different error code
        }
    }
    for (int q = 0; q < timeRegScentMatrix.size(); q++)
    {
        //std::cout << "simData::addTo - timeRegScentMatrix[" << q << "] size is " << timeRegScentMatrix[q].nRow() << " x " << timeRegScentMatrix[q].nCol() << "\n";
        for (int i = 0; i < timeRegScentMatrix[q].nRow(); i++)
            for (int j = 0; j < timeRegScentMatrix[q].nCol(); j++)
                addtoTRSM[q](i, j) += timeRegScentMatrix[q](i, j);
    }

    // Cumulative counts of events time function
    std::vector<std::vector<long long int>>& addtoTRCNTR = addto->timeRegCumulativeNTR;
    std::vector<std::vector<long long int>>& addtoTRCNBH = addto->timeRegCumulativeNBoundaryHit;
    std::vector<std::vector<long long int>>& addtoTRCNTH = addto->timeRegCumulativeNTargetHit;
    std::vector<std::vector<long long int>>& addtoTRCNHH = addto->timeRegCumulativeNHomeHit;
    if (addtoTRCNTR.size() != timeRegCumulativeNTR.size()
        || addtoTRCNBH.size() != timeRegCumulativeNBoundaryHit.size()
        || addtoTRCNTH.size() != timeRegCumulativeNTargetHit.size()
        || addtoTRCNHH.size() != timeRegCumulativeNHomeHit.size())
    {
        std::cerr << "Error: timeRegCumulativeEvents dimensions do not match for summation." << std::endl;
        return 4; // Return a different error code
    }
    for (int q = 0; q < addtoTRCNTR.size(); q++)
    {
        if (addtoTRCNTR[q].size() != timeRegCumulativeNTR[q].size()
            || addtoTRCNBH[q].size() != timeRegCumulativeNBoundaryHit[q].size()
            || addtoTRCNTH[q].size() != timeRegCumulativeNTargetHit[q].size()
            || addtoTRCNHH[q].size() != timeRegCumulativeNHomeHit[q].size()) {
            std::cerr << "Error: timeRegCumulativeEvents matrix dimensions do not match for summation." << std::endl;
            return 4; // Return a different error code
        }
    }
    for (int q = 0; q < addtoTRCNTR.size(); q++)
    {
        for (int w = 0; w < addtoTRCNTR[q].size(); w++)
        {
            addtoTRCNTR[q][w] += timeRegCumulativeNTR[q][w];
            addtoTRCNBH[q][w] += timeRegCumulativeNBoundaryHit[q][w];
            addtoTRCNTH[q][w] += timeRegCumulativeNTargetHit[q][w];
            addtoTRCNHH[q][w] += timeRegCumulativeNHomeHit[q][w];
        }
    }

    return 0; // Success
}

} // namespace vc3_phys

#endif // VC3_PHYS_KSNP_SIMDATA