#ifndef VC3_PHYS_KSNP_LOGDATA
#define VC3_PHYS_KSNP_LOGDATA

#include <iostream>

#include "../include/math/containers/matrix.cu"
#include "setup.cu"

namespace vc3_phys
{

struct logData
{
    // variables
    flt2 timeSimulator_runSimulations;
    flt2 timeRunner_runSimulation;
    flt2 timeRunner_runOneStep;
    flt2 timeRunner_collectStat;
    flt2 timeSolver_moveParticles;
    flt2 timeSolver_leaveScentMarks;
    flt2 timeSolver_postParticles;
    flt2 timeSolver_storeHistory;

    // member functions

    /** Initialize data structures based on setup parameters **/
    int initialize();

    /** Add this object's data to another logData object **/
    int addTo(logData* addto) const;
};

int logData::initialize()
{
    timeSimulator_runSimulations = 0.0;
    timeRunner_runSimulation = 0.0;
    timeRunner_runOneStep = 0.0;
    timeRunner_collectStat = 0.0;
    timeSolver_moveParticles = 0.0;
    timeSolver_leaveScentMarks = 0.0;
    timeSolver_postParticles = 0.0;
    timeSolver_storeHistory = 0.0;

    return 0; // Success
}

int logData::addTo(logData* addto) const
{
    // Safety check: pointer should not be null
    if (addto == nullptr) {
        std::cerr << "Error: 'addto' logData pointer is null." << std::endl;
        return 1; // Return an error code
    }

    // Add the number of computed iterations
    addto->timeRunner_runSimulation += this->timeRunner_runSimulation;
    addto->timeRunner_runOneStep += this->timeRunner_runOneStep;
    addto->timeRunner_collectStat += this->timeRunner_collectStat;
    addto->timeSolver_moveParticles += this->timeSolver_moveParticles;
    addto->timeSolver_leaveScentMarks += this->timeSolver_leaveScentMarks;
    addto->timeSolver_postParticles += this->timeSolver_postParticles;
    addto->timeSolver_storeHistory += this->timeSolver_storeHistory;
    
    return 0; // Success
}

} // namespace vc3_phys

#endif // VC3_PHYS_KSNP_LOGDATA