#ifndef VC3_PHYS_KSNP_GPUDATA
#define VC3_PHYS_KSNP_GPUDATA

#include <curand_kernel.h>

#include "../include/types.cu"
#include "../include/cumath/cuvector2D.cu"
#include "../include/cumath/cusizevector2D.cu"

#include "setup.cu"

namespace vc3_phys{

struct gpu_setup_variables{

    // searcher parameters
    int nParticles; // number of particles, default = 1
    int chemosensitivityModel; // chemosensitivity model: 0 - ~ gradient of SC, 1 - gradient of log(SC)
    // searcher velocity, default =  1
    int V0_distr; // 0 - single value, 1 - bimodal, 2 - uniform, 3 - log-uniform, 4 - gaussian, 5 - log-gaussian
    flt2 V0, V0_min, V0_max, V0_bias, V0_mean, V0_sigma;
    bool keepV0Constant; // use equations with constant V0, default = false
    flt2 VDecayTime; // searcher velocity relaxation rate, default = 1.0
    // rotationbal diffusion coefficient, default = 1.0
    int rotationalDiffusion_distr; // 0 - single value, 1 - bimodal, 2 - uniform, 3 - log-uniform, 4 - gaussian, 5 - log-gaussian
    flt2 rotationalDiffusion, rotationalDiffusion_min, rotationalDiffusion_max, rotationalDiffusion_bias, rotationalDiffusion_mean, rotationalDiffusion_sigma;
    flt2 Rscent; // scent raduis, default = 0.05
    // searcher chemodeposition rate, default = 0.005
    int beta0_distr; // 0 - single value, 1 - bimodal, 2 - uniform, 3 - log-uniform, 4 - gaussian, 5 - log-gaussian
    flt2 beta0, beta0_min, beta0_max, beta0_bias, beta0_mean, beta0_sigma;
    // searcher chemodeposition relaxation rate, default = 1.0
    flt2 betaDecayTime;
    // rotational chemosensitivity, default =  0.2
    int chiRot_distr; // 0 - single value, 1 - bimodal, 2 - uniform, 3 - log-uniform, 4 - gaussian, 5 - log-gaussian
    flt2 chiRot, chiRot_min, chiRot_max, chiRot_bias, chiRot_mean, chiRot_sigma;
    // translational chemosensitivity, default =  0.015
    int chiTrans_distr; // 0 - single value, 1 - bimodal, 2 - uniform, 3 - log-uniform, 4 - gaussian, 5 - log-gaussian
    flt2 chiTrans, chiTrans_min, chiTrans_max, chiTrans_bias, chiTrans_mean, chiTrans_sigma;
    // scent noise level, default = 1.00
    int SC0_distr; // 0 - single value, 1 - bimodal, 2 - uniform, 3 - log-uniform, 4 - gaussian, 5 - log-gaussian
    flt2 SC0, SC0_min, SC0_max, SC0_bias, SC0_mean, SC0_sigma;

    // pairwise searcher interactions
    flt2 PPepsilon; // LJ epsilon
    flt2 PPsigma; // LJ sigma

    // system parameters
    flt2 boxSize; // the grid on which the concentration shall be calculated is 2L-by-2L, default = 1
    int boundaryType; // boundary: 0 - circle, 1 - square, 2 - square PBC
    flt2 scentDecayTime; // scent decay time, default = 1.00
    int initialNTRtype; // initian next time time reset type: 0 - same as regular, 1 - random at (0, timedResetMeanTime)
    int initialParticlePos; // 0 - center of arena, 1 - uniform distribution across arene area

    // home potential
    int homeType; // home type: 1 - parabolic potential, 2 - conical potential
    flt2 homeRadius; // radius of the circular area from which particles instantly resets to the center
    flt2 homePotentialKT; // strength of the home potential, translational strength
    flt2 homePotentialKR; // strength of the home potential, rotational strength

    // timed reset parameters
    int timedResetTimerType; // timed reset timer type: 0 - regular time intervals (=meanResettingTime), 1 - stochastic time intervals with exp distribution
    int timedResetType; // timed reset type: 0 - position to the center + random direction, 1 - direction to the center + one timestep towards center,
    //   2 - reverse direction + one timestep backward
    flt2 timedResetMeanTime; // average time between resets, default = 50
    bool timedResetHomePotential; // does timed reset switches on home potential sensing
    flt2 globalResetTime; // time marker for all-particle reset, default -1 (not set), active when non-negative

    // boundary parameters
    int boundaryResetType; // boundary reset: 0 - position to the center + random direction, 1 - direction to the center + one timestep towards center,
    //   2 - reverse direction + one timestep backward
    flt2 betaBoundaryMult; // beta multiplication coefficient when particle hits boundary
    flt2 VBoundary; // velocity when particle hits boundary
    bool boundaryResetSRtime; // does boundary hit reset stochastic reset time or not
    bool boundaryHomePotential; // does boundary hit switches on home potential sensing

    // target parameters
    int nTargets; // number of targets (active and inavtive)

    // computation parameters
    int latticeSize; // number of nodes on each side of the grid, default = 200
    flt2 timeStep; // length of time step, default = 0.005
    flt2 Rscent_cutoffMultiplier; // scent radius cutoff multiplier (Rscent_cutoff = Rscent * Rscent_cutoffMultiplier), default = 3.0
    long long int scentDecayRescalingNsteps; // number of steps between inflation rescaling, defaul = 100000
    int gpuHistoryLength; // number of steps stored on GPU before it's flushed to CPU and analyzed

    // statistics parameters
    int SCblockSize; // block size with which SC is copied to host

    __host__ void set(const setupParameters &p);

}; // struct gpu_variables

struct gpu_precomputes {

    long long int matrixSize; // = latticeSize * latticeSize
    flt2 dl; // = boxSize / flt2(latticeSize - 1)
    flt2 invdl; // = 1.00 / dl
    flt2 halfBox; // = boxSize * 0.50
    flt2 betaDecayRate; // = gpusetup_variables->betaDecayTime > 0.00 ? exp(-gpusetup_variables->timeStep / gpusetup_variables->betaDecayTime) : 1.00
    flt2 VDecayRate; // = gpusetup_variables->VDecayTime > 0.00 ? exp(-gpusetup_variables->timeStep / gpusetup_variables->VDecayTime) : 1.00
    flt2 Rscentinv; // = 1 / Rscent
    flt2 Rscent2inv; // = 1 / Rscent / Rscent
    flt2 SCmark_cutoff; // = gpusetup_variables->Rscent * gpusetup_variables->Rscent_cutoffMultiplier
    flt2 SCmark_cutoff2; // = SCmark_cutoff * SCmark_cutoff
    int SCmark_dl_cutoff; // = SCmark_cutoff / dl [+1]
    int SCmark_window; // = 2 * SCmark_dl_cutoff + 1
    int SCmark_size; // = SCmark_window * SCmark_window
    flt2 PPcutoff2; // pairwise interaction cutoff radius squared
    // Spatial Hashing
    flt2 PPcell_size;
    flt2 PPinv_cell_size;
    int PPgrid_width;
    int PPhash_table_size;
    flt2 SCdecayRate; // = gpusetup_variables->scentDecayTime > 0.00 ? exp(-gpusetup_variables->timeStep / gpusetup_variables->scentDecayTime) : 1.00
    flt2 scentDivideRateInv; // = 1.00 / SCdecayRate
    int SCblockLatticeSize; // block lattice size which SC is copied to host

    __host__ void set(const setupParameters& p);

}; //struct gpu_precomputes

struct gpu_targets{
    // $ Per target

    /** Is target active
    * Array of bool, size: nTargets **/
    bool* target_active;

    /** Was target deactivated
    * Array of bool, size: nTargets **/
    bool* target_deactivated;

    /** Target position
    * Array of cuvector2D, size: nTargets **/
    vc3_cumath::planar::cuvector* target_pos;

    /** Target radius and raduis^2
    * Array of flt2, size: nTargets **/
    flt2* target_radius;
    flt2* target_radius2;

    /** Target weight
    * Array of int, size: nTargets **/
    int* target_hit_max;

    /** Target hit count
    * Array of int, size: nTargets **/
    int* target_hit_count;

    /** Target appear time
    * Array of flt2, size: nTargets **/
    flt2* target_appearTime;

    /** Target termination type
    * Array of int, size: nTargets **/
    int* target_terminationType;

    /** Target termination time
    * Array of flt2, size: nTargets **/
    flt2* target_terminationTime;

    /** Target particle reset type
    * Array of int, size: nTargets **/
    int* target_resetType;

    /** Target beta multiplication
    * Array of flt2, size: nTargets **/
    flt2* target_betaMult;

    /** Target particle velocity after hit
    * Array of flt2, size: nTargets **/
    flt2* target_V;

    /** Is target hit resets stochastic reset time or not
    * Array of int, size: nTargets **/
    int* target_resetSRtime;

    /** Does target hit switches on home potential sensing
    * Array of int, size: nTargets **/
    int* target_homeType;

}; // struct gpu_targets

struct gpu_variables{

    // $ Single entries

    /** Current step number **/
    long long int currentStep;

    /** Current time **/
    flt2 currentTime;

    /** Current scent divide factor **/
    flt2 scentDivideFactor;

    /** Steps since last scent divide factor renormalization **/
    long long int stepsSinceScentDivideFactorRenorm;

    /** History write offset **/
    int historyOffset;

    /** Errors **/
    int error_vnan; // velocity is nan after calculations
    int error_posnan; // position is nan after calculations
    int error_omeganan; //rotational velocity is nan after calculations
    int error_bhposnan; // position is nan after calculations

}; // struct gpu_variables

struct gpu_matrixes {

    /** Scent concentration matrix
    * Array of flt2, size: latticeSize x latticeSize **/
    flt2 *SC;

    /** Coarsened (blocked) scent concentration matrix
    * Array of flt2, size: SCblockLatticeSize x SCblockLatticeSize **/
    flt2* SCblocked;

}; // struct gpu_matrixes

struct gpu_particles{
    // $ Per particle

    vc3_cumath::planar::cuvector *particle_pos; // Particle position
    vc3_cumath::planar::cusizevector *particle_posLattice; // Particle lattice positions
    vc3_cumath::planar::cuvector *particle_dpos; // Current step particle position change
    flt2 *particle_angle; // Particle orientation angles, radians
    vc3_cumath::planar::cuvector *particle_rot; // Particle orientation angle vector
    flt2 *particle_SC; // Scent value at particle position
    vc3_cumath::planar::cuvector *particle_GSC; // Scent gradient at particle position
    vc3_cumath::planar::cuvector* particle_PPforce; // Particle-particle interaction force
    vc3_cumath::planar::cuvector* particle_PPforceSpatialHashing; // // Particle-particle interaction force, spatial hashing computed
    flt2 *particle_V0; // Particle default velocity
    flt2 *particle_V; // Particle current velocity
    flt2 *particle_beta0; // Particle default chemoattractant deposition rate, default
    flt2 *particle_beta; // Particle chemoattractant deposition rate, current
    flt2 *particle_chiT; // Particle translational chemosensitivity
    flt2 *particle_chiR; // Particle rotational chemosensitivity
    flt2 *particle_c0; // Particle chemosensitivity noise level
    flt2 *particle_DR; // Particle rotational diffusion coefficient
    flt2 *particle_NRT; // Particle next reset time
    bool *particle_flag_boundaryHit; // Particle flags, boundary hit
    int *particle_flag_targetHit; // Particle flags, target hit, -1 means no hit, >=0 means target ID
    bool *particle_flag_timedReset; // Particle flags, timed reset
    bool *particle_flag_wasResetToCenter; // Particle flags, was reset to center (due to any possible reason)
    curandState *particle_curandstate; // Random number generator 
    int *particle_ID; // Particle ID
    // Arrays for the linked-list hash grid
    int *particle_next_in_cell;
    vc3_cumath::planar::cusizevector* particle_cell_coord;

}; //struct gpu_particles

struct gpu_ppinteractions
{
    int* grid_cell_heads;
}; //struct gpu_ppinteractions

struct gpu_particle_stats
{
    /** Particle position history
    * Array of cuvectors, size: nParticles x gpuHistoryLength **/
    vc3_cumath::planar::cuvector* particle_pos_history;

    /** Particle internal direction angle history
    * Array of flt2, size: nParticles x gpuHistoryLength **/
    flt2* particle_angle_history;

    /** Particle current velocity history
    * Array of flt2, size: nParticles x gpuHistoryLength **/
    flt2* particle_V_history;

    /** Particle current beta history
    * Array of flt2, size: nParticles x gpuHistoryLength **/
    flt2* particle_beta_history;

    /** Particle flags history
    * Array of bool, size: nParticles x gpuHistoryLength **/
    bool* particle_flag_boundaryHit_history; // boundary hit
    int* particle_flag_targetHit_history; // target hit, -1 means no hit, >=0 means target ID
    bool* particle_flag_timedReset_history; // timed reset
    bool* particle_flag_wasResetToCenter_history; // particle was reset to center (due to any possible reason)

}; //struct gpu_particle_stats



__host__ __device__ inline long long int get_SCoffset(int nx, int ny, int latticeSize)
{
    /*! Changing SC offset style will affect (to the level of break):
    * __kernel_moveParticles - calculateGradientBicubic()
    !*/
    return (long long int)nx * latticeSize + ny;
}

__host__ void gpu_setup_variables::set(const setupParameters &p)
{
    // searcher parameters
    nParticles = p.nParticles; // number of particles, default = 1
    chemosensitivityModel = p.chemosensitivityModel; // chemosensitivity model: 0 - ~ gradient of SC, 1 - gradient of log(SC)
    // searcher velocity
    V0_distr = p.V0_distr;
    V0 = p.V0;
    V0_min = p.V0_min;
    V0_max = p.V0_max;
    V0_bias = p.V0_bias;
    V0_mean = p.V0_mean;
    V0_sigma = p.V0_sigma;
    keepV0Constant = p.keepV0Constant; // use equations with constant V0, default = false
    VDecayTime = p.VDecayTime; // searcher velocity relaxation rate, default = 1.0
    // rotationbal diffusion coefficient
    rotationalDiffusion_distr = p.rotationalDiffusion_distr;
    rotationalDiffusion = p.rotationalDiffusion;
    rotationalDiffusion_min = p.rotationalDiffusion_min;
    rotationalDiffusion_max = p.rotationalDiffusion_max;
    rotationalDiffusion_bias = p.rotationalDiffusion_bias;
    rotationalDiffusion_mean = p.rotationalDiffusion_mean;
    rotationalDiffusion_sigma = p.rotationalDiffusion_sigma;
    Rscent = p.Rscent; // scent raduis, default = 0.05
    // searcher chemodeposition rate
    beta0_distr = p.beta0_distr;
    beta0 = p.beta0;
    beta0_min = p.beta0_min;
    beta0_max = p.beta0_max;
    beta0_bias = p.beta0_bias;
    beta0_mean = p.beta0_mean;
    beta0_sigma = p.beta0_sigma;
    // searcher chemodeposition relaxation rate
    betaDecayTime = p.betaDecayTime;
    // rotational chemosensitivity
    chiRot_distr = p.chiRot_distr;
    chiRot = p.chiRot;
    chiRot_min = p.chiRot_min;
    chiRot_max = p.chiRot_max;
    chiRot_bias = p.chiRot_bias;
    chiRot_mean = p.chiRot_mean;
    chiRot_sigma = p.chiRot_sigma;
    // translational chemosensitivity
    chiTrans_distr = p.chiTrans_distr;
    chiTrans = p.chiTrans;
    chiTrans_min = p.chiTrans_min;
    chiTrans_max = p.chiTrans_max;
    chiTrans_bias = p.chiTrans_bias;
    chiTrans_mean = p.chiTrans_mean;
    chiTrans_sigma = p.chiTrans_sigma;
    // scent noise level
    SC0_distr = p.SC0_distr;
    SC0 = p.SC0;
    SC0_min = p.SC0_min;
    SC0_max = p.SC0_max;
    SC0_bias = p.SC0_bias;
    SC0_mean = p.SC0_mean;
    SC0_sigma = p.SC0_sigma;

    // pairwise searcher interactions
    PPepsilon = p.PPepsilon; // LJ epsilon
    PPsigma = p.PPsigma; // LJ sigma

    // system parameters
    boxSize = p.boxSize; // the grid on which the concentration shall be calculated is 2L-by-2L, default = 1
    boundaryType = p.boundaryType; // boundary: 0 - square, 1 - circle
    scentDecayTime = p.scentDecayTime; // scent decay time, default = 1.00
    initialNTRtype = p.initialNTRtype; // initian next time time reset type: 0 - same as regular, 1 - random at (0, timedResetMeanTime)
    initialParticlePos = p.initialParticlePos; // 0 - center of arena, 1 - uniform distribution across arena area

    // home potential
    homeType = p.homeType; // home type: 1 - parabolic potential, 2 - conical potential
    homeRadius = p.homeRadius; // radius of the circular area from which particles instantly resets to the center
    homePotentialKT = p.homePotentialKT; // strength of the home potential, translational strength
    homePotentialKR = p.homePotentialKR; // strength of the home potential, rotational strength

    // timed reset parameters
    timedResetTimerType = p.timedResetTimerType; // timed reset timer type: 0 - regular time intervals (=meanResettingTime), 1 - stochastic time intervals with exp distribution
    timedResetType = p.timedResetType; // timed reset type: 0 - position to the center + random direction, 1 - direction to the center + one timestep towards center,
    //   2 - reverse direction + one timestep backward
    timedResetMeanTime = p.timedResetMeanTime; // average time between resets, default = 50
    timedResetHomePotential = p.timedResetHomePotential; // does timed reset switches on home potential sensing
    globalResetTime = p.globalResetTime; // time marker for all-particle reset, default -1 (not set), active when non-negative

    // boundary parameters
    boundaryResetType = p.boundaryResetType; // boundary reset: 0 - position to the center + random direction, 1 - direction to the center + one timestep towards center,
    //   2 - reverse direction + one timestep backward
    betaBoundaryMult = p.betaBoundaryMult; // beta multiplication coefficient when particle hits boundary
    VBoundary = p.VBoundary; // velocity when particle hits boundary
    boundaryResetSRtime = p.boundaryResetSRtime; // does boundary hit reset stochastic reset time or not
    boundaryHomePotential = p.boundaryHomePotential; // does boundary hit switches on home potential sensing

    // target parameters
    nTargets = p.targetList.size(); // number of targets (active and inactive)
    if (nTargets == 0) nTargets = 1; // imaginary target will be generated to avoid zero array size problems

    // computation parameters
    latticeSize = p.latticeSize; // number of nodes on each side of the grid, default = 200
    timeStep = p.timeStep; // length of time step, default = 0.005
    Rscent_cutoffMultiplier = p.Rscent_cutoffMultiplier; // scent radius cutoff multiplier (Rscent_cutoff = Rscent * Rscent_cutoffMultiplier), default = 3.0
    scentDecayRescalingNsteps = p.scentDecayRescalingNsteps; // number of steps between inflation rescaling, defaul = 100000
    gpuHistoryLength = p.gpuHistoryLength; // number of steps stored on GPU before it's flushed to CPU and analyzed

    // statistics parameters
    SCblockSize = p.SCReg_blockSize; // block size with which SC is copied to host
}

__host__ void gpu_precomputes::set(const setupParameters& p)
{
    matrixSize = (long long int)p.latticeSize * p.latticeSize;
    dl = p.boxSize / flt2(p.latticeSize - 1);
    invdl = 1.00 / dl;
    halfBox = p.boxSize * 0.50;
    // Beta decay: positive = there is a decay, zero = no decay, negative = reset to beta0 every timestep
    betaDecayRate = 0.00;
    if(p.betaDecayTime > 0.00) betaDecayRate = exp(-p.timeStep / p.betaDecayTime);
    else if(p.betaDecayTime < 0.00) betaDecayRate = -1.00;
    // Velocity decay: positive = there is a decay, zero = no decay, negative = reset to v0 every timestep
    VDecayRate = 0.00;
    if(p.VDecayTime > 0.00) VDecayRate = exp(-p.timeStep / p.VDecayTime);
    else if(p.VDecayTime < 0.00) VDecayRate = -1.00;
    Rscentinv = 1.00 / p.Rscent;
    Rscent2inv = 1.00 / p.Rscent / p.Rscent;
    SCmark_cutoff = p.Rscent * p.Rscent_cutoffMultiplier;
    SCmark_cutoff2 = SCmark_cutoff * SCmark_cutoff;
    SCmark_dl_cutoff = SCmark_cutoff / dl;
    if (SCmark_dl_cutoff * dl < SCmark_cutoff) SCmark_dl_cutoff++;
    SCmark_window = 2 * SCmark_dl_cutoff + 1;
    SCmark_size = SCmark_window * SCmark_window;
    PPcutoff2 = p.PPsigma * p.PPsigma;
    if (p.PPepsilon > 0.0)
    {
        PPcell_size = p.PPsigma;
        PPinv_cell_size = 1.0 / PPcell_size;
        PPgrid_width = ceilf(p.boxSize / PPcell_size);
        PPhash_table_size = 1;
        while (PPhash_table_size < p.nParticles * 2) PPhash_table_size *= 2;
    }
    else PPhash_table_size = 0;
    SCdecayRate = p.scentDecayTime > 0.00 ? exp(-p.timeStep / p.scentDecayTime) : 1.00;
    scentDivideRateInv = 1.00 / SCdecayRate;
    SCblockLatticeSize = (p.latticeSize + p.SCReg_blockSize - 1) / p.SCReg_blockSize;
}

} //namespace vc3_phys

// Declare device constants
__constant__ vc3_phys::gpu_setup_variables __gpusetup_variables;
__constant__ vc3_phys::gpu_precomputes __gpuprecomputes;

#endif
