#ifndef VC3_PHYS_KSNP_KERNELS_POSTPARTICLES
#define VC3_PHYS_KSNP_KERNELS_POSTPARTICLES


#include "../include/types.cu"
#include "../include/cumath/cumath_consts.cu"
#include "../include/cumath/cuvector2D.cu"
#include "../include/cumath/cusizevector2D.cu"

#include "gpudata.cu"



/** Device kernel
    Produce set of procedures after particles are moved, single thread

    To be ran with Nblocks=(1) and Nthreads=(1)
**/
__global__ void __kernel_postParticles(vc3_phys::gpu_variables* gpuvariables, 
    const vc3_phys::gpu_matrixes* __restrict__ gpumatrixes, const vc3_phys::gpu_particles* gpuparticles)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i > 0) return;

    // $ Counters increment routine
    // ============================
    gpuvariables->currentStep++;
    gpuvariables->currentTime = flt2(gpuvariables->currentStep) * __gpusetup_variables.timeStep;
    gpuvariables->historyOffset++;
    if (gpuvariables->historyOffset >= __gpusetup_variables.gpuHistoryLength) gpuvariables->historyOffset = 0;

    // Update decay coefficient
    gpuvariables->stepsSinceScentDivideFactorRenorm++;
    if (gpuvariables->stepsSinceScentDivideFactorRenorm >= __gpusetup_variables.scentDecayRescalingNsteps)
    {
        gpuvariables->scentDivideFactor = 1.00;
        gpuvariables->stepsSinceScentDivideFactorRenorm = 0;
    }
    else gpuvariables->scentDivideFactor *= __gpuprecomputes.scentDivideRateInv;
    
}


#endif // VC3_PHYS_KSNP_KERNELS_POSTPARTICLES
