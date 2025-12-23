#ifndef VC3_PHYS_KSNP_KERNELS_UPDATETARGETS
#define VC3_PHYS_KSNP_KERNELS_UPDATETARGETS


#include "../include/types.cu"
#include "../include/cumath/cumath_consts.cu"
#include "../include/cumath/cuvector2D.cu"
#include "../include/cumath/cusizevector2D.cu"

#include "gpudata.cu"



/** Device kernel
    Produce set of procedures to move particles across the arena

    To be ran with Nblocks=((nTargets + threadsPerBlock - 1) / threadsPerBlock) and Nthreads=(threadsPerBlock)
    threadsPerBlock = 256 or 512 or 1024, depending on architechture
**/
__global__ void __kernel_updateTargets(vc3_phys::gpu_variables* gpuvariables, 
    vc3_phys::gpu_targets* gputargets)
{
    int toffset = blockIdx.x * blockDim.x + threadIdx.x;
    if (toffset >= __gpusetup_variables.nTargets) return;
    
    if (gputargets->target_active[toffset])
    { // 1. Check if an active target becomes inactive  
        if (gputargets->target_terminationType[toffset] == 0)
        { // 0 - terminate by time
            if (gpuvariables->currentTime >= gputargets->target_terminationTime[toffset])
            {
                gputargets->target_active[toffset] = false;
                printf("target inactive: step=%lld, time=%.15le, tID=%d\n", gpuvariables->currentStep, gpuvariables->currentTime, toffset);
            }
        }
        else if (gputargets->target_terminationType[toffset] == 1)
        { // 1 - terminate after capture
            if (gputargets->target_hit_count[toffset] >= gputargets->target_hit_max[toffset])
            {
                gputargets->target_active[toffset] = false;
                printf("target inactive: step=%lld, time=%.15le, tID=%d\n", gpuvariables->currentStep, gpuvariables->currentTime, toffset);
            }
        }
    }
    else
    { // 2. Check if an inactive target becomes active
        if (gputargets->target_appearTime[toffset] >= gpuvariables->currentTime - __gpusetup_variables.timeStep
            && gputargets->target_appearTime[toffset] < gpuvariables->currentTime)
        {
            gputargets->target_active[toffset] = true;
            gputargets->target_hit_count[toffset] = 0;
            printf("target active: step=%lld, time=%.15le, tID=%d\n", gpuvariables->currentStep, gpuvariables->currentTime, toffset);
        }
    }
}


#endif // VC3_PHYS_KSNP_KERNELS_UPDATETARGETS
