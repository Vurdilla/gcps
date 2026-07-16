#ifndef VC3_PHYS_KSNP_KERNELS_SCENARIOSPARTICLES
#define VC3_PHYS_KSNP_KERNELS_SCENARIOSPARTICLES


#include "../include/types.cu"
#include "../include/cumath/cumath_consts.cu"
#include "../include/cumath/cubasicmath.cu"
#include "../include/cumath/cuvector2D.cu"
#include "../include/cumath/cusizevector2D.cu"

#include "gpudata.cu"



/** Device kernel
    Particle scenario that sets a particluar V0 for all the particles

    To be ran with Nblocks=((nParticles + threadsPerBlock - 1) / threadsPerBlock) and Nthreads=(threadsPerBlock)
    threadsPerBlock = 256 or 512 or 1024, depending on architechture
**/
__global__ void __kernel_scenarioParticles_V0(vc3_phys::gpu_variables* gpuvariables, 
    vc3_phys::gpu_particles* gpuparticles, flt2 particle_V0)
{
    int poffset = blockIdx.x * blockDim.x + threadIdx.x;
    if (poffset >= __gpusetup_variables.nParticles) return;
    
    
    // $ update particle_V0
    gpuparticles->particle_V0[poffset] = particle_V0;
    gpuparticles->particle_V[poffset] = particle_V0;
}


/** Device kernel
    Particle scenario that sets a particular beta0 for all the particles

    To be ran with Nblocks=((nParticles + threadsPerBlock - 1) / threadsPerBlock) and Nthreads=(threadsPerBlock)
    threadsPerBlock = 256 or 512 or 1024, depending on architechture
**/
__global__ void __kernel_scenarioParticles_beta0(vc3_phys::gpu_variables* gpuvariables,
    vc3_phys::gpu_particles* gpuparticles, flt2 particle_beta0)
{
    int poffset = blockIdx.x * blockDim.x + threadIdx.x;
    if (poffset >= __gpusetup_variables.nParticles) return;


    // $ update particle_beta0
    gpuparticles->particle_beta0[poffset] = particle_beta0;
}


/** Device kernel
    Particle scenario that sets a particular chiT for all the particles

    To be ran with Nblocks=((nParticles + threadsPerBlock - 1) / threadsPerBlock) and Nthreads=(threadsPerBlock)
    threadsPerBlock = 256 or 512 or 1024, depending on architechture
**/
__global__ void __kernel_scenarioParticles_chiT(vc3_phys::gpu_variables* gpuvariables,
    vc3_phys::gpu_particles* gpuparticles, flt2 particle_chiT)
{
    int poffset = blockIdx.x * blockDim.x + threadIdx.x;
    if (poffset >= __gpusetup_variables.nParticles) return;


    // $ update particle_chiT
    gpuparticles->particle_chiT[poffset] = particle_chiT;
}


/** Device kernel
    Particle scenario that sets a particular chiR for all the particles

    To be ran with Nblocks=((nParticles + threadsPerBlock - 1) / threadsPerBlock) and Nthreads=(threadsPerBlock)
    threadsPerBlock = 256 or 512 or 1024, depending on architechture
**/
__global__ void __kernel_scenarioParticles_chiR(vc3_phys::gpu_variables* gpuvariables,
    vc3_phys::gpu_particles* gpuparticles, flt2 particle_chiR)
{
    int poffset = blockIdx.x * blockDim.x + threadIdx.x;
    if (poffset >= __gpusetup_variables.nParticles) return;


    // $ update particle_chiR
    gpuparticles->particle_chiR[poffset] = particle_chiR;
}


/** Device kernel
    Particle scenario that sets a particular c0 for all the particles

    To be ran with Nblocks=((nParticles + threadsPerBlock - 1) / threadsPerBlock) and Nthreads=(threadsPerBlock)
    threadsPerBlock = 256 or 512 or 1024, depending on architechture
**/
__global__ void __kernel_scenarioParticles_c0(vc3_phys::gpu_variables* gpuvariables,
    vc3_phys::gpu_particles* gpuparticles, flt2 particle_c0)
{
    int poffset = blockIdx.x * blockDim.x + threadIdx.x;
    if (poffset >= __gpusetup_variables.nParticles) return;


    // $ update particle_c0
    gpuparticles->particle_c0[poffset] = particle_c0;
}


/** Device kernel
    Particle scenario that sets a particular DR for all the particles

    To be ran with Nblocks=((nParticles + threadsPerBlock - 1) / threadsPerBlock) and Nthreads=(threadsPerBlock)
    threadsPerBlock = 256 or 512 or 1024, depending on architechture
**/
__global__ void __kernel_scenarioParticles_DR(vc3_phys::gpu_variables* gpuvariables,
    vc3_phys::gpu_particles* gpuparticles, flt2 particle_DR)
{
    int poffset = blockIdx.x * blockDim.x + threadIdx.x;
    if (poffset >= __gpusetup_variables.nParticles) return;


    // $ update particle_DR
    gpuparticles->particle_DR[poffset] = particle_DR;
}


#endif // VC3_PHYS_KSNP_KERNELS_SCENARIOSPARTICLES
