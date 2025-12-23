#ifndef VC3_PHYS_KSNP_KERNELS_INIT
#define VC3_PHYS_KSNP_KERNELS_INIT


#include "../include/types.cu"
#include "../include/cumath/cumath_consts.cu"
#include "../include/cumath/cuvector2D.cu"
#include "../include/cumath/cusizevector2D.cu"

#include "gpudata.cu"



/** Device kernel
    Initializes values in GPU matrixes arrays (gpu_matrixes)

    To be ran with Nblocks=((latticeSize2 + threadsPerBlock - 1) / threadsPerBlock) and Nthreads=(threadsPerBlock)
    threadsPerBlock = 256 or 512 or 1024, depending on architechture

    Obsolete
    Works for latticeSize < 46000 (32 bit -> 16GB limit)
**/
__global__ void __kernel_init_matrixes(vc3_phys::gpu_matrixes* __restrict__ gpumatrixes)
{
    long long int scoffset = blockIdx.x * blockDim.x + threadIdx.x;
    if (scoffset < __gpuprecomputes.matrixSize)
        gpumatrixes->SC[scoffset] = 0.00;
}

/** Device kernel
    Initializes values in GPU matrixes arrays (gpu_matrixes)

    To be ran with Nblocks=... and Nthreads=...
**/
__global__ void __kernel_init_matrixes_2D(vc3_phys::gpu_matrixes* __restrict__ gpumatrixes)
{
    // Calculate the unique 2D global indices for this thread
    int nx = blockIdx.x * blockDim.x + threadIdx.x;
    int ny = blockIdx.y * blockDim.y + threadIdx.y;

    // Boundary check to ensure we don't write past the matrix edges
    if (nx < __gpusetup_variables.latticeSize && ny < __gpusetup_variables.latticeSize)
    {
        long long int scoffset = vc3_phys::get_SCoffset(nx, ny, __gpusetup_variables.latticeSize);
        gpumatrixes->SC[scoffset] = 0.00;
    }
}

/** Device kernel
    Initializes values in GPU particles arrays (gpu_particles)

    To be ran with Nblocks=((nParticles + threadsPerBlock - 1) / threadsPerBlock) and Nthreads=(threadsPerBlock)
    threadsPerBlock = 256 or 512 or 1024, depending on architechture
**/
__global__ void __kernel_init_particles(vc3_phys::gpu_particles* __restrict__ gpuparticles, int randseed)
{
    int poffset = blockIdx.x * blockDim.x + threadIdx.x;
    if (poffset < __gpusetup_variables.nParticles)
    {
        /** Random number generator **/
        curand_init(randseed + poffset, poffset, 0, &(gpuparticles->particle_curandstate[poffset]));
        /** Particle position **/
        gpuparticles->particle_pos[poffset].x = __gpuprecomputes.halfBox;
        gpuparticles->particle_pos[poffset].y = __gpuprecomputes.halfBox;
        /** Particle lattice positions **/
        gpuparticles->particle_posLattice[poffset].x = gpuparticles->particle_pos[poffset].x * __gpuprecomputes.invdl;
        gpuparticles->particle_posLattice[poffset].y = gpuparticles->particle_pos[poffset].y * __gpuprecomputes.invdl;
        /** Current step particle position change **/
        gpuparticles->particle_dpos[poffset].x = 0.00;
        gpuparticles->particle_dpos[poffset].y = 0.00;
        /** Particle orientation angles, radians **/
        gpuparticles->particle_angle[poffset] = curand_uniform(&(gpuparticles->particle_curandstate[poffset])) * vc3_cumath::TwoPi;
        /** Particle orientation angle vector **/
        sincos(gpuparticles->particle_angle[poffset], &(gpuparticles->particle_rot[poffset].y), &(gpuparticles->particle_rot[poffset].x));
        /** Scent value at particle position **/
        gpuparticles->particle_SC[poffset] = 0.00;
        /** Scent gradient at particle position **/
        gpuparticles->particle_GSC[poffset].x = 0.00;
        gpuparticles->particle_GSC[poffset].y = 0.00;
        /** Particle-particle interaction force **/
        gpuparticles->particle_PPforce[poffset].x = 0.00;
        gpuparticles->particle_PPforce[poffset].y = 0.00;
        gpuparticles->particle_PPforceSpatialHashing[poffset].x = 0.00;
        gpuparticles->particle_PPforceSpatialHashing[poffset].y = 0.00;
        /** Particle default chemoattractant deposition rate **/
        gpuparticles->particle_beta0[poffset] = __gpusetup_variables.beta0;
        /** Particle chemoattractant deposition rate **/
        gpuparticles->particle_beta[poffset] = __gpusetup_variables.beta0;
        /** Particle translational chemosensitivity **/
        gpuparticles->particle_chiT[poffset] = __gpusetup_variables.chiTrans;
        /** Particle rotational chemosensitivity **/
        gpuparticles->particle_chiR[poffset] = __gpusetup_variables.chiRot;
        /** Particle chemosensitivity noise level **/
        gpuparticles->particle_c0[poffset] = __gpusetup_variables.SC0;
        /** Particle rotational diffusion coefficient **/
        gpuparticles->particle_DR[poffset] = __gpusetup_variables.rotationalDiffusion;
        /** Particle next reset time **/
        if (__gpusetup_variables.initialNTRtype == 0) 
        { // 0 - same as regular
            /*!
            This considers only regular resetting time at start
            !*/
            gpuparticles->particle_NRT[poffset] = __gpusetup_variables.timedResetMeanTime;
        }
        else if (__gpusetup_variables.initialNTRtype == 1)
        { // 1 - random at [0.00, timedResetMeanTime)
            gpuparticles->particle_NRT[poffset] = curand_uniform(&(gpuparticles->particle_curandstate[poffset])) * __gpusetup_variables.timedResetMeanTime;
        }
        /** Particle ID **/
        gpuparticles->particle_ID[poffset] = poffset;
        gpuparticles->particle_next_in_cell[poffset] = -1;
        gpuparticles->particle_cell_coord[poffset].x = 0;
        gpuparticles->particle_cell_coord[poffset].y = 0;
        /** Particle flags **/
        gpuparticles->particle_flag_boundaryHit[poffset] = false;
        gpuparticles->particle_flag_targetHit[poffset] = -1;
        gpuparticles->particle_flag_timedReset[poffset] = false;

    }
}


#endif // VC3_PHYS_KSNP_KERNELS_INIT
