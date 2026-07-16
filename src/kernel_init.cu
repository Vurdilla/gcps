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
        flt2 val, log_min, log_max;
        /** Particle position **/
        if (__gpusetup_variables.initialParticlePos == 1)
        {
            // Random uniform distribution
            if (__gpusetup_variables.boundaryType == 0) // Circular arena
            {
                // Generate uniformly inside a circle: r = R * sqrt(u)
                flt2 u1 = curand_uniform(&(gpuparticles->particle_curandstate[poffset]));
                flt2 u2 = curand_uniform(&(gpuparticles->particle_curandstate[poffset]));
                flt2 r = (__gpuprecomputes.halfBox - __gpuprecomputes.dl) * sqrtf(u1);
                flt2 theta = vc3_cumath::TwoPi * u2;
                gpuparticles->particle_pos[poffset].x = __gpuprecomputes.halfBox + r * cosf(theta);
                gpuparticles->particle_pos[poffset].y = __gpuprecomputes.halfBox + r * sinf(theta);
            }
            else // Square arena
            {
                // Generate uniformly inside the square box
                flt2 u1 = curand_uniform(&(gpuparticles->particle_curandstate[poffset]));
                flt2 u2 = curand_uniform(&(gpuparticles->particle_curandstate[poffset]));

                gpuparticles->particle_pos[poffset].x = __gpuprecomputes.dl + (__gpusetup_variables.boxSize - 2.00 * __gpuprecomputes.dl) * u1;
                gpuparticles->particle_pos[poffset].y = __gpuprecomputes.dl + (__gpusetup_variables.boxSize - 2.00 * __gpuprecomputes.dl) * u2;
            }
        }
        else 
        {
            // Center (initialParticlePos == 0 or default)
            gpuparticles->particle_pos[poffset].x = __gpuprecomputes.halfBox;
            gpuparticles->particle_pos[poffset].y = __gpuprecomputes.halfBox;
        }
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
        /** Particle velocity **/
        if (__gpusetup_variables.V0_distr == 0) // single value
            gpuparticles->particle_V0[poffset] = __gpusetup_variables.V0;
        else if (__gpusetup_variables.V0_distr == 1) // bimodal
        {
            gpuparticles->particle_V0[poffset] = (curand_uniform(&(gpuparticles->particle_curandstate[poffset])) > __gpusetup_variables.V0_bias) ?
                __gpusetup_variables.V0_min : __gpusetup_variables.V0_max;
        }
        else if (__gpusetup_variables.V0_distr == 2) // uniform
        {
            gpuparticles->particle_V0[poffset] = __gpusetup_variables.V0_min 
                + curand_uniform(&(gpuparticles->particle_curandstate[poffset])) * (__gpusetup_variables.V0_max - __gpusetup_variables.V0_min);
        }
        else if (__gpusetup_variables.V0_distr == 3) // log-uniform
        {
            gpuparticles->particle_V0[poffset] = exp10(
                log10(__gpusetup_variables.V0_min)
                + curand_uniform(&(gpuparticles->particle_curandstate[poffset])) * ( log10(__gpusetup_variables.V0_max) - log10(__gpusetup_variables.V0_min) ) );
        }
        else if (__gpusetup_variables.V0_distr == 4) // gaussian (truncated)
        {
            do {
                val = __gpusetup_variables.V0_mean
                    + __gpusetup_variables.V0_sigma *
                    curand_normal(&(gpuparticles->particle_curandstate[poffset]));
            } while (val < __gpusetup_variables.V0_min || val > __gpusetup_variables.V0_max);
            gpuparticles->particle_V0[poffset] = val;
        }
        else if (__gpusetup_variables.V0_distr == 5) // log-gaussian (truncated log-normal)
        {
            log_min = log(__gpusetup_variables.V0_min);
            log_max = log(__gpusetup_variables.V0_max);
            do {
                val = log(__gpusetup_variables.V0_mean)
                    + __gpusetup_variables.V0_sigma *
                    curand_normal(&(gpuparticles->particle_curandstate[poffset]));
            } while (val < log_min || val > log_max);
            gpuparticles->particle_V0[poffset] = exp(val);
        }
        else printf("ERR INIT: V0_distr=%d unavailable\n", __gpusetup_variables.V0_distr);
        gpuparticles->particle_V[poffset] = gpuparticles->particle_V0[poffset];
        /** Particle default chemoattractant deposition rate **/
        if (__gpusetup_variables.beta0_distr == 0) // single value
            gpuparticles->particle_beta0[poffset] = __gpusetup_variables.beta0;
        else if (__gpusetup_variables.beta0_distr == 1) // bimodal
        {
            gpuparticles->particle_beta0[poffset] = (curand_uniform(&(gpuparticles->particle_curandstate[poffset])) > __gpusetup_variables.beta0_bias) ?
                __gpusetup_variables.beta0_min : __gpusetup_variables.beta0_max;
        }
        else if (__gpusetup_variables.beta0_distr == 2) // uniform
        {
            gpuparticles->particle_beta0[poffset] = __gpusetup_variables.beta0_min
                + curand_uniform(&(gpuparticles->particle_curandstate[poffset])) * (__gpusetup_variables.beta0_max - __gpusetup_variables.beta0_min);
        }
        else if (__gpusetup_variables.beta0_distr == 3) // log-uniform
        {
            gpuparticles->particle_beta0[poffset] = exp10(
                log10(__gpusetup_variables.beta0_min)
                + curand_uniform(&(gpuparticles->particle_curandstate[poffset])) * (log10(__gpusetup_variables.beta0_max) - log10(__gpusetup_variables.beta0_min)));
        }
        else if (__gpusetup_variables.beta0_distr == 4) // gaussian (truncated)
        {
            do {
                val = __gpusetup_variables.beta0_mean
                    + __gpusetup_variables.beta0_sigma *
                    curand_normal(&(gpuparticles->particle_curandstate[poffset]));
            } while (val < __gpusetup_variables.beta0_min || val > __gpusetup_variables.beta0_max);
            gpuparticles->particle_beta0[poffset] = val;
        }
        else if (__gpusetup_variables.beta0_distr == 5) // log-gaussian (truncated log-normal)
        {
            log_min = log(__gpusetup_variables.beta0_min);
            log_max = log(__gpusetup_variables.beta0_max);
            do {
                val = log(__gpusetup_variables.beta0_mean)
                    + __gpusetup_variables.beta0_sigma *
                    curand_normal(&(gpuparticles->particle_curandstate[poffset]));
            } while (val < log_min || val > log_max);
            gpuparticles->particle_beta0[poffset] = exp(val);
        }
        else printf("ERR INIT: beta0_distr=%d unavailable\n", __gpusetup_variables.beta0_distr);
        /** Particle chemoattractant deposition rate **/
        gpuparticles->particle_beta[poffset] = gpuparticles->particle_beta0[poffset];
        /** Particle translational chemosensitivity **/
        if (__gpusetup_variables.chiTrans_distr == 0) // single value
            gpuparticles->particle_chiT[poffset] = __gpusetup_variables.chiTrans;
        else if (__gpusetup_variables.chiTrans_distr == 1) // bimodal
        {
            gpuparticles->particle_chiT[poffset] = (curand_uniform(&(gpuparticles->particle_curandstate[poffset])) > __gpusetup_variables.chiTrans_bias) ?
                __gpusetup_variables.chiTrans_min : __gpusetup_variables.chiTrans_max;
        }
        else if (__gpusetup_variables.chiTrans_distr == 2) // uniform
        {
            gpuparticles->particle_chiT[poffset] = __gpusetup_variables.chiTrans_min
                + curand_uniform(&(gpuparticles->particle_curandstate[poffset])) * (__gpusetup_variables.chiTrans_max - __gpusetup_variables.chiTrans_min);
        }
        else if (__gpusetup_variables.chiTrans_distr == 3) // log-uniform
        {
            gpuparticles->particle_chiT[poffset] = exp10(
                log10(__gpusetup_variables.chiTrans_min)
                + curand_uniform(&(gpuparticles->particle_curandstate[poffset])) * (log10(__gpusetup_variables.chiTrans_max) - log10(__gpusetup_variables.chiTrans_min)));
        }
        else if (__gpusetup_variables.chiTrans_distr == 4) // gaussian (truncated)
        {
            do {
                val = __gpusetup_variables.chiTrans_mean
                    + __gpusetup_variables.chiTrans_sigma *
                    curand_normal(&(gpuparticles->particle_curandstate[poffset]));
            } while (val < __gpusetup_variables.chiTrans_min || val > __gpusetup_variables.chiTrans_max);
            gpuparticles->particle_chiT[poffset] = val;
        }
        else if (__gpusetup_variables.chiTrans_distr == 5) // log-gaussian (truncated log-normal)
        {
            log_min = log(__gpusetup_variables.chiTrans_min);
            log_max = log(__gpusetup_variables.chiTrans_max);
            do {
                val = log(__gpusetup_variables.chiTrans_mean)
                    + __gpusetup_variables.chiTrans_sigma *
                    curand_normal(&(gpuparticles->particle_curandstate[poffset]));
            } while (val < log_min || val > log_max);
            gpuparticles->particle_chiT[poffset] = exp(val);
        }
        else printf("ERR INIT: chiTrans_distr=%d unavailable\n", __gpusetup_variables.chiTrans_distr);
        /** Particle rotational chemosensitivity **/
        if (__gpusetup_variables.chiRot_distr == 0) // single value
            gpuparticles->particle_chiR[poffset] = __gpusetup_variables.chiRot;
        else if (__gpusetup_variables.chiRot_distr == 1) // bimodal
        {
            gpuparticles->particle_chiR[poffset] = (curand_uniform(&(gpuparticles->particle_curandstate[poffset])) > __gpusetup_variables.chiRot_bias) ?
                __gpusetup_variables.chiRot_min : __gpusetup_variables.chiRot_max;
        }
        else if (__gpusetup_variables.chiRot_distr == 2) // uniform
        {
            gpuparticles->particle_chiR[poffset] = __gpusetup_variables.chiRot_min
                + curand_uniform(&(gpuparticles->particle_curandstate[poffset])) * (__gpusetup_variables.chiRot_max - __gpusetup_variables.chiRot_min);
        }
        else if (__gpusetup_variables.chiRot_distr == 3) // log-uniform
        {
            gpuparticles->particle_chiR[poffset] = exp10(
                log10(__gpusetup_variables.chiRot_min)
                + curand_uniform(&(gpuparticles->particle_curandstate[poffset])) * (log10(__gpusetup_variables.chiRot_max) - log10(__gpusetup_variables.chiRot_min)));
        }
        else if (__gpusetup_variables.chiRot_distr == 4) // gaussian (truncated)
        {
            do {
                val = __gpusetup_variables.chiRot_mean
                    + __gpusetup_variables.chiRot_sigma *
                    curand_normal(&(gpuparticles->particle_curandstate[poffset]));
            } while (val < __gpusetup_variables.chiRot_min || val > __gpusetup_variables.chiRot_max);
            gpuparticles->particle_chiR[poffset] = val;
        }
        else if (__gpusetup_variables.chiRot_distr == 5) // log-gaussian (truncated log-normal)
        {
            log_min = log(__gpusetup_variables.chiRot_min);
            log_max = log(__gpusetup_variables.chiRot_max);
            do {
                val = log(__gpusetup_variables.chiRot_mean)
                    + __gpusetup_variables.chiRot_sigma *
                    curand_normal(&(gpuparticles->particle_curandstate[poffset]));
            } while (val < log_min || val > log_max);
            gpuparticles->particle_chiR[poffset] = exp(val);
        }
        else printf("ERR INIT: chiRot_distr=%d unavailable\n", __gpusetup_variables.chiRot_distr);
        /** Particle chemosensitivity noise level **/
        if (__gpusetup_variables.SC0_distr == 0) // single value
            gpuparticles->particle_c0[poffset] = __gpusetup_variables.SC0;
        else if (__gpusetup_variables.SC0_distr == 1) // bimodal
        {
            gpuparticles->particle_c0[poffset] = (curand_uniform(&(gpuparticles->particle_curandstate[poffset])) > __gpusetup_variables.SC0_bias) ?
                __gpusetup_variables.SC0_min : __gpusetup_variables.SC0_max;
        }
        else if (__gpusetup_variables.SC0_distr == 2) // uniform
        {
            gpuparticles->particle_c0[poffset] = __gpusetup_variables.SC0_min
                + curand_uniform(&(gpuparticles->particle_curandstate[poffset])) * (__gpusetup_variables.SC0_max - __gpusetup_variables.SC0_min);
        }
        else if (__gpusetup_variables.SC0_distr == 3) // log-uniform
        {
            gpuparticles->particle_c0[poffset] = exp10(
                log10(__gpusetup_variables.SC0_min)
                + curand_uniform(&(gpuparticles->particle_curandstate[poffset])) * (log10(__gpusetup_variables.SC0_max) - log10(__gpusetup_variables.SC0_min)));
        }
        else if (__gpusetup_variables.SC0_distr == 4) // gaussian (truncated)
        {
            do {
                val = __gpusetup_variables.SC0_mean
                    + __gpusetup_variables.SC0_sigma *
                    curand_normal(&(gpuparticles->particle_curandstate[poffset]));
            } while (val < __gpusetup_variables.SC0_min || val > __gpusetup_variables.SC0_max);
            gpuparticles->particle_c0[poffset] = val;
        }
        else if (__gpusetup_variables.SC0_distr == 5) // log-gaussian (truncated log-normal)
        {
            log_min = log(__gpusetup_variables.SC0_min);
            log_max = log(__gpusetup_variables.SC0_max);
            do {
                val = log(__gpusetup_variables.SC0_mean)
                    + __gpusetup_variables.SC0_sigma *
                    curand_normal(&(gpuparticles->particle_curandstate[poffset]));
            } while (val < log_min || val > log_max);
            gpuparticles->particle_c0[poffset] = exp(val);
        }
        else printf("ERR INIT: SC0_distr=%d unavailable\n", __gpusetup_variables.SC0_distr);
        /** Particle rotational diffusion coefficient **/
        if (__gpusetup_variables.rotationalDiffusion_distr == 0) // single value
            gpuparticles->particle_DR[poffset] = __gpusetup_variables.rotationalDiffusion;
        else if (__gpusetup_variables.rotationalDiffusion_distr == 1) // bimodal
        {
            gpuparticles->particle_DR[poffset] = (curand_uniform(&(gpuparticles->particle_curandstate[poffset])) > __gpusetup_variables.rotationalDiffusion_bias) ?
                __gpusetup_variables.rotationalDiffusion_min : __gpusetup_variables.rotationalDiffusion_max;
        }
        else if (__gpusetup_variables.rotationalDiffusion_distr == 2) // uniform
        {
            gpuparticles->particle_DR[poffset] = __gpusetup_variables.rotationalDiffusion_min
                + curand_uniform(&(gpuparticles->particle_curandstate[poffset])) * (__gpusetup_variables.rotationalDiffusion_max - __gpusetup_variables.rotationalDiffusion_min);
        }
        else if (__gpusetup_variables.rotationalDiffusion_distr == 3) // log-uniform
        {
            gpuparticles->particle_DR[poffset] = exp10(
                log10(__gpusetup_variables.rotationalDiffusion_min)
                + curand_uniform(&(gpuparticles->particle_curandstate[poffset])) * (log10(__gpusetup_variables.rotationalDiffusion_max) - log10(__gpusetup_variables.rotationalDiffusion_min)));
        }
        else if (__gpusetup_variables.rotationalDiffusion_distr == 4) // gaussian (truncated)
        {
            do {
                val = __gpusetup_variables.rotationalDiffusion_mean
                    + __gpusetup_variables.rotationalDiffusion_sigma *
                    curand_normal(&(gpuparticles->particle_curandstate[poffset]));
            } while (val < __gpusetup_variables.rotationalDiffusion_min || val > __gpusetup_variables.rotationalDiffusion_max);
            gpuparticles->particle_DR[poffset] = val;
        }
        else if (__gpusetup_variables.rotationalDiffusion_distr == 5) // log-gaussian (truncated log-normal)
        {
            log_min = log(__gpusetup_variables.rotationalDiffusion_min);
            log_max = log(__gpusetup_variables.rotationalDiffusion_max);
            do {
                val = log(__gpusetup_variables.rotationalDiffusion_mean)
                    + __gpusetup_variables.rotationalDiffusion_sigma *
                    curand_normal(&(gpuparticles->particle_curandstate[poffset]));
            } while (val < log_min || val > log_max);
            gpuparticles->particle_DR[poffset] = exp(val);
        }
        else printf("ERR INIT: rotationalDiffusion_distr=%d unavailable\n", __gpusetup_variables.rotationalDiffusion_distr);
        /** Particle next reset time **/
        if (__gpusetup_variables.initialNTRtype == 0) 
        { // 0 - same as regular
            if (__gpusetup_variables.timedResetTimerType == 0)
            {
                gpuparticles->particle_NRT[poffset] = __gpusetup_variables.timedResetMeanTime;
                if (__gpusetup_variables.globalResetTime >= 0.00 && gpuparticles->particle_NRT[poffset] >= __gpusetup_variables.globalResetTime)
                    gpuparticles->particle_NRT[poffset] = __gpusetup_variables.globalResetTime;
            }
            else if (__gpusetup_variables.timedResetTimerType == 1)
            {
                flt2 z3 = curand_uniform(&(gpuparticles->particle_curandstate[poffset]));
                gpuparticles->particle_NRT[poffset] = - log(z3) * __gpusetup_variables.timedResetMeanTime;
                if (__gpusetup_variables.globalResetTime >= 0.00 && gpuparticles->particle_NRT[poffset] >= __gpusetup_variables.globalResetTime)
                    gpuparticles->particle_NRT[poffset] = __gpusetup_variables.globalResetTime;
            }
        }
        else if (__gpusetup_variables.initialNTRtype == 1)
        { // 1 - random at [0.00, timedResetMeanTime)
            gpuparticles->particle_NRT[poffset] = curand_uniform(&(gpuparticles->particle_curandstate[poffset])) * __gpusetup_variables.timedResetMeanTime;
            if (__gpusetup_variables.globalResetTime >= 0.00 && gpuparticles->particle_NRT[poffset] >= __gpusetup_variables.globalResetTime)
                gpuparticles->particle_NRT[poffset] = __gpusetup_variables.globalResetTime;
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
        gpuparticles->particle_flag_wasResetToCenter[poffset] = false;

    }
}


#endif // VC3_PHYS_KSNP_KERNELS_INIT
