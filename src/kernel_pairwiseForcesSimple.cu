#ifndef VC3_PHYS_KSNP_KERNELS_PPSIMPLE
#define VC3_PHYS_KSNP_KERNELS_PPSIMPLE


#include "../include/types.cu"
#include "../include/cumath/cumath_consts.cu"
#include "../include/cumath/cubasicmath.cu"
#include "../include/cumath/cuvector2D.cu"
#include "../include/cumath/cusizevector2D.cu"

#include "gpudata.cu"



/** Device kernel
    Calculates pairwise particle-particle forces in a straightforward O(N^2) manner.

    Each thread is assigned to a single particle 'p' and iterates through all other
    particles 'j' in the system to compute the net force. This method is
    computationally expensive but serves as a correct ground truth for validation.
    It does not require neighbor lists or spatial hashing.

    To be ran with a 1D grid of Nblocks=((nParticles + threadsPerBlock - 1) / threadsPerBlock)
    and Nthreads=(threadsPerBlock), where threadsPerBlock is typically 256 or 512.
**/
__global__ void __kernel_PPforcesN2(vc3_phys::gpu_variables* gpuvariables,
    vc3_phys::gpu_particles* gpuparticles)
{
    int poffset = blockIdx.x * blockDim.x + threadIdx.x;
    if (poffset >= __gpusetup_variables.nParticles) return;
    
    vc3_cumath::planar::cuvector force(0.00, 0.00);
    const vc3_cumath::planar::cuvector p_pos = gpuparticles->particle_pos[poffset];

    // Loop through all particles
    for (int q = 0; q < __gpusetup_variables.nParticles; q++) 
    {
        // A particle does not interact with itself
        if (poffset == q) continue;

        const vc3_cumath::planar::cuvector dr = p_pos - gpuparticles->particle_pos[q];
        const flt2 dr2 = dr.length2();
        if (dr2 < __gpuprecomputes.PPcutoff2 && dr2 > 1e-20)
        {
            const flt2 r = vc3_cumath::msqrt_flt2(dr2);
            force += __gpusetup_variables.PPepsilon * (1.00 - r / __gpusetup_variables.PPsigma) / r * dr;
        }
    }

    // After checking all other particles, write the final net force to global memory
    gpuparticles->particle_PPforce[poffset] = force;
}

__global__ void __kernel_PPforcesN2_NOATOMIC(vc3_phys::gpu_variables* gpuvariables,
    vc3_phys::gpu_particles* gpuparticles)
{
    int poffset = blockIdx.x * blockDim.x + threadIdx.x;
    if (poffset >= __gpusetup_variables.nParticles) return;

    vc3_cumath::planar::cuvector force(0.00, 0.00);
    const vc3_cumath::planar::cuvector p_pos = gpuparticles->particle_pos[poffset];

    for (int q = 0; q < __gpusetup_variables.nParticles; q++)
    {
        if (poffset == q) continue;

        const vc3_cumath::planar::cuvector dr = p_pos - gpuparticles->particle_pos[q];
        const flt2 dr2 = dr.length2();

        if (dr2 < __gpuprecomputes.PPcutoff2 && dr2 > 1e-20)
        {
            // Exact same instructions as the original spatial hashing kernel
            const flt2 inv_r = rsqrtf(dr2);
            const flt2 r = dr2 * inv_r;
            const flt2 F = __gpusetup_variables.PPepsilon * (1.00 - r / __gpusetup_variables.PPsigma);
            force += (F * inv_r) * dr;
        }
    }

    // Write to the hash array so the move kernel picks it up unchanged
    gpuparticles->particle_PPforceSpatialHashing[poffset] = force;
}

__global__ void __kernel_PPforcesN2_PBC_NOATOMIC(vc3_phys::gpu_variables* gpuvariables,
    vc3_phys::gpu_particles* gpuparticles)
{
    int poffset = blockIdx.x * blockDim.x + threadIdx.x;
    if (poffset >= __gpusetup_variables.nParticles) return;

    vc3_cumath::planar::cuvector force(0.00, 0.00);
    const vc3_cumath::planar::cuvector p_pos = gpuparticles->particle_pos[poffset];

    for (int q = 0; q < __gpusetup_variables.nParticles; q++)
    {
        if (poffset == q) continue;

        vc3_cumath::planar::cuvector dr = p_pos - gpuparticles->particle_pos[q];

        // Exact same PBC Minimum Image Convention wrapping logic
        if (dr.x > __gpuprecomputes.halfBox) dr.x -= __gpusetup_variables.boxSize;
        else if (dr.x < -__gpuprecomputes.halfBox) dr.x += __gpusetup_variables.boxSize;
        if (dr.y > __gpuprecomputes.halfBox) dr.y -= __gpusetup_variables.boxSize;
        else if (dr.y < -__gpuprecomputes.halfBox) dr.y += __gpusetup_variables.boxSize;

        const flt2 dr2 = dr.length2();
        if (dr2 < __gpuprecomputes.PPcutoff2 && dr2 > 1e-20)
        {
            const flt2 inv_r = rsqrtf(dr2);
            const flt2 r = dr2 * inv_r;
            const flt2 F = __gpusetup_variables.PPepsilon * (1.00 - r / __gpusetup_variables.PPsigma);
            force += (F * inv_r) * dr;
        }
    }

    gpuparticles->particle_PPforceSpatialHashing[poffset] = force;
}


#endif // VC3_PHYS_KSNP_KERNELS_PPSIMPLE
