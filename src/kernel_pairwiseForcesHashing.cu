#ifndef VC3_PHYS_KSNP_KERNELS_PPSPATIALHASHING
#define VC3_PHYS_KSNP_KERNELS_PPSPATIALHASHING


#include "../include/types.cu"
#include "../include/cumath/cumath_consts.cu"
#include "../include/cumath/cubasicmath.cu"
#include "../include/cumath/cuvector2D.cu"
#include "../include/cumath/cusizevector2D.cu"

#include "gpudata.cu"


/** Device kernel
    Builds the spatial hash grid by placing each particle into a linked list
    within its corresponding hash table slot.
    To be ran with a 1D grid, one thread per particle.
**/
__global__ void __kernel_PPbuildHashLists(
    vc3_phys::gpu_particles* __restrict__ gpuparticles, 
    vc3_phys::gpu_ppinteractions* __restrict__ gpuppinteractions)
{
    const int poffset = blockIdx.x * blockDim.x + threadIdx.x;
    if (poffset >= __gpusetup_variables.nParticles) return;

    // Calculate cell coordinates as SIGNED integers.
    const int cell_x = floorf(gpuparticles->particle_pos[poffset].x * __gpuprecomputes.PPinv_cell_size);
    const int cell_y = floorf(gpuparticles->particle_pos[poffset].y * __gpuprecomputes.PPinv_cell_size);
    gpuparticles->particle_cell_coord[poffset].x = cell_x;
    gpuparticles->particle_cell_coord[poffset].y = cell_y;

    // Use a proper integer-pair hash function that does not form a large intermediate key.
    // Large prime numbers scramble the bits of the signed coordinates.
    unsigned int hash = ((unsigned int)cell_x * 73856093u) ^ ((unsigned int)cell_y * 19349663u);
    hash &= (__gpuprecomputes.PPhash_table_size - 1);
    int previous_head = atomicExch(&(gpuppinteractions->grid_cell_heads[hash]), poffset);
    gpuparticles->particle_next_in_cell[poffset] = previous_head;
}

/** Device kernel
    Calculates pairwise DPD forces using a hash grid and a "full neighbor list"
    approach to avoid atomics.
    To be ran with a 1D grid, one thread per particle.
**/
__global__ void __kernel_PPforcesHashGrid(
    vc3_phys::gpu_particles* __restrict__ gpuparticles,
    vc3_phys::gpu_ppinteractions* __restrict__ gpuppinteractions)
{
    const int poffset = blockIdx.x * blockDim.x + threadIdx.x;
    if (poffset >= __gpusetup_variables.nParticles) return;

    vc3_cumath::planar::cuvector force(0.0, 0.0);
    const vc3_cumath::planar::cuvector p_pos = gpuparticles->particle_pos[poffset];
    const int p_cell_x = gpuparticles->particle_cell_coord[poffset].x;
    const int p_cell_y = gpuparticles->particle_cell_coord[poffset].y;

    // Loop through the 9 neighboring cells
    for (int dy = -1; dy <= 1; ++dy) 
    {
        for (int dx = -1; dx <= 1; ++dx) 
        {
            const int neighbor_cell_x = p_cell_x + dx;
            const int neighbor_cell_y = p_cell_y + dy;

            // For a non-periodic system, we simply skip cells that are outside the valid grid.
            if (neighbor_cell_x < 0 || neighbor_cell_x >= __gpuprecomputes.PPgrid_width ||
                neighbor_cell_y < 0 || neighbor_cell_y >= __gpuprecomputes.PPgrid_width)
                continue;

            unsigned int hash = ((unsigned int)neighbor_cell_x * 73856093u) ^ ((unsigned int)neighbor_cell_y * 19349663u);
            hash &= (__gpuprecomputes.PPhash_table_size - 1);

            int j = gpuppinteractions->grid_cell_heads[hash];
            while (j != -1)
            {
                // HASH COLLISION RESOLUTION:
                // Check if particle 'j' is actually in the correct neighbor cell.
                if (gpuparticles->particle_cell_coord[j].x == neighbor_cell_x &&
                    gpuparticles->particle_cell_coord[j].y == neighbor_cell_y)
                {
                    if (poffset != j) 
                    {
                        const vc3_cumath::planar::cuvector dr = p_pos - gpuparticles->particle_pos[j];
                        const flt2 dr2 = dr.length2();

                        if (dr2 < __gpuprecomputes.PPcutoff2 && dr2 > 1e-20) 
                        {
                            // Use the fast reciprocal square root intrinsic
                            const flt2 inv_r = rsqrtf(dr2);
                            const flt2 r = dr2 * inv_r;
                            const flt2 F = __gpusetup_variables.PPepsilon * (1.00 - r / __gpusetup_variables.PPsigma);
                            force += (F * inv_r) * dr;
                        }
                    }
                }
                // Move to the next particle in the list
                j = gpuparticles->particle_next_in_cell[j];
            }
        }
    }

    gpuparticles->particle_PPforceSpatialHashing[poffset] = force;
}


#endif // VC3_PHYS_KSNP_KERNELS_PPSPATIALHASHING
