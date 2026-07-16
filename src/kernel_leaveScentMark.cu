#ifndef VC3_PHYS_KSNP_KERNELS_PUTSC
#define VC3_PHYS_KSNP_KERNELS_PUTSC

#include <stdio.h>

#include "../include/types.cu"
#include "../include/cumath/cubasicmath.cu"
#include "../include/cumath/cuvector2D.cu"
#include "../include/cumath/cusizevector2D.cu"

#include "gpudata.cu"


__global__ void __kernel_leaveScentMarks(const vc3_phys::gpu_variables* gpuvariables, 
    vc3_phys::gpu_matrixes* __restrict__ gpumatrixes, const vc3_phys::gpu_particles* gpuparticles)
{
    /**
        LIMITED TO 65535 PARTICLES
    **/

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= __gpuprecomputes.SCmark_size) return;
    //printf("A\t%d\t%d\t%di=\t%d\tN=\t%d\tdl=%f\n", blockIdx.x, blockDim.x, threadIdx.x, i, N, dl);

    // Map linear i -> (lx, ly) such that ly (columns) varies fastest for coalescing
    int ly = i % __gpuprecomputes.SCmark_window;                  // delta in y (columns)
    int lx = i / __gpuprecomputes.SCmark_window;                  // delta in x (rows)
    int nx = gpuparticles->particle_posLattice[blockIdx.y].x - __gpuprecomputes.SCmark_dl_cutoff + lx;
    int ny = gpuparticles->particle_posLattice[blockIdx.y].y - __gpuprecomputes.SCmark_dl_cutoff + ly;

    //printf("B\t%d\t%d\t%d\ti=\t%d\tlx=\t%d\tly=\t%d\tnx=\t%d\tny=\t%d\n", blockIdx.x, blockDim.x, threadIdx.x, i, lx, ly, nx, ny);

    // Skip out-of-bounds (when window crosses edges)
    if (nx < 0 || nx >= __gpusetup_variables.latticeSize) return;
    if (ny < 0 || ny >= __gpusetup_variables.latticeSize) return;

    // Compute contribution
    // mypos = dl * ((nx+0.5), (ny+0.5))
    flt2 dx = gpuparticles->particle_pos[blockIdx.y].x - __gpuprecomputes.dl * flt2(nx);
    flt2 dy = gpuparticles->particle_pos[blockIdx.y].y - __gpuprecomputes.dl * flt2(ny);
    flt2 dr2 = dx * dx + dy * dy;
    if (dr2 <= __gpuprecomputes.SCmark_cutoff2)
    {
        // Add scent
        //flt2 nd = vc3_cumath::normdistcopt(dr2, __gpuprecomputes.Rscentinv, __gpuprecomputes.Rscent2inv);
        flt2 nd = vc3_cumath::normdistcoptf(dr2, __gpuprecomputes.Rscentinv, __gpuprecomputes.Rscent2inv);
        flt2 dSC = gpuparticles->particle_beta[blockIdx.y] * nd * gpuvariables->scentDivideFactor;
        long long int scoffset = vc3_phys::get_SCoffset(nx, ny, __gpusetup_variables.latticeSize);
        atomicAdd(&(gpumatrixes->SC[scoffset]), dSC);
    }
}

__global__ void __kernel_leaveScentMarks_PBC(const vc3_phys::gpu_variables* gpuvariables,
    vc3_phys::gpu_matrixes* __restrict__ gpumatrixes, const vc3_phys::gpu_particles* gpuparticles)
{
    /**
        LIMITED TO 65535 PARTICLES
        Modified for Periodic Boundary Conditions (PBC)
    **/

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= __gpuprecomputes.SCmark_size) return;
    //printf("A\t%d\t%d\t%di=\t%d\tN=\t%d\tdl=%f\n", blockIdx.x, blockDim.x, threadIdx.x, i, N, dl);

    // Map linear i -> (lx, ly)
    int ly = i % __gpuprecomputes.SCmark_window;                  // delta in y (columns)
    int lx = i / __gpuprecomputes.SCmark_window;                  // delta in x (rows)
    int nx = gpuparticles->particle_posLattice[blockIdx.y].x - __gpuprecomputes.SCmark_dl_cutoff + lx;
    int ny = gpuparticles->particle_posLattice[blockIdx.y].y - __gpuprecomputes.SCmark_dl_cutoff + ly;

    //printf("B\t%d\t%d\t%d\ti=\t%d\tlx=\t%d\tly=\t%d\tnx=\t%d\tny=\t%d\n", blockIdx.x, blockDim.x, threadIdx.x, i, lx, ly, nx, ny);

    // Skip out-of-bounds (when window crosses edges)
    // - PBC => no skip, wrap coordinates instead

    // Compute contribution
    // mypos = dl * ((nx+0.5), (ny+0.5))
    flt2 dx = gpuparticles->particle_pos[blockIdx.y].x - __gpuprecomputes.dl * flt2(nx);
    flt2 dy = gpuparticles->particle_pos[blockIdx.y].y - __gpuprecomputes.dl * flt2(ny);
    flt2 dr2 = dx * dx + dy * dy;
    if (dr2 <= __gpuprecomputes.SCmark_cutoff2)
    {
        // Wrap coordinates
        if (nx < 0) nx += __gpusetup_variables.latticeSize;
        else if (nx >= __gpusetup_variables.latticeSize) nx -= __gpusetup_variables.latticeSize;
        if (ny < 0) ny += __gpusetup_variables.latticeSize;
        else if (ny >= __gpusetup_variables.latticeSize) ny -= __gpusetup_variables.latticeSize;

        // Add scent
        //flt2 nd = vc3_cumath::normdistcopt(dr2, __gpuprecomputes.Rscentinv, __gpuprecomputes.Rscent2inv);
        flt2 nd = vc3_cumath::normdistcoptf(dr2, __gpuprecomputes.Rscentinv, __gpuprecomputes.Rscent2inv);
        flt2 dSC = gpuparticles->particle_beta[blockIdx.y] * nd * gpuvariables->scentDivideFactor;
        long long int scoffset = vc3_phys::get_SCoffset(nx, ny, __gpusetup_variables.latticeSize);
        atomicAdd(&(gpumatrixes->SC[scoffset]), dSC);
    }
}

__global__ void __kernel_leaveScentMarks_NOATOMIC(const vc3_phys::gpu_variables* gpuvariables,
    vc3_phys::gpu_matrixes* __restrict__ gpumatrixes, const vc3_phys::gpu_particles* gpuparticles)
{
    // DETERMINISTIC GATHER APPROACH (Open Boundaries)
    int nx = blockIdx.x * blockDim.x + threadIdx.x;
    int ny = blockIdx.y * blockDim.y + threadIdx.y;

    if (nx >= __gpusetup_variables.latticeSize || ny >= __gpusetup_variables.latticeSize) return;

    flt2 cell_dSC = 0.00;
    int cutoff = __gpuprecomputes.SCmark_dl_cutoff;

    // Strictly ordered floating-point summation (0 to N-1)
    for (int q = 0; q < __gpusetup_variables.nParticles; q++)
    {
        int center_x = gpuparticles->particle_posLattice[q].x;
        int center_y = gpuparticles->particle_posLattice[q].y;

        // Mimics the exact original iteration bounds
        if (nx >= center_x - cutoff && nx <= center_x + cutoff &&
            ny >= center_y - cutoff && ny <= center_y + cutoff)
        {
            // Exact same unwrapped distance calculation
            flt2 dist_x = gpuparticles->particle_pos[q].x - __gpuprecomputes.dl * flt2(nx);
            flt2 dist_y = gpuparticles->particle_pos[q].y - __gpuprecomputes.dl * flt2(ny);
            flt2 dr2 = dist_x * dist_x + dist_y * dist_y;

            if (dr2 <= __gpuprecomputes.SCmark_cutoff2) {
                flt2 nd = vc3_cumath::normdistcoptf(dr2, __gpuprecomputes.Rscentinv, __gpuprecomputes.Rscent2inv);
                cell_dSC += gpuparticles->particle_beta[q] * nd;
            }
        }
    }

    if (cell_dSC > 0.00)
    {
        long long int scoffset = vc3_phys::get_SCoffset(nx, ny, __gpusetup_variables.latticeSize);
        gpumatrixes->SC[scoffset] += cell_dSC * gpuvariables->scentDivideFactor;
    }
}

__global__ void __kernel_leaveScentMarks_PBC_NOATOMIC(const vc3_phys::gpu_variables* gpuvariables,
    vc3_phys::gpu_matrixes* __restrict__ gpumatrixes, const vc3_phys::gpu_particles* gpuparticles)
{
    // DETERMINISTIC GATHER APPROACH (Periodic Boundaries)
    int nx = blockIdx.x * blockDim.x + threadIdx.x;
    int ny = blockIdx.y * blockDim.y + threadIdx.y;

    if (nx >= __gpusetup_variables.latticeSize || ny >= __gpusetup_variables.latticeSize) return;

    flt2 cell_dSC = 0.00;
    int cutoff = __gpuprecomputes.SCmark_dl_cutoff;

    for (int q = 0; q < __gpusetup_variables.nParticles; q++)
    {
        int center_x = gpuparticles->particle_posLattice[q].x;
        int center_y = gpuparticles->particle_posLattice[q].y;

        // Iterate through the exact same theoretical scatter box
        for (int dy = -cutoff; dy <= cutoff; dy++) {
            int unwrapped_y = center_y + dy;
            int wrapped_y = unwrapped_y % __gpusetup_variables.latticeSize;
            if (wrapped_y < 0) wrapped_y += __gpusetup_variables.latticeSize;

            if (wrapped_y == ny) {
                for (int dx = -cutoff; dx <= cutoff; dx++) {
                    int unwrapped_x = center_x + dx;
                    int wrapped_x = unwrapped_x % __gpusetup_variables.latticeSize;
                    if (wrapped_x < 0) wrapped_x += __gpusetup_variables.latticeSize;

                    if (wrapped_x == nx) {
                        // Crucially, evaluate geometric distance using UNWRAPPED coordinates
                        // to perfectly match the legacy scatter math.
                        flt2 dist_x = gpuparticles->particle_pos[q].x - __gpuprecomputes.dl * flt2(unwrapped_x);
                        flt2 dist_y = gpuparticles->particle_pos[q].y - __gpuprecomputes.dl * flt2(unwrapped_y);
                        flt2 dr2 = dist_x * dist_x + dist_y * dist_y;

                        if (dr2 <= __gpuprecomputes.SCmark_cutoff2) {
                            flt2 nd = vc3_cumath::normdistcoptf(dr2, __gpuprecomputes.Rscentinv, __gpuprecomputes.Rscent2inv);
                            cell_dSC += gpuparticles->particle_beta[q] * nd;
                        }
                    }
                }
            }
        }
    }

    if (cell_dSC > 0.00)
    {
        long long int scoffset = vc3_phys::get_SCoffset(nx, ny, __gpusetup_variables.latticeSize);
        gpumatrixes->SC[scoffset] += cell_dSC * gpuvariables->scentDivideFactor;
    }
}


#endif // VC3_PHYS_KSNP_KERNELS_PUTSC
