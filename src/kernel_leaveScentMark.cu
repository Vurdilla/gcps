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




#endif // VC3_PHYS_KSNP_KERNELS_PUTSC
