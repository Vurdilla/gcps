#ifndef VC3_PHYS_KSNP_KERNELS_RENORMSC
#define VC3_PHYS_KSNP_KERNELS_RENORMSC


#include "../include/types.cu"
#include "../include/cumath/cumath_consts.cu"

#include "gpudata.cu"



/** Device kernel
    Renormalizes SC matrix via dividing by scentDivideFactor

    To be ran with Nblocks=((latticeSize2 + threadsPerBlock - 1) / threadsPerBlock) and Nthreads=(threadsPerBlock)
    threadsPerBlock = 256 or 512 or 1024, depending on architechture

    Obsolete
    Works for latticeSize < 46000 (32 bit -> 16GB limit)
**/
__global__ void __kernel_renormSC(const vc3_phys::gpu_variables* gpuvariables, vc3_phys::gpu_matrixes* __restrict__ gpumatrixes)
{
    long long int scoffset = blockIdx.x * blockDim.x + threadIdx.x;
    flt2 scentDivideFactorInv = 1.00 / gpuvariables->scentDivideFactor;
    if (scoffset < __gpuprecomputes.matrixSize)
        gpumatrixes->SC[scoffset] *= scentDivideFactorInv;
}


/** Device kernel
    Renormalizes SC matrix via dividing by scentDivideFactor

    To be ran with Nblocks=... and Nthreads=...
**/
__global__ void __kernel_renormSC_2D(const vc3_phys::gpu_variables* gpuvariables, vc3_phys::gpu_matrixes* __restrict__ gpumatrixes)
{
    // Calculate the unique 2D global indices for this thread
    int nx = blockIdx.x * blockDim.x + threadIdx.x;
    int ny = blockIdx.y * blockDim.y + threadIdx.y;

    // Boundary check to ensure we don't write past the matrix edges
    if (nx < __gpusetup_variables.latticeSize && ny < __gpusetup_variables.latticeSize)
    {
        long long int scoffset = vc3_phys::get_SCoffset(nx, ny, __gpusetup_variables.latticeSize);
        flt2 scentDivideFactorInv = 1.00 / gpuvariables->scentDivideFactor;
        gpumatrixes->SC[scoffset] *= scentDivideFactorInv;
    }
}


#endif // VC3_PHYS_KSNP_KERNELS_RENORMSC
