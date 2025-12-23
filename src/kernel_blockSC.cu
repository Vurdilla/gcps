#ifndef VC3_PHYS_KSNP_KERNELS_BLOCKSC
#define VC3_PHYS_KSNP_KERNELS_BLOCKSC


#include "../include/types.cu"
#include "../include/cumath/cumath_consts.cu"

#include "gpudata.cu"



/** Device kernel
    Coarsens the large SC matrix into the smaller SCblocked matrix by summing blocks.

    Each thread in the grid is responsible for calculating the sum of one block. All
    necessary size parameters are read from the __gpusetup_variables constant memory.

    To be ran with a 2D grid of 2D blocks (e.g., 16x16 threads per block).
    The grid dimensions should be sufficient to cover the entire SCblocked matrix.
**/
__global__ void __kernel_blockSC(vc3_phys::gpu_matrixes* __restrict__ gpumatrixes)
{
    // 1. Identify the target cell this thread is responsible for in the DESTINATION matrix.
    const int dest_nx = blockIdx.x * blockDim.x + threadIdx.x;
    const int dest_ny = blockIdx.y * blockDim.y + threadIdx.y;

    // Boundary check for destination matrix
    if (dest_nx >= __gpuprecomputes.SCblockLatticeSize || dest_ny >= __gpuprecomputes.SCblockLatticeSize) {
        return;
    }
    long long int dest_offset = (long long int)dest_ny * __gpuprecomputes.SCblockLatticeSize + dest_nx;

    // 2. Determine the top-left corner of the corresponding block in the SOURCE matrix.
    const int source_start_nx = dest_nx * __gpusetup_variables.SCblockSize;
    const int source_start_ny = dest_ny * __gpusetup_variables.SCblockSize;

    // 3. Loop over the block in the source matrix and accumulate the sum.
    flt2 block_sum = 0.00;
    for (int j = 0; j < __gpusetup_variables.SCblockSize; ++j) 
    {
        const int current_source_ny = source_start_ny + j;
        for (int i = 0; i < __gpusetup_variables.SCblockSize; ++i) 
        {
            const int current_source_nx = source_start_nx + i;
            // Boundary check for source matrix
            if (current_source_nx < __gpusetup_variables.latticeSize &&
                current_source_ny < __gpusetup_variables.latticeSize)
            {
                long long int source_offset = vc3_phys::get_SCoffset(current_source_nx, current_source_ny, __gpusetup_variables.latticeSize);
                block_sum += gpumatrixes->SC[source_offset];
            }
        }
    }

    gpumatrixes->SCblocked[dest_offset] = block_sum;
}


#endif // VC3_PHYS_KSNP_KERNELS_BLOCKSC
