#ifndef __KERNEL_UTILS_CUH__
#define __KERNEL_UTILS_CUH__

void printKernelDims (const char* identifier, dim3 grid_dims, dim3 block_dims);
void initCalcAccelsDims ();
void initCalcChangesDims ();
void setSumAccelsDims (int entity_count);

extern dim3 calc_changes_grid_dims;
extern dim3 calc_changes_block_dims;

extern dim3 calc_accels_grid_dims;
extern dim3 calc_accels_block_dims;

extern dim3 sum_accels_grid_dims;
extern dim3 sum_accels_block_dims;

#endif