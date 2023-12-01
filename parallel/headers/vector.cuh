#ifndef __TYPES_CUH__
#define __TYPES_CUH__

typedef double vector3[3];
#define FILL_VECTOR(vector,a,b,c) {vector[0]=a;vector[1]=b;vector[2]=c;}

// Kernel Config Arguments

extern dim3 calc_changes_grid_dims;
extern dim3 calc_changes_block_dims;

extern dim3 calc_accels_grid_dims;
extern dim3 calc_accels_block_dims;

extern dim3 sum_accels_grid_dims;
extern dim3 sum_accels_block_dims;

void setSumAccelsDims (int entity_count, dim3* grid_dims, dim3* block_dims);

#endif