#ifndef __TYPES_H__
#define __TYPES_H__

typedef double vector3[3];
#define FILL_VECTOR(vector,a,b,c) {vector[0]=a;vector[1]=b;vector[2]=c;}

// Host Memory

extern vector3* host_velocities;
extern vector3* host_positions;
extern double* host_masses;

// Device Memory

extern vector3* device_velocities;
extern vector3* device_positions;
extern double* device_masses;

extern vector3** accels;

// Kernel Config Arguments

extern dim3 calc_changes_grid_dims;
extern dim3 calc_changes_block_dims;

extern dim3 calc_accels_grid_dims;
extern dim3 calc_accels_block_dims;

extern dim3 sum_accels_grid_dims;
extern dim3 sum_accels_block_dims;

void setSumAccelsDims (int entity_count, dim3* grid_dims, dim3* block_dims);

#endif