#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <cooperative_groups.h>
#include "vector.h"
#include "config.h"

__global__ void calcAccels (vector3** accels, vector3* positions, double* masses) {

	int local_row = threadIdx.y;
	int local_col = threadIdx.x;

	int global_row = (blockIdx.y * blockDim.y) + local_row;
	int global_col = (blockIdx.x * blockDim.x) + local_col;
	int spatial_axis = threadIdx.z;

	if (NUMENTITIES <= global_col || NUMENTITIES <= global_row) return;

	if (global_row == global_col) {

		accels[global_row][global_col][spatial_axis] = 0;

	} else {

		__shared__ vector3 distances[BLOCK_WIDTH][BLOCK_WIDTH];

		distances[local_row][local_col][spatial_axis] = positions[global_row][spatial_axis] - positions[global_col][spatial_axis];

		__syncthreads();

		double magnitude_sq = distances[local_row][local_col][0] * distances[local_row][local_col][0] + distances[local_row][local_col][1] * distances[local_row][local_col][1] + distances[local_row][local_col][2] * distances[local_row][local_col][2];
		double magnitude = sqrt(magnitude_sq);
		double accelmag = -1 * GRAV_CONSTANT * masses[global_col] / magnitude_sq;
		accels[global_row][global_col][spatial_axis] = accelmag * distances[local_row][local_col][spatial_axis] / magnitude;

	}
}

__global__ void sumAccels (vector3** accels) {

	int global_row = blockIdx.y;
	int spatial_axis = threadIdx.z;
	
	double sum = 0;

	for (int i = 0; i < NUMENTITIES; i++) {
		sum += accels[global_row][i][spatial_axis];
	}

	accels[global_row][0][spatial_axis] = sum;
}

__global__ void calcChanges (vector3** accels, vector3* velocities, vector3* positions) {

	int global_row = blockIdx.y;
	int spatial_axis = threadIdx.z;

	velocities[global_row][spatial_axis] += accels[global_row][0][spatial_axis] * INTERVAL;
	positions[global_row][spatial_axis] += velocities[global_row][spatial_axis] * INTERVAL; 
}

void compute () {

	#ifdef DEBUG
	cudaError_t e = cudaGetLastError();
	if (e != cudaSuccess)
		printf("Error before compute! %s: %s\n",
			cudaGetErrorName(e),
			cudaGetErrorString(e)
		);
	#endif

	int grid_width = NUMENTITIES % BLOCK_WIDTH == 0 ? NUMENTITIES / BLOCK_WIDTH : (NUMENTITIES / BLOCK_WIDTH) + 1;
	dim3 calc_grid_dims (grid_width, grid_width, 1);
	dim3 calc_block_dims (BLOCK_WIDTH, BLOCK_WIDTH, SPATIAL_DIMS);

	// Calculate Accelerations
	calcAccels<<<calc_grid_dims, calc_block_dims>>>(accels, device_positions, device_masses);
	#ifdef DEBUG
	cudaError_t calc_accels_error = cudaGetLastError();
	if (calc_accels_error != cudaSuccess) 
		printf("calcAccels kernel launch failed! %s: %s\n",
			cudaGetErrorName(calc_accels_error),
			cudaGetErrorString(calc_accels_error)
		);
	#endif
	cudaDeviceSynchronize();

	dim3 sum_grid_dims (1, NUMENTITIES, 1);
	dim3 sum_block_dims (1, 1, SPATIAL_DIMS);

	// Sum Accelerations
	sumAccels<<<sum_grid_dims, sum_block_dims>>>(accels);
	#ifdef DEBUG
	cudaError_t sum_accels_error = cudaGetLastError();
	if (sum_accels_error != cudaSuccess) 
		printf("sumAccels kernel launch failed! %s: %s\n",
			cudaGetErrorName(sum_accels_error),
			cudaGetErrorString(sum_accels_error)
		);
	#endif
	cudaDeviceSynchronize();

	// Calculating Changes
	calcChanges<<<sum_grid_dims, sum_block_dims>>>(accels, device_velocities, device_positions);
	#ifdef DEBUG
	cudaError_t calc_changes_error = cudaGetLastError();
	if (calc_changes_error != cudaSuccess) 
		printf("calcChanges kernel launch failed! %s: %s\n",
			cudaGetErrorName(calc_changes_error),
			cudaGetErrorString(calc_changes_error)
		);
	#endif
	cudaDeviceSynchronize();
}
