#include <stdlib.h>
#include <stdio.h>
#include <math.h>
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

		__shared__ vector3 distances[CALC_ACCELS_BLOCK_WIDTH][CALC_ACCELS_BLOCK_WIDTH];

		distances[local_row][local_col][spatial_axis] = positions[global_row][spatial_axis] - positions[global_col][spatial_axis];

		__syncthreads();

		double magnitude_sq = distances[local_row][local_col][0] * distances[local_row][local_col][0] + distances[local_row][local_col][1] * distances[local_row][local_col][1] + distances[local_row][local_col][2] * distances[local_row][local_col][2];
		double magnitude = sqrt(magnitude_sq);
		double accelmag = -1 * GRAV_CONSTANT * masses[global_col] / magnitude_sq;
		accels[global_row][global_col][spatial_axis] = accelmag * distances[local_row][local_col][spatial_axis] / magnitude;

	}
}

__global__ void sumAccels (vector3** accels, int global_sum_length) {

	/**
	 * I used this resource to help me optimize my code.
	 * https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf
	 */

	int local_col = threadIdx.x;

	int global_row = blockIdx.y;
	int global_col = (blockIdx.x * blockDim.x) + local_col;
	int spatial_axis = blockIdx.z;

	extern __shared__ double sums[];
	int sums_length = blockDim.x;

	sums[local_col] = 0;

	if (global_sum_length <= global_col) return;

	sums[local_col] = accels[global_row][global_col][spatial_axis];

	__syncthreads();

	for (int stride = sums_length / 2; 0 < stride; stride >>= 1) {
		if (local_col < stride) sums[local_col] += sums[local_col + stride];
		__syncthreads();
	}

	if (local_col == 0) accels[global_row][blockIdx.x][spatial_axis] = sums[local_col];
}

__global__ void calcChanges (vector3** accels, vector3* velocities, vector3* positions) {

	int local_row = threadIdx.y;
	int global_row = (blockIdx.y * blockDim.y) + local_row;
	int spatial_axis = blockIdx.z;

	if (NUMENTITIES <= global_row) return;

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
	fflush(stdout);
	#endif

	// Calculate Accelerations

	calcAccels<<<calc_accels_grid_dims, calc_accels_block_dims>>>(accels, device_positions, device_masses);

	#ifdef DEBUG
	cudaError_t calc_accels_error = cudaGetLastError();
	if (calc_accels_error != cudaSuccess) {
		printf("calcAccels kernel launch failed! %s: %s\n",
			cudaGetErrorName(calc_accels_error),
			cudaGetErrorString(calc_accels_error)
		);
		printf("\tcalcAccels Config: gridDims: {%d %d %d}, blockDims: {%d %d %d}\n",
			calc_accels_grid_dims.x,
			calc_accels_grid_dims.y,
			calc_accels_grid_dims.z,
			calc_accels_block_dims.x,
			calc_accels_block_dims.y,
			calc_accels_block_dims.z
		);
	}
	fflush(stdout);
	#endif

	// Sum Accelerations

	int global_sum_length = NUMENTITIES;

	while (1 < sum_accels_block_dims.x) {

		setSumAccelsDims(global_sum_length, &sum_accels_grid_dims, &sum_accels_block_dims);

		sumAccels<<<sum_accels_grid_dims, sum_accels_block_dims, sizeof(double) * (sum_accels_block_dims.x)>>>(accels, global_sum_length);

		#ifdef DEBUG
		cudaError_t sum_accels_error = cudaGetLastError();
		if (sum_accels_error != cudaSuccess) {
			printf("sumAccels kernel launch failed! %s: %s\n",
				cudaGetErrorName(sum_accels_error),
				cudaGetErrorString(sum_accels_error)
			);
			printf("\tsumAccels Config: gridDims: {%d %d %d}, blockDims: {%d %d %d}\n",
				sum_accels_grid_dims.x,
				sum_accels_grid_dims.y,
				sum_accels_grid_dims.z,
				sum_accels_block_dims.x,
				sum_accels_block_dims.y,
				sum_accels_block_dims.z
			);
		}
		fflush(stdout);
		#endif

		global_sum_length = sum_accels_grid_dims.x;
	}

	// Calculating Changes

	calcChanges<<<calc_changes_grid_dims, calc_changes_block_dims>>>(accels, device_velocities, device_positions);

	#ifdef DEBUG
	cudaError_t calc_changes_error = cudaGetLastError();
	if (calc_changes_error != cudaSuccess) {
		printf("calcChanges kernel launch failed! %s: %s\n",
			cudaGetErrorName(calc_changes_error),
			cudaGetErrorString(calc_changes_error)
		);
		printf("\tcalcChanges Config: gridDims: {%d %d %d}, blockDims: {%d %d %d}\n",
			calc_changes_grid_dims.x,
			calc_changes_grid_dims.y,
			calc_changes_grid_dims.z,
			calc_changes_block_dims.x,
			calc_changes_block_dims.y,
			calc_changes_block_dims.z
		);
	}
	fflush(stdout);
	#endif
}
