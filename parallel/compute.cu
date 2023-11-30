#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "vector.h"
#include "config.h"

int calcGridDim (int block_width, int entity_count) {

	if (entity_count < block_width) return 1;

	int grid_width = entity_count / block_width;

	if (entity_count % block_width == 0) return grid_width;

	return grid_width + 1;
}

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
	// Double our offsets because each thread is assigned two data points globally
	int global_col = (blockIdx.x * blockDim.x * 2) + (local_col * 2);
	int spatial_axis = threadIdx.z;

	__shared__ vector3 sums[SUM_LENGTH];

	/**
	 * We need to zero out the values in sums because some local_cols may not be
	 * needed. If they don't zero out their values, some other thread may try
	 * to add a garbage value to their own good value.
	 */
	sums[local_col][spatial_axis] = 0;

	if (global_sum_length <= global_col) return;

	// Below initializes the shared array with summed global values.

	int offset = 1;
	bool neighbor_exceeds_bounds = global_sum_length <= global_col + offset;

	if (!neighbor_exceeds_bounds) {

		sums[local_col][spatial_axis] = accels[global_row][global_col][spatial_axis] + accels[global_row][global_col + 1][spatial_axis];

	} else {

		sums[local_col][spatial_axis] = accels[global_row][global_col][spatial_axis];

	}

	// We will now treat the local_col as a thread id
	int thread_id = local_col;

	for (offset = 1; offset < SUM_LENGTH; offset *= 2) {
		// This will produce a strided index from the thread_id
		int i = 2 * offset * thread_id;
		// This allows for a non-divergent branch within the loop.
		if (i < SUM_LENGTH) {
			sums[i][spatial_axis] += sums[i + offset][spatial_axis];
		}
		__syncthreads();
	}

	if (local_col == 0) accels[global_row][blockIdx.x][spatial_axis] = sums[local_col][spatial_axis];
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
	dim3 sum_block_dims (SUM_LENGTH, 1, SPATIAL_DIMS);

	while (1 < global_sum_length) {

		int sum_grid_width = calcGridDim(SUM_LENGTH * 2, global_sum_length); // Multiply by two because each thread reduces two data points in accels.
		dim3 sum_grid_dims (sum_grid_width, NUMENTITIES, 1);

		sumAccels<<<sum_grid_dims, sum_block_dims>>>(accels, global_sum_length);

		#ifdef DEBUG
		cudaError_t sum_accels_error = cudaGetLastError();
		if (sum_accels_error != cudaSuccess) {
			printf("sumAccels kernel launch failed! %s: %s\n",
				cudaGetErrorName(sum_accels_error),
				cudaGetErrorString(sum_accels_error)
			);
			printf("\tsumAccels Config: gridDims: {%d %d %d}, blockDims: {%d %d %d}\n",
				sum_grid_dims.x,
				sum_grid_dims.y,
				sum_grid_dims.z,
				sum_block_dims.x,
				sum_block_dims.y,
				sum_block_dims.z
			);
		}
		fflush(stdout);
		#endif

		global_sum_length = sum_grid_width;
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
