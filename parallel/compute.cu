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

		__shared__ vector3 distances[SQUARE_SIZE][SQUARE_SIZE];

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

	int local_row = threadIdx.y;
	int global_row = (blockIdx.y * blockDim.y) + local_row;
	int spatial_axis = threadIdx.z;

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

	int accels_grid_width = calcGridDim(SQUARE_SIZE, NUMENTITIES);
	dim3 accels_grid_dims (accels_grid_width, accels_grid_width, 1);
	dim3 accels_block_dims (SQUARE_SIZE, SQUARE_SIZE, SPATIAL_DIMS);

	calcAccels<<<accels_grid_dims, accels_block_dims>>>(accels, device_positions, device_masses);

	#ifdef DEBUG
	cudaError_t calc_accels_error = cudaDeviceSynchronize();
	if (calc_accels_error != cudaSuccess) {
		printf("calcAccels kernel launch failed! %s: %s\n",
			cudaGetErrorName(calc_accels_error),
			cudaGetErrorString(calc_accels_error)
		);
		printf("\tcalcAccels Config: gridDims: {%d %d %d}, blockDims: {%d %d %d}\n",
			accels_grid_dims.x,
			accels_grid_dims.y,
			accels_grid_dims.z,
			accels_block_dims.x,
			accels_block_dims.y,
			accels_block_dims.z
		);
	}
	fflush(stdout);
	#endif

	// Sum Accelerations

	/**
	 * TODO: This isn't the best right now. Only assigns three threads per row.
	 */
	dim3 sum_grid_dims (1, NUMENTITIES, 1);
	dim3 sum_block_dims (1, 1, SPATIAL_DIMS);

	sumAccels<<<sum_grid_dims, sum_block_dims>>>(accels);

	#ifdef DEBUG
	cudaError_t sum_accels_error = cudaDeviceSynchronize();;
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

	// Calculating Changes

	calcChanges<<<calc_changes_grid_dims, calc_changes_block_dims>>>(accels, device_velocities, device_positions);

	#ifdef DEBUG
	cudaError_t calc_changes_error = cudaDeviceSynchronize();;
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
