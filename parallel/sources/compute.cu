#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "config.cuh"
#include "kernel_utils.cuh"
#include "memory_utils.cuh"
#include "nbody.cuh"
#include "vector.cuh"

__global__ void calcAccels (vector3* accels, size_t accels_pitch, vector3* positions, double* masses) {

	int local_row = threadIdx.y;
	int local_col = threadIdx.x;

	int global_row = (blockIdx.y * blockDim.y) + local_row;
	int global_col = (blockIdx.x * blockDim.x) + local_col;
	int spatial_axis = threadIdx.z;

	if (NUMENTITIES <= global_col || NUMENTITIES <= global_row) return;

	vector3* accels_row = (vector3*)((char*)(accels) + global_row * accels_pitch);

	if (global_row == global_col) {

		accels_row[global_col][spatial_axis] = 0;

	} else {

		__shared__ vector3 distances[CALC_ACCELS_BLOCK_WIDTH][CALC_ACCELS_BLOCK_WIDTH];

		distances[local_row][local_col][spatial_axis] = positions[global_row][spatial_axis] - positions[global_col][spatial_axis];

		__syncthreads();

		double magnitude_sq = distances[local_row][local_col][0] * distances[local_row][local_col][0] + distances[local_row][local_col][1] * distances[local_row][local_col][1] + distances[local_row][local_col][2] * distances[local_row][local_col][2];
		double magnitude = sqrt(magnitude_sq);
		double accelmag = -1 * GRAV_CONSTANT * masses[global_col] / magnitude_sq;
		accels_row[global_col][spatial_axis] = accelmag * distances[local_row][local_col][spatial_axis] / magnitude;

	}
}

__global__ void sumAccels (vector3* accels, size_t accels_pitch, int global_sum_length) {

	int local_col = threadIdx.x;

	int global_row = blockIdx.y;
	int global_col = (blockIdx.x * blockDim.x) + local_col;
	int spatial_axis = blockIdx.z;

	extern __shared__ double sums[];
	int sums_length = blockDim.x;

	sums[local_col] = 0;

	if (global_sum_length <= global_col) return;

	vector3* accels_row = (vector3*)((char*)(accels) + global_row * accels_pitch);

	sums[local_col] = accels_row[global_col][spatial_axis];

	__syncthreads();

	for (int stride = sums_length / 2; 0 < stride; stride >>= 1) {
		if (local_col < stride) sums[local_col] += sums[local_col + stride];
		__syncthreads();
	}

	if (local_col == 0) accels_row[blockIdx.x][spatial_axis] = sums[local_col];
}

__global__ void calcChanges (vector3* accels, size_t accels_pitch, vector3* velocities, vector3* positions) {

	int local_row = threadIdx.y;
	int global_row = (blockIdx.y * blockDim.y) + local_row;
	int spatial_axis = blockIdx.z;

	if (NUMENTITIES <= global_row) return;

	vector3* accels_row = (vector3*)((char*)(accels) + global_row * accels_pitch);

	velocities[global_row][spatial_axis] += accels_row[0][spatial_axis] * INTERVAL;
	positions[global_row][spatial_axis] += velocities[global_row][spatial_axis] * INTERVAL; 
}

void compute () {

	#ifdef DEBUG
	cudaError_t e = cudaGetLastError();
	handleCudaError(e, "compute");
	#endif

	// Calculate Accelerations

	calcAccels<<<calc_accels_grid_dims, calc_accels_block_dims>>>(accels, accels_pitch, device_positions, device_masses);

	#ifdef DEBUG
	e = cudaGetLastError();
	if (e != cudaSuccess) {
		printf("Error in Kernel Detected!\n");
		printKernelDims("calcAccels", calc_accels_grid_dims, calc_accels_block_dims);
	}
	handleCudaError(cudaGetLastError(), "calcAccels");
	#endif

	// Sum Accelerations

	int global_sum_length = NUMENTITIES;

	while (1 < sum_accels_block_dims.x) {

		setSumAccelsDims(global_sum_length, &sum_accels_grid_dims, &sum_accels_block_dims);

		sumAccels<<<sum_accels_grid_dims, sum_accels_block_dims, sizeof(double) * (sum_accels_block_dims.x)>>>(accels, accels_pitch, global_sum_length);

		#ifdef DEBUG
		e = cudaGetLastError();
		if (e != cudaSuccess) {
			printf("Error in Kernel Detected!\n");
			printKernelDims("sumAccels", sum_accels_grid_dims, sum_accels_block_dims);
		}
		handleCudaError(cudaGetLastError(), "sumAccels");
		#endif

		global_sum_length = sum_accels_grid_dims.x;
	}

	// Calculating Changes

	calcChanges<<<calc_changes_grid_dims, calc_changes_block_dims>>>(accels, accels_pitch, device_velocities, device_positions);

	#ifdef DEBUG
	e = cudaGetLastError();
	if (e != cudaSuccess) {
		printf("Error in Kernel Detected!\n");
		printKernelDims("calcChanges", calc_changes_grid_dims, calc_changes_block_dims);
	}
	handleCudaError(cudaGetLastError(), "calcChanges");
	#endif
}
