#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "vector.h"
#include "config.h"

#define SPATIAL_AXES 3

__global__ void calcAccels (vector3** accels, vector3* positions, double* masses) {

	int row = blockIdx.x;
	int col = (blockDim.x * blockIdx.y) + threadIdx.x;
	int axis = blockIdx.z;

	if (NUMENTITIES <= col) return;

	if (row == col) {

		accels[row][col][axis] = 0;

	} else {

		vector3 distance = {0, 0, 0};

		for (int i = 0; i < 3; i++) {
			distance[i] = positions[row][i] - positions[col][i];
		}

		double magnitude_sq = distance[0] * distance[0] + distance[1] * distance[1] + distance[2] * distance[2];
		double magnitude = sqrt(magnitude_sq);
		double accelmag = -1 * GRAV_CONSTANT * masses[col] / magnitude_sq;
		accels[row][col][axis] = accelmag * distance[axis] / magnitude;

	}
}

__global__ void sumAccels (vector3** accels, vector3* accel_sums) {

	int row = blockIdx.x;
	int col = (blockDim.x * blockIdx.y) + threadIdx.x;
	int axis = blockIdx.z;

	if (col != 0) return;

	double accel_sum = 0;

	for (int i = 0; i < NUMENTITIES; i++) {
		accel_sum += accels[row][i][axis];
	}

	accel_sums[row][axis] = accel_sum;
}

__global__ void calcChanges (vector3* accel_sums, vector3* velocities, vector3* positions) {

	int row = blockIdx.x;
	int col = (blockDim.x * blockIdx.y) + threadIdx.x;
	int axis = blockIdx.z;

	if (col != 0) return;

	velocities[row][axis] += accel_sums[row][axis] * INTERVAL;
	positions[row][axis] += velocities[row][axis] * INTERVAL;
}

void compute () {

	dim3 blocks(NUMENTITIES, blocks_per_row, SPATIAL_AXES);
	dim3 threads(threads_per_block);

	#ifdef DEBUG
	cudaError_t e = cudaGetLastError();
	if (e != cudaSuccess)
		printf("Error before compute! %s: %s\n",
			cudaGetErrorName(e),
			cudaGetErrorString(e)
		);
	#endif

	// Calculate Accelerations
	calcAccels<<<blocks, threads>>>(accels, device_positions, device_masses);
	#ifdef DEBUG
	cudaError_t calc_accels_error = cudaGetLastError();
	if (calc_accels_error != cudaSuccess) 
		printf("calcAccels kernel launch failed! %s: %s\n",
			cudaGetErrorName(calc_accels_error),
			cudaGetErrorString(calc_accels_error)
		);
	#endif
	cudaDeviceSynchronize();

	// Sum Accelerations
	sumAccels<<<blocks, threads>>>(accels, accel_sums);
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
	calcChanges<<<blocks, threads>>>(accel_sums, device_velocities, device_positions);
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
