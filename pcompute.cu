#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "pvector.h"
#include "config.h"

#define THREAD_MAXIMUM 1024
#define BLOCKS_PER_ROW (ceil((double)(NUMENTITIES) / (double)(THREAD_MAXIMUM)))
#define THREADS_PER_BLOCK (THREAD_MAXIMUM < NUMENTITIES ? THREAD_MAXIMUM : NUMENTITIES)
#define SPATIAL_AXES 3


__global__ void calcDists (vector3** dists, vector3* positions, double* masses) {

	int row = blockIdx.x;
	int col = (THREADS_PER_BLOCK * blockIdx.y) + threadIdx.x;
	int axis = blockIdx.z;

	if (NUMENTITIES <= col) return;

	if (row == col) {

		dists[row][col][axis] = 0;

	} else {

		dists[row][col][axis] = positions[row][axis] - positions[col][axis];

	}
}


__global__ void calcAccels (vector3** accels, vector3** dists, double* masses) {

	int row = blockIdx.x;
	int col = (THREADS_PER_BLOCK * blockIdx.y) + threadIdx.x;
	int axis = blockIdx.z;

	if (NUMENTITIES <= col) return;

	if (row == col) {

		accels[row][col][axis] = 0;

	} else {

		#ifdef STRICT_ACCELS

		double magnitude_sq = __dadd_rn(__dadd_rn(__dmul_rn(dists[row][col][0], dists[row][col][0]), __dmul_rn(dists[row][col][1], dists[row][col][1])), __dmul_rn(dists[row][col][2], dists[row][col][2]));

		#else

		double magnitude_sq = dists[row][col][0] * dists[row][col][0] + dists[row][col][1] * dists[row][col][1] + dists[row][col][2] * dists[row][col][2];

		#endif

		double magnitude = sqrt(magnitude_sq);
		double accelmag = -1 * GRAV_CONSTANT * masses[col] / magnitude_sq;
		accels[row][col][axis] = accelmag * dists[row][col][axis] / magnitude;

	}
}

__global__ void sumAccels (vector3** accels, vector3* accel_sums) {

	int row = blockIdx.x;
	int col = (THREADS_PER_BLOCK * blockIdx.y) + threadIdx.x;
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
	int col = (THREADS_PER_BLOCK * blockIdx.y) + threadIdx.x;
	int axis = blockIdx.z;

	if (col != 0) return;

	velocities[row][axis] += accel_sums[row][axis] * INTERVAL;
	positions[row][axis] += velocities[row][axis] * INTERVAL;
}

void compute () {

	dim3 blocks (NUMENTITIES, BLOCKS_PER_ROW, SPATIAL_AXES);
	dim3 threads (THREADS_PER_BLOCK);

	#ifdef DEBUG
	cudaError_t e = cudaGetLastError();
	if (e != cudaSuccess)
		printf("Error before compute! %s: %s\n",
			cudaGetErrorName(e),
			cudaGetErrorString(e)
		);
	#endif

	// Calculate Distances
	calcDists<<<blocks, threads>>>(dists, device_positions, device_masses);
	#ifdef DEBUG
	cudaError_t calc_dists_error = cudaGetLastError();
	if (calc_dists_error != cudaSuccess)
		printf("calcDists kernel launch failed! %s: %s\n",
			cudaGetErrorName(calc_dists_error),
			cudaGetErrorString(calc_dists_error)
		);
	#endif
	cudaDeviceSynchronize();

	// Calculate Accelerations
	calcAccels<<<blocks, threads>>>(accels, dists, device_masses);
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
