#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "vector.h"
#include "config.h"

#define THREAD_MAXIMUM 1024
#define BLOCKS_PER_ROW (ceil((double)(NUMENTITIES) / (double)(THREAD_MAXIMUM)))
#define THREADS_PER_BLOCK (THREAD_MAXIMUM < NUMENTITIES ? THREAD_MAXIMUM : NUMENTITIES)

__global__ void calculateAccelerations (vector3** accels, vector3* hPos, double* masses) {

	int row = blockIdx.x;
	int col = (THREADS_PER_BLOCK * blockIdx.y) + threadIdx.x;

	if (NUMENTITIES <= col) return;

	if (row == col) {

		FILL_VECTOR(accels[row][col], 0, 0, 0);

	} else {

		vector3 distance;

		for (int i = 0; i < 3; i++) {
			distance[i] = hPos[row][i] - hPos[col][i];
		}

		#ifdef STRICT_CALC_ACCELS

		/**
		 * Below's line of code is brought to you by CUDA's fused multiply-add. Fused multiply-add is faster and 
		 * more accurate than separate operations, but it causes the math to differ from the CPU's math. 
		 * So we need to do the below to disable it. Also it doesn't seem like fused multiple-add is *that* much
		 * faster than doing it the regular way, so we should be fine keeping it like below.
		 * 
		 * Here is where I found the solution to this: 
		 * https://stackoverflow.com/questions/14406364/different-results-for-cuda-addition-on-host-and-on-gpu
		 * 
		 * And here is some more in-depth reading: 
		 * https://docs.nvidia.com/cuda/floating-point/index.html
		 * 
		 */

		double magnitude_sq = __dadd_rn(__dadd_rn(__dmul_rn(distance[0], distance[0]), __dmul_rn(distance[1], distance[1])), __dmul_rn(distance[2], distance[2]));

		#else

		double magnitude_sq = distance[0] * distance[0] + distance[1] * distance[1] + distance[2] * distance[2];

		#endif

		double magnitude = sqrt(magnitude_sq);
		double accelmag = -1 * GRAV_CONSTANT * masses[col] / magnitude_sq;

		FILL_VECTOR(
			accels[row][col],
			accelmag * distance[0] / magnitude,
			accelmag * distance[1] / magnitude,
			accelmag * distance[2] / magnitude
		);
	}
}

__global__ void sumAccelerations (vector3** accels, vector3* accel_sums) {

	int row = blockIdx.x;
	int col = (THREADS_PER_BLOCK * blockIdx.y) + threadIdx.x;

	if (col != 0) return;

	for (int i = 0; i < 3; i++) {
		accel_sums[row][i] = 0;
	}

	for (int i = 0; i < NUMENTITIES; i++) {
		for (int j = 0; j < 3; j++) {
			accel_sums[row][j] += accels[row][i][j];
		}
	}
}

void compute () {

	dim3 blocks (NUMENTITIES, BLOCKS_PER_ROW);
	dim3 threads (THREADS_PER_BLOCK);

	// Calculate Accelerations Starts Here

	vector3* accel_values;
	cudaMallocManaged(&accel_values, sizeof(vector3) * NUMENTITIES * NUMENTITIES);
	vector3** accels;
	cudaMallocManaged(&accels, sizeof(vector3*) * NUMENTITIES);

	for (int i = 0; i < NUMENTITIES; i++) {
		accels[i] = &accel_values[i * NUMENTITIES];
	}

	calculateAccelerations<<<blocks, threads>>>(accels, hPos, mass);
	cudaError_t calculate_accelerations_error = cudaGetLastError();
	if (calculate_accelerations_error != cudaSuccess) 
		printf("calculateAccelerations kernel launch failed with Error: %s\n",
			cudaGetErrorString(calculate_accelerations_error)
		);
	cudaDeviceSynchronize();

	// Summing Accelerations Starts Here

	vector3* accel_sums;
	cudaMallocManaged(&accel_sums, sizeof(vector3) * NUMENTITIES);

	sumAccelerations<<<blocks, threads>>>(accels, accel_sums);
	cudaError_t sum_accelerations_error = cudaGetLastError();
	if (sum_accelerations_error != cudaSuccess) 
		printf("sumAccelerations kernel launch failed with Error: %s\n",
			cudaGetErrorString(sum_accelerations_error)
		);
	cudaDeviceSynchronize();

	// Calculating Changes Starts Here

	for (int i = 0; i < NUMENTITIES; i++) {
		for (int j = 0; j < 3; j++){
			hVel[i][j] += accel_sums[i][j] * INTERVAL;
			hPos[i][j] += hVel[i][j] * INTERVAL;
		}
	}

	cudaFree(accel_values);
	cudaFree(accels);
}
