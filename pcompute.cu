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

		/**
		 * Below's incredibly horrible line of code is brought to you by CUDA's implementation of fused multiply-add.
		 * Fused multiply-add is supposed to be faster and more accurate than separate operations, but it causes
		 * the math to differ from the CPU's math. So we need to do the below to disable it.
		 * 
		 * Original Implentation:
		 * double magnitude_sq = distance[0] * distance[0] + distance[1] * distance[1] + distance[2] * distance[2]
		 * 
		 * Here is where I found the solution to this: 
		 * https://stackoverflow.com/questions/14406364/different-results-for-cuda-addition-on-host-and-on-gpu
		 * 
		 * And here is some more in-depth reading: 
		 * https://docs.nvidia.com/cuda/floating-point/index.html
		 */
		double magnitude_sq = __dadd_rn(__dadd_rn(__dmul_rn(distance[0], distance[0]), __dmul_rn(distance[1], distance[1])), __dmul_rn(distance[2], distance[2]));
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

__global__ void sumAccelerations (vector3** accels, int iteration, int offset, vector3* accel_sums) {

	int row = blockIdx.x;
	int col = (THREADS_PER_BLOCK * blockIdx.y) + threadIdx.x;

	// bool col_past_length = NUMENTITIES <= col;
	// bool col_needs_adding = col % offset == 0;
	// bool offset_past_length = NUMENTITIES <= col + offset;

	// if (col_past_length || !col_needs_adding || offset_past_length) return;

	/**
	 * 
	 * TODO: Optimize the below for loop!
	 * 
	 * Below causes some threads to do work when they really don't need to be doing work.
	 * For example, iteration 0 will check to see if col % (2^0) == 0. This is true for
	 * all numbers, so all threads will be adding with their offset neighbors. But in the
	 * next iteration, only columns with col % (2^1) == 0 are needed (the even columns).
	 * So all the odd columns in iteration 0 are doing work for no reason. This issue applies
	 * to every iteration... in fact, all iterations are technically doing two times as much
	 * "work" as they need to be, because every other thread's work will be ignored in the
	 * next iteration!
	 */

	/**

	for (int i = 0; i < 3; i++) {
		accels[row][col][i] = __dadd_ru(accels[row][col][i], accels[row][col + offset][i]);
	}

	*/

	if (col == 0) {

		for (int i = 0; i < 3; i++) {
			accel_sums[row][i] = 0;
		}

		for (int i = 0; i < NUMENTITIES; i++) {
			for (int j = 0; j < 3; j++) {
				accel_sums[row][j] += accels[row][i][j];
			}
		}
	}
}

void compute () {

	dim3 blocks (NUMENTITIES, BLOCKS_PER_ROW);
	dim3 threads (THREADS_PER_BLOCK);

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

	int i = 0;
	int offset = pow(2, i);

	vector3* accel_sums;
	cudaMallocManaged(&accel_sums, sizeof(vector3) * NUMENTITIES);

	while (offset < NUMENTITIES) {

		sumAccelerations<<<blocks, threads>>>(accels, i, offset, accel_sums);
		cudaError_t sum_accelerations_error = cudaGetLastError();
		if (sum_accelerations_error != cudaSuccess) 
			printf("sumAccelerations kernel launch failed with Error: %s\n",
				cudaGetErrorString(sum_accelerations_error)
			);
		cudaDeviceSynchronize();

		i++;
		offset = pow(2, i);
	}

	for (int i = 0; i < NUMENTITIES; i++) {
		for (int j = 0; j < 3; j++){
			hVel[i][j] += accel_sums[i][j] * INTERVAL;
			hPos[i][j] += hVel[i][j] * INTERVAL;
		}
	}

	cudaFree(accel_values);
	cudaFree(accels);
}
