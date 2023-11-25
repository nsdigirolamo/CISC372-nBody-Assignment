#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "vector.h"
#include "config.h"

#define THREAD_MAXIMUM 1024
#define BLOCKS_PER_ROW (ceil((double)(NUMENTITIES) / (double)(THREAD_MAXIMUM)))
#define ENTITIES_PER_BLOCK ((THREAD_MAXIMUM / 3) < NUMENTITIES ? (THREAD_MAXIMUM / 3) : NUMENTITIES)
#define THREADS_PER_ENTITY 3
#define THREADS_PER_BLOCK (ENTITIES_PER_BLOCK * THREADS_PER_ENTITY)

__device__ void calculateDistances (int entity_id, int thread_id, int row, int col, vector3* distances, vector3* hPos) {

	if (NUMENTITIES <= col) return;

	if (row == col) {

		distances[entity_id][thread_id] = 0;

	} else {

		distances[entity_id][thread_id] = hPos[row][thread_id] - hPos[col][thread_id];

	}
}

__global__ void calculateAccelerations (vector3** accels, vector3* hPos, double* masses) {

	int entity_id = threadIdx.x;
	int thread_id = threadIdx.y;
	int row = blockIdx.x;
	int col = (ENTITIES_PER_BLOCK * blockIdx.y) + entity_id;

	if (NUMENTITIES <= col) return;

	if (row == col) {

		accels[row][col][thread_id] = 0;

	} else {

		__shared__ vector3 distances[ENTITIES_PER_BLOCK];

		calculateDistances(entity_id, thread_id, row, col, distances, hPos);

		__syncthreads();

		/**
		 * Below's incredibly horrible line of code is brought to you by CUDA's implementation of fused multiply-add.
		 * Fused multiply-add is supposed to be faster and more accurate than separate operations, but it causes
		 * the math to differ from the CPU's math. So we need to do the below to disable it.
		 * 
		 * Here is where I found the solution to this: 
		 * https://stackoverflow.com/questions/14406364/different-results-for-cuda-addition-on-host-and-on-gpu
		 * And here is some more in-depth reading: 
		 * https://docs.nvidia.com/cuda/floating-point/index.html
		 */
		double magnitude_sq = __dadd_rn(__dadd_rn(__dmul_rn(distances[entity_id][0], distances[entity_id][0]), __dmul_rn(distances[entity_id][1], distances[entity_id][1])), __dmul_rn(distances[entity_id][2], distances[entity_id][2]));
		double magnitude = sqrt(magnitude_sq);
		double accelmag = -1 * GRAV_CONSTANT * masses[col] / magnitude_sq;
		accels[row][col][thread_id] = accelmag * distances[entity_id][thread_id] / magnitude;

	}
}

void compute () {

	int i, j, k;

	dim3 blocks(NUMENTITIES, BLOCKS_PER_ROW);
	dim3 threads(ENTITIES_PER_BLOCK, THREADS_PER_ENTITY);

	vector3* accel_values;
	cudaMallocManaged(&accel_values, sizeof(vector3) * NUMENTITIES * NUMENTITIES);
	vector3** accels;
	cudaMallocManaged(&accels, sizeof(vector3*) * NUMENTITIES);

	for (i = 0; i < NUMENTITIES; i++) {
		accels[i] = &accel_values[i * NUMENTITIES];
	}

	calculateAccelerations<<<blocks, threads>>>(accels, hPos, mass);
	cudaError_t calculate_accelerations_error = cudaGetLastError();
	if (calculate_accelerations_error != cudaSuccess) 
		printf("calculateAccelerations kernel launch failed with Error: %s\n",
			cudaGetErrorString(calculate_accelerations_error)
		);
	cudaDeviceSynchronize();

	for (i=0;i<NUMENTITIES;i++){
		vector3 accel_sum={0,0,0};
		for (j=0;j<NUMENTITIES;j++){
			for (k=0;k<3;k++)
				accel_sum[k]+=accels[i][j][k];
		}
		for (k=0;k<3;k++){
			hVel[i][k]+=accel_sum[k]*INTERVAL;
			hPos[i][k]+=hVel[i][k]*INTERVAL;
		}
	}

	cudaFree(accel_values);
	cudaFree(accels);
}
