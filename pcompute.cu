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
		 * Here is where I found the solution to this: 
		 * https://stackoverflow.com/questions/14406364/different-results-for-cuda-addition-on-host-and-on-gpu
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

void compute () {

	int i, j, k;

	dim3 blocks(NUMENTITIES, BLOCKS_PER_ROW);
	dim3 threads(THREADS_PER_BLOCK);

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
