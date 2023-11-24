#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "vector.h"
#include "config.h"

#define THREAD_MAXIMUM 1024

__global__ void calculateAccelerations (
		vector3* hVel, 
		vector3* hPos, 
		double* mass, 
		vector3* values, 
		vector3** accels, 
		int threads_per_block
	) {

	int row = blockIdx.x;
	int col = (threads_per_block * blockIdx.y) + threadIdx.x;

	if (NUMENTITIES <= col) return; // right now, the row variable will never be greater than NUMENTITIES

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
		double accelmag = -1 * GRAV_CONSTANT * mass[col] / magnitude_sq;

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

	vector3* values;
	vector3** accels;

	cudaMallocManaged(&values, sizeof(vector3) * NUMENTITIES * NUMENTITIES);
	cudaMallocManaged(&accels, sizeof(vector3*) * NUMENTITIES);

	for (i = 0; i < NUMENTITIES; i++) {
		accels[i] = &values[i * NUMENTITIES];
	}

	int blocks_per_row = ceil((double)(NUMENTITIES) / (double)(THREAD_MAXIMUM));
	int threads_per_block = THREAD_MAXIMUM < NUMENTITIES ? (THREAD_MAXIMUM / 3) : NUMENTITIES;

	dim3 blocks(NUMENTITIES, blocks_per_row);
	dim3 threads(threads_per_block);

	calculateAccelerations<<<blocks, threads>>>(hVel, hPos, mass, values, accels, threads_per_block);

	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) 
		printf("Kernel Launch Failed with Error: %s\n", cudaGetErrorString(err));
	
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

	cudaFree(accels);
	cudaFree(values);
}
