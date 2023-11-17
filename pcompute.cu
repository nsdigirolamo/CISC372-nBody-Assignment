#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "vector.h"
#include "config.h"

#define THREAD_MAXIMUM 1024

#define BLOCKS_PER_ROW ceil((double)(NUMENTITIES) / (double)(THREAD_MAXIMUM))
#define THREADS_PER_BLOCK THREAD_MAXIMUM < NUMENTITIES ? THREAD_MAXIMUM : NUMENTITIES

__global__ void calculateAccelerations (vector3* hVel, vector3* hPos, double* mass, vector3* values, vector3** accels) {

	int row = blockIdx.x;

	int first_col = THREADS_PER_BLOCK * blockIdx.y;
	int col = first_col + threadIdx.x;

	if (NUMENTITIES <= col) {
		return;
	}

	if (row == col) { 

		FILL_VECTOR(accels[row][col], 0, 0, 0); 

	} else {

		vector3 distance;

		for (int i = 0; i < 3; i++) {
			distance[i] = hPos[row][i] - hPos[col][i];
		}

		double magnitude_sq = distance[0] * distance[0] + distance[1] * distance[1] + distance[2] * distance[2];
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

	dim3 blocks(NUMENTITIES, BLOCKS_PER_ROW);
	dim3 threads(THREADS_PER_BLOCK);

	calculateAccelerations<<<blocks, threads>>>(hVel, hPos, mass, values, accels);

	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) 
		printf("Kernel Launch Failed with Error: %s\n", cudaGetErrorString(err));
	
	cudaDeviceSynchronize();

	for (i = 0; i < NUMENTITIES; i++) {

		vector3 accel_sum={0,0,0};

		for (j = 0; j < NUMENTITIES; j++) {
			for (k = 0; k < 3; k++) {
				accel_sum[k] += accels[i][j][k];
			}
		}

		for (k = 0; k < 3; k++){
			hVel[i][k] += accel_sum[k] * INTERVAL;
			hPos[i][k] += hVel[i][k] * INTERVAL;
		}
	}

	cudaFree(accels);
	cudaFree(values);
}
