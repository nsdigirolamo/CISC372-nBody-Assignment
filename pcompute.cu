#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "vector.h"
#include "config.h"

#define THREAD_MAXIMUM 1024
#define VECTOR_MAXIMUM (THREAD_MAXIMUM / 3)
#define WARP_SIZE 32

__global__ void fillAccels (
		vector3* hVel, 
		vector3* hPos, 
		double* mass, 
		vector3* values, 
		vector3** accels, 
		int vectors_per_block
	) {

	int vector_id = threadIdx.x;
	int thread_id = threadIdx.y;
	int row = blockIdx.x;
	int col = (vectors_per_block * blockIdx.y) + vector_id;

	if (col < NUMENTITIES) {
		if (row == col) { 

			FILL_VECTOR(accels[row][col], 0, 0, 0); 

		} else {

			__shared__ vector3 distances[VECTOR_MAXIMUM];

			distances[vector_id][thread_id] = hPos[row][thread_id] - hPos[col][thread_id];

			__syncthreads();

			double magnitude_sq = distances[vector_id][0] * distances[vector_id][0] + distances[vector_id][1] * distances[vector_id][1] + distances[vector_id][2] * distances[vector_id][2];
			double magnitude = sqrt(magnitude_sq);
			double accelmag = -1 * GRAV_CONSTANT * mass[col] / magnitude_sq;

			accels[row][col][thread_id] = accelmag * distances[vector_id][thread_id] / magnitude;

		}
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
	int vectors_per_block = VECTOR_MAXIMUM < NUMENTITIES ? VECTOR_MAXIMUM : NUMENTITIES;

	dim3 blocks(NUMENTITIES, blocks_per_row);
	dim3 threads(vectors_per_block, 3);

	fillAccels<<<blocks, threads>>>(hVel, hPos, mass, values, accels, vectors_per_block);

	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) 
		printf("Kernel Launch Failed with Error: %s\n", cudaGetErrorString(err));
	
	cudaDeviceSynchronize();

	for (i=0;i<NUMENTITIES;i++){
		for (j=0;j<NUMENTITIES;j++){
			if (i==j) {
				FILL_VECTOR(accels[i][j],0,0,0);
			}
			else{
				vector3 distance;
				for (k=0;k<3;k++) distance[k]=hPos[i][k]-hPos[j][k];
				double magnitude_sq=distance[0]*distance[0]+distance[1]*distance[1]+distance[2]*distance[2];
				double magnitude=sqrt(magnitude_sq);
				double accelmag=-1*GRAV_CONSTANT*mass[j]/magnitude_sq;
				FILL_VECTOR(accels[i][j],accelmag*distance[0]/magnitude,accelmag*distance[1]/magnitude,accelmag*distance[2]/magnitude);
			}
		}
	}

	//sum up the rows of our matrix to get effect on each entity, then update velocity and position.
	for (i=0;i<NUMENTITIES;i++){
		vector3 accel_sum={0,0,0};
		for (j=0;j<NUMENTITIES;j++){
			for (k=0;k<3;k++)
				accel_sum[k]+=accels[i][j][k];
		}
		//compute the new velocity based on the acceleration and time interval
		//compute the new position based on the velocity and time interval
		for (k=0;k<3;k++){
			hVel[i][k]+=accel_sum[k]*INTERVAL;
			hPos[i][k]+=hVel[i][k]*INTERVAL;
		}
	}

	cudaFree(accels);
	cudaFree(values);
}
