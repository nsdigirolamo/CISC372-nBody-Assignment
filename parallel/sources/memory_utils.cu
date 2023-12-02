#include <stdlib.h>
#include <stdio.h>

#include "config.cuh"
#include "vector.cuh"

// Host Memory

vector3* host_velocities;
vector3* host_positions;
double* host_masses;

// Device Memory

vector3* device_velocities;
vector3* device_positions;
double* device_masses;

size_t accels_pitch;
vector3* accels;

void initHostMemory () {

	host_velocities = (vector3*) malloc(sizeof(vector3) * NUMENTITIES);
	host_positions = (vector3*) malloc(sizeof(vector3) * NUMENTITIES);
	host_masses = (double*) malloc(sizeof(double) * NUMENTITIES);

}

void initDeviceMemory () {

	cudaMalloc(&device_velocities, sizeof(vector3) * NUMENTITIES);
	cudaMalloc(&device_positions, sizeof(vector3) * NUMENTITIES);
	cudaMalloc(&device_masses, sizeof(double) * NUMENTITIES);
	cudaMallocPitch(&accels, &accels_pitch, sizeof(vector3) * (NUMENTITIES + 1), (NUMENTITIES + 1));

	#ifdef DEBUG
	cudaError_t e = cudaGetLastError();
	if (e != cudaSuccess)
		printf("Error in initDeviceMemory! %s: %s\n",
			cudaGetErrorName(e),
			cudaGetErrorString(e)
		);
	fflush(stdout);
	#endif
}

void copyHostToDevice () {

	cudaMemcpy(device_velocities, host_velocities, sizeof(vector3) * NUMENTITIES, cudaMemcpyHostToDevice);
	cudaMemcpy(device_positions, host_positions, sizeof(vector3) * NUMENTITIES, cudaMemcpyHostToDevice);
	cudaMemcpy(device_masses, host_masses, sizeof(double) * NUMENTITIES, cudaMemcpyHostToDevice);

	#ifdef DEBUG
	cudaError_t e = cudaGetLastError();
	if (e != cudaSuccess)
		printf("Error in copyHostToDevice! %s: %s\n",
			cudaGetErrorName(e),
			cudaGetErrorString(e)
		);
	fflush(stdout);
	#endif
}

void copyDeviceToHost () {

	cudaMemcpy(host_velocities, device_velocities, sizeof(vector3) * NUMENTITIES, cudaMemcpyDeviceToHost);
	cudaMemcpy(host_positions, device_positions, sizeof(vector3) * NUMENTITIES, cudaMemcpyDeviceToHost);

	#ifdef DEBUG
	cudaError_t e = cudaGetLastError();
	if (e != cudaSuccess)
		printf("Error in copyDeviceToHost! %s: %s\n",
			cudaGetErrorName(e),
			cudaGetErrorString(e)
		);
	fflush(stdout);
	#endif
}

void freeHostMemory () {

	free(host_velocities);
	free(host_positions);
	free(host_masses);

}

void freeDeviceMemory () {

	cudaFree(device_velocities);
	cudaFree(device_positions);
	cudaFree(device_masses);
	cudaFree(accels);

}