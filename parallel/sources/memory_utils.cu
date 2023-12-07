#include <stdlib.h>
#include <stdio.h>

#include "config.cuh"
#include "nbody.cuh"
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

/**
 * Initializes memory on the host.
 */
void initHostMemory () {

	host_velocities = (vector3*) malloc(sizeof(vector3) * NUMENTITIES);
	host_positions = (vector3*) malloc(sizeof(vector3) * NUMENTITIES);
	host_masses = (double*) malloc(sizeof(double) * NUMENTITIES);

}

/**
 * Initializes memory on the device.
 */
void initDeviceMemory () {

	#ifdef DEBUG
	handleCudaError(cudaGetLastError(), "initDeviceMemory filter");
	#endif

	cudaMalloc(&device_velocities, sizeof(vector3) * NUMENTITIES);
	#ifdef DEBUG
	handleCudaError(cudaGetLastError(), "initDeviceMemory velocities");
	#endif

	cudaMalloc(&device_positions, sizeof(vector3) * NUMENTITIES);
	#ifdef DEBUG
	handleCudaError(cudaGetLastError(), "initDeviceMemory positions");
	#endif

	cudaMalloc(&device_masses, sizeof(double) * NUMENTITIES);
	#ifdef DEBUG
	handleCudaError(cudaGetLastError(), "initDeviceMemory masses");
	#endif

	cudaMallocPitch(&accels, &accels_pitch, sizeof(vector3) * NUMENTITIES, NUMENTITIES);
	#ifdef DEBUG
	handleCudaError(cudaGetLastError(), "initDeviceMemory accels");
	#endif
}

/**
 * Copies memory from the host to the device.
 */
void copyHostToDevice () {

	#ifdef DEBUG
	handleCudaError(cudaGetLastError(), "copyHostToDevice filter");
	#endif

	cudaMemcpy(device_velocities, host_velocities, sizeof(vector3) * NUMENTITIES, cudaMemcpyHostToDevice);
	#ifdef DEBUG
	handleCudaError(cudaGetLastError(), "copyHostToDevice velocities");
	#endif

	cudaMemcpy(device_positions, host_positions, sizeof(vector3) * NUMENTITIES, cudaMemcpyHostToDevice);
	#ifdef DEBUG
	handleCudaError(cudaGetLastError(), "copyHostToDevice positions");
	#endif

	cudaMemcpy(device_masses, host_masses, sizeof(double) * NUMENTITIES, cudaMemcpyHostToDevice);
	#ifdef DEBUG
	handleCudaError(cudaGetLastError(), "copyHostToDevice masses");
	#endif
}

/**
 * Copies memory from the device to the host.
 */
void copyDeviceToHost () {

	#ifdef DEBUG
	handleCudaError(cudaGetLastError(), "copyDeviceToHost filter");
	#endif

	cudaMemcpy(host_velocities, device_velocities, sizeof(vector3) * NUMENTITIES, cudaMemcpyDeviceToHost);
	#ifdef DEBUG 
	handleCudaError(cudaGetLastError(), "copyDeviceToHost velocities"); 
	#endif

	cudaMemcpy(host_positions, device_positions, sizeof(vector3) * NUMENTITIES, cudaMemcpyDeviceToHost);
	#ifdef DEBUG
	handleCudaError(cudaGetLastError(), "copyDeviceToHost positions");
	#endif
}

/**
 * Frees the host memory.
 */
void freeHostMemory () {

	free(host_velocities);
	free(host_positions);
	free(host_masses);

}

/**
 * Frees the device memory.
 */
void freeDeviceMemory () {

	#ifdef DEBUG
	handleCudaError(cudaGetLastError(), "freeDeviceMemory filter");
	#endif

	cudaFree(device_velocities);
	#ifdef DEBUG 
	handleCudaError(cudaGetLastError(), "freeDeviceMemory velocities"); 
	#endif

	cudaFree(device_positions);
	#ifdef DEBUG 
	handleCudaError(cudaGetLastError(), "freeDeviceMemory positions"); 
	#endif

	cudaFree(device_masses);
	#ifdef DEBUG 
	handleCudaError(cudaGetLastError(), "freeDeviceMemory masses"); 
	#endif

	cudaFree(accels);
	#ifdef DEBUG 
	handleCudaError(cudaGetLastError(), "freeDeviceMemory accels"); 
	#endif
}