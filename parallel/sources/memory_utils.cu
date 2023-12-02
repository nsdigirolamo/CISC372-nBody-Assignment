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

void initHostMemory () {

	host_velocities = (vector3*) malloc(sizeof(vector3) * NUMENTITIES);
	host_positions = (vector3*) malloc(sizeof(vector3) * NUMENTITIES);
	host_masses = (double*) malloc(sizeof(double) * NUMENTITIES);

}

void initDeviceMemory () {\

	cudaError_t e;

	e = cudaMalloc(&device_velocities, sizeof(vector3) * NUMENTITIES);
	#ifdef DEBUG
	handleCudaError(e, "initDeviceMemory velocities");
	#endif

	e = cudaMalloc(&device_positions, sizeof(vector3) * NUMENTITIES);
	#ifdef DEBUG
	handleCudaError(e, "initDeviceMemory positions");
	#endif

	e = cudaMalloc(&device_masses, sizeof(double) * NUMENTITIES);
	#ifdef DEBUG
	handleCudaError(e, "initDeviceMemory masses");
	#endif

	e = cudaMallocPitch(&accels, &accels_pitch, sizeof(vector3) * (NUMENTITIES + 1), (NUMENTITIES + 1));
	#ifdef DEBUG
	handleCudaError(e, "initDeviceMemory accels");
	#endif
}

void copyHostToDevice () {

	cudaError_t e;

	e = cudaMemcpy(device_velocities, host_velocities, sizeof(vector3) * NUMENTITIES, cudaMemcpyHostToDevice);
	#ifdef DEBUG
	handleCudaError(e, "copyHostToDevice velocities");
	#endif

	e = cudaMemcpy(device_positions, host_positions, sizeof(vector3) * NUMENTITIES, cudaMemcpyHostToDevice);
	#ifdef DEBUG
	handleCudaError(e, "copyHostToDevice positions");
	#endif

	e = cudaMemcpy(device_masses, host_masses, sizeof(double) * NUMENTITIES, cudaMemcpyHostToDevice);
	#ifdef DEBUG
	handleCudaError(e, "copyHostToDevice masses");
	#endif
}

void copyDeviceToHost () {

	cudaError_t e;

	e = cudaMemcpy(host_velocities, device_velocities, sizeof(vector3) * NUMENTITIES, cudaMemcpyDeviceToHost);
	#ifdef DEBUG 
	handleCudaError(e, "copyDeviceToHost velocities"); 
	#endif

	e = cudaMemcpy(host_positions, device_positions, sizeof(vector3) * NUMENTITIES, cudaMemcpyDeviceToHost);
	#ifdef DEBUG
	handleCudaError(e, "copyDeviceToHost positions");
	#endif
}

void freeHostMemory () {

	free(host_velocities);
	free(host_positions);
	free(host_masses);

}

void freeDeviceMemory () {

	cudaError_t e;

	e = cudaFree(device_velocities);
	#ifdef DEBUG 
	handleCudaError(e, "freeDeviceMemory velocities"); 
	#endif

	e = cudaFree(device_positions);
	#ifdef DEBUG 
	handleCudaError(e, "freeDeviceMemory positions"); 
	#endif

	e = cudaFree(device_masses);
	#ifdef DEBUG 
	handleCudaError(e, "freeDeviceMemory masses"); 
	#endif

	e = cudaFree(accels);
	#ifdef DEBUG 
	handleCudaError(e, "freeDeviceMemory accels"); 
	#endif
}