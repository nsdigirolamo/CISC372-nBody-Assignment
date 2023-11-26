#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include "pvector.h"
#include "config.h"
#include "planets.h"
#include "compute.h"

vector3* host_velocities;
vector3* host_positions;
double* host_masses;

vector3* device_velocities;
vector3* device_positions;
double* device_masses;

vector3** dists;
vector3** accels;
vector3* accel_sums;

void initHostMemory (int numObjects) {

	host_velocities = (vector3*) malloc(sizeof(vector3) * numObjects);
	host_positions = (vector3*) malloc(sizeof(vector3) * numObjects);
	host_masses = (double*) malloc(sizeof(double) * numObjects);

}

// todo: make this a kernel so we can just make all that data local to the device only.

void initDeviceMemory (int numObjects) {

	// Allocating device memory for velocities, positions, masses, and acceleration sums

	cudaMalloc(&device_velocities, sizeof(vector3) * numObjects);
	cudaMalloc(&device_positions, sizeof(vector3) * numObjects);
	cudaMalloc(&device_masses, sizeof(double) * numObjects);
	cudaMalloc(&accel_sums, sizeof(vector3) * numObjects);

	// Allocating device memory for distances

	cudaMalloc(&dists, sizeof(vector3*) * numObjects);
	vector3* host_dists[numObjects];
	for (int i = 0; i < numObjects; i++) {
		cudaMalloc(&host_dists[i], sizeof(vector3) * NUMENTITIES);
	}
	cudaMemcpy(dists, host_dists, sizeof(vector3*) * numObjects, cudaMemcpyHostToDevice);

	// Allocating device memory for accelerations

	cudaMalloc(&accels, sizeof(vector3*) * numObjects);
	vector3* host_accels[numObjects];
	for (int i = 0; i < numObjects; i++) {
		cudaMalloc(&host_accels[i], sizeof(vector3) * NUMENTITIES);
	}
	cudaMemcpy(accels, host_accels, sizeof(vector3*) * numObjects, cudaMemcpyHostToDevice);

	#ifdef DEBUG
	cudaError_t e = cudaGetLastError();
	if (e != cudaSuccess)
		printf("Error in initDeviceMemory! %s: %s\n",
			cudaGetErrorName(e),
			cudaGetErrorString(e)
		);
	#endif
}

void copyHostToDevice (int numObjects) {

	cudaMemcpy(device_velocities, host_velocities, sizeof(vector3) * numObjects, cudaMemcpyHostToDevice);
	cudaMemcpy(device_positions, host_positions, sizeof(vector3) * numObjects, cudaMemcpyHostToDevice);
	cudaMemcpy(device_masses, host_masses, sizeof(double) * numObjects, cudaMemcpyHostToDevice);

	#ifdef DEBUG
	cudaError_t e = cudaGetLastError();
	if (e != cudaSuccess)
		printf("Error in copyHostToDevice! %s: %s\n",
			cudaGetErrorName(e),
			cudaGetErrorString(e)
		);
	#endif
}

void copyDeviceToHost (int numObjects) {

	cudaMemcpy(host_velocities, device_velocities, sizeof(vector3) * numObjects, cudaMemcpyDeviceToHost);
	cudaMemcpy(host_positions, device_positions, sizeof(vector3) * numObjects, cudaMemcpyDeviceToHost);

	#ifdef DEBUG
	cudaError_t e = cudaGetLastError();
	if (e != cudaSuccess)
		printf("Error in copyDeviceToHost! %s: %s\n",
			cudaGetErrorName(e),
			cudaGetErrorString(e)
		);
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
	cudaFree(dists);
	cudaFree(accels);
	cudaFree(accel_sums);

}

void planetFill () {

	int i, j;
	double data[][7] = {SUN, MERCURY, VENUS, EARTH, MARS, JUPITER, SATURN, URANUS, NEPTUNE};

	for (i = 0; i <= NUMPLANETS;i ++) {
		for (j = 0; j < 3; j++) {
			host_positions[i][j] = data[i][j];
			host_velocities[i][j] = data[i][j+3];
		}
		host_masses[i]=data[i][6];
	}
}

void randomFill (int start, int count) {

	int i, j = start;

	for (i = start; i < start + count; i++) {
		for (j = 0; j < 3; j++) {
			host_velocities[i][j] = (double)rand() / RAND_MAX * MAX_DISTANCE * 2 - MAX_DISTANCE;
			host_positions[i][j] = (double)rand() / RAND_MAX * MAX_VELOCITY * 2 - MAX_VELOCITY;
			host_masses[i] = (double)rand() / RAND_MAX * MAX_MASS;
		}
	}
}

void printSystem(FILE* handle){

	int i, j;

	for (i = 0; i < NUMENTITIES; i++) {

		fprintf(handle, "pos=(");
		for (j = 0; j < 3; j++) {
			fprintf(handle, "%lf,", host_positions[i][j]);
		}

		printf("),v=(");
		for (j = 0; j < 3; j++) {
			fprintf(handle, "%lf,", host_velocities[i][j]);
		}

		fprintf(handle,"),m=%lf\n",host_masses[i]);
	}
}

int main(int argc, char **argv)
{
	clock_t t0 = clock();
	int t_now;

	srand(1234);
	initHostMemory(NUMENTITIES);
	initDeviceMemory(NUMENTITIES);
	planetFill();
	randomFill(NUMPLANETS + 1, NUMASTEROIDS);
	copyHostToDevice(NUMENTITIES);

	#ifdef DEBUG
	printSystem(stdout);
	#endif

	for (t_now=0;t_now<DURATION;t_now+=INTERVAL) {
		compute();
	}

	copyDeviceToHost(NUMENTITIES);

	clock_t t1 = clock() - t0;

	#ifdef DEBUG
	printSystem(stdout);
	#endif

	printf("This took a total time of %f seconds\n",(double)t1/CLOCKS_PER_SEC);

	freeHostMemory();
}
