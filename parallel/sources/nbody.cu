#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "compute.cuh"
#include "config.cuh"
#include "kernel_utils.cuh"
#include "memory_utils.cuh"
#include "planets.cuh"
#include "vector.cuh"

void handleCudaError (cudaError_t e, char* identifier, bool exits = true) {

	if (e == cudaSuccess) return;

	fprintf(stderr, "%s %s: %s\n",
		identifier,
		cudaGetErrorName(e),
		cudaGetErrorString(e)
	);

	fflush(NULL);

	if (exits) exit(1);
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

void printSystem (FILE* handle){

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

void printSystemAlt (FILE* handle) {
	
	int i, j;

	for (i = 0; i < NUMENTITIES; i++) {

		fprintf(handle, "pos=(");
		for (j = 0; j < 3; j++) {
			fprintf(handle, "%1.5e,", host_positions[i][j]);
		}

		printf("),v=(");
		for (j = 0; j < 3; j++) {
			fprintf(handle, "%1.5e,", host_velocities[i][j]);
		}

		fprintf(handle,"),m=%1.5e\n", host_masses[i]);
	}
}

int main(int argc, char **argv)
{
	clock_t t0 = clock();
	int t_now;

	srand(1234);
	initCalcChangesDims();
	initCalcAccelsDims();
	initHostMemory();
	initDeviceMemory();
	planetFill();
	randomFill(NUMPLANETS + 1, NUMASTEROIDS);
	copyHostToDevice();

	#ifdef DEBUG
	printSystem(stdout);
	#endif

	for (t_now=0;t_now<DURATION;t_now+=INTERVAL) {
		compute();
	}

	copyDeviceToHost();

	clock_t t1 = clock() - t0;

	/**
	 * 
	 * 	#ifdef DEBUG
	 *	printSystem(stdout);
	 *	#endif
	 * 
	 * Below is different from the original file. The original file's
	 * code is kept above for documentation purposes.
	 */

	#ifdef DEBUG
		#ifdef ALT_PRINT_SYSTEM
		printSystemAlt(stdout);
		#else
		printSystem(stdout);
		#endif
	#endif

	printf("This took a total time of %f seconds\n", (double)(t1) / CLOCKS_PER_SEC);

	freeHostMemory();
	freeDeviceMemory();
}
