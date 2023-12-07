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

/**
 * Prints the CUDA error if it exists, then exits if required.
 * 
 * @param e The possible CUDA error to be printed.
 * @param identifier A short name to prefix the message.
 * @param exits Whether or not the function should exit the program if it encounters an error.
 */
void handleCudaError (cudaError_t e, const char* identifier, bool exits = true) {

	if (e == cudaSuccess) return;

	fprintf(stdout, "%s %s: %s\n",
		identifier,
		cudaGetErrorName(e),
		cudaGetErrorString(e)
	);

	fflush(NULL);

	if (exits) exit(1);
}

// planetFill: Fill the first NUMPLANETS+1 entries of the entity arrays with an estimation
//				of our solar system (Sun+NUMPLANETS)
// Parameters: None
// Returns: None
// Fills the first 8 entries of our system with an estimation of the sun plus our 8 planets.
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

// randomFill: FIll the rest of the objects in the system randomly starting at some entry in the list
// Parameters: 	start: The index of the first open entry in our system (after planetFill).
//				count: The number of random objects to put into our system
// Returns: None
// Side Effects: Fills count entries in our system starting at index start (0 based)
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

// printSystem: Prints out the entire system to the supplied file
// Parameters: 	handle: A handle to an open file with write access to prnt the data to
// Returns: 		none
// Side Effects: Modifies the file handle by writing to it.
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

/**
 * Prints the entire system in scientific notation to the supplied file.
 */
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

int main(int argc, char **argv) {

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
