#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>

#include "vector.cuh"
#include "config.cuh"
#include "planets.cuh"
#include "compute.cuh"
#include "memory_utils.cuh"

// Kernel Config Arguments

dim3 calc_changes_grid_dims;
dim3 calc_changes_block_dims;

dim3 calc_accels_grid_dims;
dim3 calc_accels_block_dims;

dim3 sum_accels_grid_dims;
dim3 sum_accels_block_dims;

void initCalcAccelsDims () {

	int grid_width = MINIMUM_DIM_SIZE;

	if (CALC_ACCELS_BLOCK_WIDTH < NUMENTITIES) {
		grid_width = NUMENTITIES / CALC_ACCELS_BLOCK_WIDTH;
		if (NUMENTITIES % CALC_ACCELS_BLOCK_WIDTH) grid_width += 1;
	}

	calc_accels_grid_dims.x = grid_width;
	calc_accels_grid_dims.y = grid_width;
	calc_accels_grid_dims.z = MINIMUM_DIM_SIZE;

	calc_accels_block_dims.x = CALC_ACCELS_BLOCK_WIDTH;
	calc_accels_block_dims.y = CALC_ACCELS_BLOCK_WIDTH;
	calc_accels_block_dims.z = SPATIAL_DIMS;

	#ifdef KERNEL_ARGS_DEBUG
	printf("initCalcAccelsDims(): gridDims: {%d, %d, %d}, blockDims: {%d, %d, %d}\n",
		calc_accels_grid_dims.x,
		calc_accels_grid_dims.y,
		calc_accels_grid_dims.z,
		calc_accels_block_dims.x,
		calc_accels_block_dims.y,
		calc_accels_block_dims.z
	);
	#endif
}

void initCalcChangesDims () {

	int grid_height = MINIMUM_DIM_SIZE;

	if (MAX_THREADS_PER_BLOCK < NUMENTITIES) {
		grid_height = NUMENTITIES / MAX_THREADS_PER_BLOCK;
		if (NUMENTITIES % MAX_THREADS_PER_BLOCK) grid_height += 1;
	}

	int block_height = NUMENTITIES / grid_height;
	if (NUMENTITIES % grid_height) block_height += 1;
	int warp_offset = block_height % WARP_SIZE;
	if (warp_offset) block_height += (WARP_SIZE - warp_offset);

	calc_changes_grid_dims.x = MINIMUM_DIM_SIZE;
	calc_changes_grid_dims.y = grid_height;
	calc_changes_grid_dims.z = SPATIAL_DIMS;

	calc_changes_block_dims.x = MINIMUM_DIM_SIZE;
	calc_changes_block_dims.y = block_height;
	calc_changes_block_dims.z = MINIMUM_DIM_SIZE;

	#ifdef KERNEL_ARGS_DEBUG
	printf("initCalcChangesDims(): gridDims: {%d, %d, %d}, blockDims: {%d, %d, %d}\n",
		calc_changes_grid_dims.x,
		calc_changes_grid_dims.y,
		calc_changes_grid_dims.z,
		calc_changes_block_dims.x,
		calc_changes_block_dims.y,
		calc_changes_block_dims.z
	);
	#endif
}

void setSumAccelsDims (int entity_count, dim3* grid_dims, dim3* block_dims) {

	int grid_width = 1;

	if (MAX_THREADS_PER_BLOCK < entity_count) {
		grid_width = entity_count / MAX_THREADS_PER_BLOCK;
		if (entity_count % MAX_THREADS_PER_BLOCK) grid_width += 1;
	}

	int block_width = entity_count / grid_width;
	if (entity_count % grid_width) block_width += 1;
	int warp_offset = block_width % WARP_SIZE;
	if (warp_offset) block_width += (WARP_SIZE - warp_offset);

	grid_dims->x = grid_width;
	grid_dims->y = entity_count;
	grid_dims->z = SPATIAL_DIMS;

	block_dims->x = block_width;
	block_dims->y = MINIMUM_DIM_SIZE;
	block_dims->z = MINIMUM_DIM_SIZE;

	#ifdef KERNEL_ARGS_DEBUG
	printf("initSumAccelsDims(): gridDims: {%d, %d, %d}, blockDims: {%d, %d, %d}\n",
		calc_changes_grid_dims.x,
		calc_changes_grid_dims.y,
		calc_changes_grid_dims.z,
		calc_changes_block_dims.x,
		calc_changes_block_dims.y,
		calc_changes_block_dims.z
	);
	#endif
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
