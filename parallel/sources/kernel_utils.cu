#include <stdio.h>

#include "config.cuh"

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

	#ifdef KERNEL_UTILS_DEBUG
	printKernelDims("initCalcAccelsDims", calc_accels_grid_dims, calc_accels_block_dims);
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

	#ifdef KERNEL_UTILS_DEBUG
	printKernelDims("initCalcChangesDims", calc_changes_grid_dims, calc_changes_block_dims);
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

	#ifdef KERNEL_UTILS_DEBUG
	printKernelDims("setSumAccelsDims", grid_dims, block_dims);
	#endif
}

void printKernelDims (const char* identifier, dim3 grid_dims, dim3 block_dims) {
	printf("%s: gridDims: {%d %d %d}, blockDims: {%d %d %d}\n",
		identifier,
		grid_dims.x,
		grid_dims.y,
		grid_dims.z,
		block_dims.x,
		block_dims.y,
		block_dims.z
	);
}