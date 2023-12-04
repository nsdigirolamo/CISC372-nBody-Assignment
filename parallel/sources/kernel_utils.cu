#include <stdio.h>

#include "config.cuh"

dim3 calc_changes_grid_dims;
dim3 calc_changes_block_dims;

dim3 calc_accels_grid_dims;
dim3 calc_accels_block_dims;

dim3 sum_accels_grid_dims;
dim3 sum_accels_block_dims;

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

void setSumAccelsDims (int entity_count) {

	int grid_width = 1;

	// Divide entity_count by 2 because each thread will handle two entities.
	// Then, we add one if halved_entity_count is uneven to make sure we have
	// enough threads. Then, we make sure halved_entity_count is a power of
	// 2 to make sure our reduction algorithm works properly.

	int halved_entity_count = entity_count / 2;
	if (entity_count % 2) halved_entity_count += 1;
	halved_entity_count = pow(2, (int)ceil(log2(halved_entity_count)));

	if (MAX_THREADS_PER_BLOCK < halved_entity_count) {
		grid_width = halved_entity_count / MAX_THREADS_PER_BLOCK;
		if (halved_entity_count % MAX_THREADS_PER_BLOCK) grid_width += 1;
	}

	int block_width = halved_entity_count / grid_width;
	if (halved_entity_count % grid_width) block_width += 1;
	int warp_offset = block_width % WARP_SIZE;
	if (warp_offset) block_width += (WARP_SIZE - warp_offset);

	sum_accels_grid_dims.x = grid_width;
	sum_accels_grid_dims.y = entity_count;
	sum_accels_grid_dims.z = SPATIAL_DIMS;

	sum_accels_block_dims.x = block_width;
	sum_accels_block_dims.y = MINIMUM_DIM_SIZE;
	sum_accels_block_dims.z = MINIMUM_DIM_SIZE;

	#ifdef KERNEL_UTILS_DEBUG
	printKernelDims("setSumAccelsDims", sum_accels_grid_dims, sum_accels_block_dims);
	#endif
}
