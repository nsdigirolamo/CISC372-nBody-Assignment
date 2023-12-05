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
		grid_width = (NUMENTITIES + (CALC_ACCELS_BLOCK_WIDTH - 1)) / CALC_ACCELS_BLOCK_WIDTH;
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
		grid_height = (NUMENTITIES + (MAX_THREADS_PER_BLOCK - 1)) / MAX_THREADS_PER_BLOCK;
	}

	int block_height = (NUMENTITIES + (grid_height - 1)) / grid_height;
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

	int pows_of_two[6] = {32, 64, 128, 256, 512, 1024};

	int halved_entity_count = (entity_count + 1) / 2;

	int grid_width = MINIMUM_DIM_SIZE;
	if (MAX_THREADS_PER_BLOCK < halved_entity_count) {
		grid_width = (halved_entity_count + (MAX_THREADS_PER_BLOCK - 1)) / MAX_THREADS_PER_BLOCK;
	}

	int block_width = (halved_entity_count + (grid_width - 1)) / grid_width;
	for (int i = 0; i < 6; i++) {
		if (block_width < pows_of_two[i]) {
			block_width = pows_of_two[i];
			break;
		}
	}

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
