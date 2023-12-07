#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "config.cuh"
#include "kernel_utils.cuh"
#include "memory_utils.cuh"
#include "nbody.cuh"
#include "vector.cuh"

/**
 * Calculates the acceleration values for each entity in the system.
 * 
 * Each thread is given a single vector within the accels array, and then
 * calculates the acceleration for a single component of that vector.
 * 
 * @param accels A device pointer to the accels array. Allocated using cudaMallocPitch.
 * @param accels_pitch The pitch of accels. Given when allocating using cudaMallocPitch.
 * @param positions A device pointer to the positions array.
 * @param masses A device pointer to the masses array.
 */
__global__ void calcAccels (vector3* accels, size_t accels_pitch, vector3* positions, double* masses) {

	int local_row = threadIdx.y;
	int local_col = threadIdx.x;

	int global_row = (blockIdx.y * blockDim.y) + local_row;
	int global_col = (blockIdx.x * blockDim.x) + local_col;
	int spatial_axis = threadIdx.z;

	if (NUMENTITIES <= global_col || NUMENTITIES <= global_row) return;

	vector3* accels_row = (vector3*)((char*)(accels) + global_row * accels_pitch);

	if (global_row == global_col) {

		accels_row[global_col][spatial_axis] = 0;

	} else {

		__shared__ vector3 distances[CALC_ACCELS_BLOCK_WIDTH][CALC_ACCELS_BLOCK_WIDTH];

		distances[local_row][local_col][spatial_axis] = positions[global_row][spatial_axis] - positions[global_col][spatial_axis];

		__syncthreads();

		double magnitude_sq = distances[local_row][local_col][0] * distances[local_row][local_col][0] + distances[local_row][local_col][1] * distances[local_row][local_col][1] + distances[local_row][local_col][2] * distances[local_row][local_col][2];
		double magnitude = sqrt(magnitude_sq);
		double accelmag = -1 * GRAV_CONSTANT * masses[global_col] / magnitude_sq;
		accels_row[global_col][spatial_axis] = accelmag * distances[local_row][local_col][spatial_axis] / magnitude;
	}
}

/**
 * Sums the acceleration values for each entity in the system via parallel
 * reduction.
 * 
 * Each thread initially takes sums two values from the global accels array, and
 * then stores that sum in the thread's spot in the shared sums array. The
 * threads iterate until the reduction offset reaches zero. Once the reduction
 * step is complete, the 0th thread of the ith block will place its block's sum
 * into accels[i]. The below slide deck was very helpful:
 * https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf
 * 
 * @param accels A device pointer to the accels array. Allocated with cudaMallocPitch.
 * @param accels_pitch The pitch of accels. Given when allocating with cudaMallocPitch.
 * @param entity_count The maximum number of entities to sum from a row of accels.
 * @param sum_length The length of the kernel's shared sums array. Must be a power of 2.
 */
__global__ void sumAccels (vector3* accels, size_t accels_pitch, int entity_count, int sum_length) {

	int local_col = threadIdx.x;

	int global_row = blockIdx.y;
	int global_col = (blockIdx.x * blockDim.x * 2) + (local_col * 2);
	int spatial_axis = blockIdx.z;

	extern __shared__ double sums[];

	if (sum_length <= local_col) return;

	// If we don't set all values to 0 some threads may try to do math on
	// uninitialized memory.

	sums[local_col] = 0;

	if (entity_count <= global_col) return; 

	vector3* accels_row = (vector3*)((char*)(accels) + global_row * accels_pitch);

	// Each thread takes two values from the global array and sums them first.

	if (entity_count <= global_col + 1) {

		sums[local_col] = accels_row[global_col][spatial_axis];

	} else {

		sums[local_col] = accels_row[global_col][spatial_axis] + accels_row[global_col + 1][spatial_axis];

	}

	__syncthreads();

	// This is where the reduction happens. During this step, it's important
	// that sum_length is a power of two. The offset will start as half the
	// sum_length, and then will decrease by half at each iteration. If
	// sum_length does not start as a power of two, the reduction will sometimes
	// skip over an index that has a value that needs to be summed.

	for (int offset = sum_length / 2; 0 < offset; offset >>= 1) {

		if (local_col < offset) sums[local_col] += sums[local_col + offset];
		__syncthreads();

	}

	if (local_col == 0) accels_row[blockIdx.x][spatial_axis] = sums[local_col];
}

/**
 * Calculates the changes to velocities and positions for each entity in the
 * system.
 * 
 * Each thread is given a single entity, and then calculates the change in
 * velocity and position for a single component of that entity.
 * 
 * @param accels A device pointer to the accels array. Allocated with cudaMallocPitch.
 * @param accels_pitch The pitch of accels. Given when allocating with cudaMallocPitch.
 * @param velocities A device pointer to the velocities array.
 * @param positions A device pointer to the positions array.
 */
__global__ void calcChanges (vector3* accels, size_t accels_pitch, vector3* velocities, vector3* positions) {

	int global_row = (blockIdx.y * blockDim.y) + threadIdx.y;
	int spatial_axis = blockIdx.z;

	if (NUMENTITIES <= global_row) return;

	vector3* accels_row = (vector3*)((char*)(accels) + global_row * accels_pitch);

	velocities[global_row][spatial_axis] += accels_row[0][spatial_axis] * INTERVAL;
	positions[global_row][spatial_axis] += velocities[global_row][spatial_axis] * INTERVAL; 
}

/**
 * Computes the changes to velocity and position for each entity in the system.
 */
void compute () {

	#ifdef DEBUG
	cudaError_t e = cudaGetLastError();
	handleCudaError(e, "compute filter");
	#endif

	// Calculate Accelerations

	calcAccels<<<calc_accels_grid_dims, calc_accels_block_dims>>>(accels, accels_pitch, device_positions, device_masses);

	#ifdef DEBUG
	e = cudaDeviceSynchronize();
	if (e != cudaSuccess) {
		printf("Error in Kernel Detected!\n");
		printKernelDims("calcAccels", calc_accels_grid_dims, calc_accels_block_dims);
	}
	handleCudaError(e, "calcAccels");
	#endif

	// Sum Accelerations

	int entity_count = NUMENTITIES;
	setSumAccelsDims(entity_count);

	while (1 < entity_count) {

		sumAccels<<<sum_accels_grid_dims, sum_accels_block_dims, sizeof(double) * sum_accels_block_dims.x>>>(accels, accels_pitch, entity_count, sum_accels_block_dims.x);

		#ifdef DEBUG
		e = cudaDeviceSynchronize();
		if (e != cudaSuccess) {
			printf("Error in Kernel Detected!\n");
			printKernelDims("sumAccels", sum_accels_grid_dims, sum_accels_block_dims);
		}
		handleCudaError(e, "sumAccels");
		#endif

		entity_count = sum_accels_grid_dims.x;
		setSumAccelsDims(entity_count);
	}

	// Calculating Changes

	calcChanges<<<calc_changes_grid_dims, calc_changes_block_dims>>>(accels, accels_pitch, device_velocities, device_positions);

	#ifdef DEBUG
	e = cudaDeviceSynchronize();
	if (e != cudaSuccess) {
		printf("Error in Kernel Detected!\n");
		printKernelDims("calcChanges", calc_changes_grid_dims, calc_changes_block_dims);
	}
	handleCudaError(e, "calcChanges");
	#endif
}
